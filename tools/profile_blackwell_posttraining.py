"""CUDA-only diagnostics for the recorded Blackwell post-training experiment.

Run as a module from the repository root. Reuses the original model adapter,
data preparation and quantization recipes; leaves the original results intact.

    python -m tools.profile_blackwell_posttraining --task sft --mode mxfp8 \
        --work-dir /workspace/miles-blackwell-experiment
"""

import argparse
import functools
import hashlib
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
import transformer_engine.pytorch as te
from transformers import AutoModelForCausalLM
from transformers.masking_utils import create_causal_mask

from tools import benchmark_blackwell_posttraining as benchmark


def _kernel_category(name):
    lower = name.lower()
    if "mxfp8" in lower and "quantize" in lower:
        return "mxfp8_quantization"
    if "nvfp4" in lower or "fp4" in lower or "hadamard" in lower or "rht_gemm" in lower:
        return "nvfp4_quantization_rht"
    if "nvjet" in lower or "gemm" in lower or "cutlass" in lower:
        return "gemm"
    if "flash" in lower or "fmha" in lower or "attention" in lower:
        return "attention"
    if "adam" in lower:
        return "optimizer"
    return "other"


def _trace_summary(path):
    events = json.loads(path.read_text())["traceEvents"]
    kernels = [event for event in events if event.get("cat") == "kernel"]
    categories, by_name = defaultdict(lambda: [0, 0.0]), defaultdict(lambda: [0, 0.0])
    for event in kernels:
        for counter, key in ((categories, _kernel_category(event["name"])), (by_name, event["name"])):
            counter[key][0] += 1
            counter[key][1] += event["dur"] / 1000
    intervals = sorted((event["ts"], event["ts"] + event["dur"]) for event in kernels)
    busy, end = 0.0, float("-inf")
    for left, right in intervals:
        busy += max(0.0, right - max(left, end))
        end = max(end, right)
    span = (max(right for _, right in intervals) - intervals[0][0]) if intervals else 0
    return dict(
        kernel_count=len(kernels),
        kernel_sum_ms=sum(event["dur"] for event in kernels) / 1000,
        kernel_union_ms=busy / 1000,
        first_to_last_kernel_span_ms=span / 1000,
        no_kernel_in_span_ms=(span - busy) / 1000,
        categories={key: dict(count=value[0], total_ms=value[1]) for key, value in categories.items()},
        top_kernels=[
            dict(name=key, count=value[0], total_ms=value[1])
            for key, value in sorted(by_name.items(), key=lambda item: -item[1][1])[:25]
        ],
    )


def _profile(function, path, iterations=3):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as profiler:
        for _ in range(iterations):
            function()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(path))
    scopes = []
    operators = []
    for event in profiler.key_averages():
        row = dict(
            name=event.key,
            device_type=str(event.device_type),
            count=event.count,
            cpu_total_ms=event.cpu_time_total / 1000,
            self_cpu_ms=event.self_cpu_time_total / 1000,
            device_total_ms=event.device_time_total / 1000,
            self_device_ms=event.self_device_time_total / 1000,
        )
        (scopes if event.key.startswith("diag/") else operators).append(row)
    return dict(
        trace=str(path),
        iterations=iterations,
        cuda=_trace_summary(path),
        scopes=scopes,
        top_cpu_operators=sorted(operators, key=lambda row: -row["self_cpu_ms"])[:25],
    )


def _annotate_module(module, name):
    original = module.forward

    @functools.wraps(original)
    def forward(*args, **kwargs):
        with torch.profiler.record_function(name):
            return original(*args, **kwargs)

    module.forward = forward


def _setup_sft(args):
    train, digest = benchmark._batches(args.work_dir, "train", args.batch_size, 512)
    batch = train[0]
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.work_dir / "model",
            dtype=torch.float32,
            attn_implementation="sdpa",
        )
        .to("cuda")
        .train()
    )
    names = benchmark._replace_mlp(model)
    recipe = benchmark._recipe(args.mode)
    enabled = args.mode != "bf16"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=True)
    return model, batch, digest, names, recipe, enabled, optimizer


def _sft(args):
    model, batch, digest, names, recipe, enabled, optimizer = _setup_sft(args)
    if args.task == "sft-static":
        batch = _static_mask_batch(model, batch)

    def step(instrument=False):
        optimizer.zero_grad(set_to_none=True)
        events = [torch.cuda.Event(enable_timing=True) for _ in range(5)] if instrument else []
        if instrument:
            events[0].record()
        with torch.profiler.record_function("diag/forward"):
            loss = benchmark._forward(model, batch, recipe, enabled)
        if instrument:
            events[1].record()
        with torch.profiler.record_function("diag/backward"):
            loss.backward()
        if instrument:
            events[2].record()
        with torch.profiler.record_function("diag/clip"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        if instrument:
            events[3].record()
        with torch.profiler.record_function("diag/optimizer"):
            optimizer.step()
        if instrument:
            events[4].record()
        return loss, events

    for _ in range(8):
        step()
    torch.cuda.synchronize()
    records = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        loss, events = step(instrument=True)
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - started) * 1000
        records.append(
            dict(
                wall_ms=wall_ms,
                loss=float(loss.detach()),
                **{
                    key: events[i].elapsed_time(events[i + 1])
                    for i, key in enumerate(("forward_ms", "backward_ms", "clip_ms", "optimizer_ms"))
                },
            )
        )
    # Add component scopes only for the separate profiler run, not latency measurements.
    for layer in model.model.layers:
        _annotate_module(layer.mlp, "diag/te_mlp" if isinstance(layer.mlp.up_proj, te.Linear) else "diag/bf16_mlp")
        _annotate_module(layer.self_attn, "diag/attention")
    _annotate_module(model.lm_head, "diag/lm_head")
    profile = _profile(step, args.output_dir / f"{args.task}-{args.mode}-b{args.batch_size}-trace.json")
    return dict(
        records=records,
        medians={key: statistics.median(row[key] for row in records) for key in records[0]},
        profile=profile,
        train_sha256=digest,
        te_modules=len(names),
        model_parameter_count=sum(p.numel() for p in model.parameters()),
        config=model.config.to_dict(),
    )


def _static_mask_batch(model, batch):
    with torch.no_grad():
        embeddings = model.model.embed_tokens(batch["input_ids"])
        positions = torch.arange(batch["input_ids"].shape[1], device="cuda").unsqueeze(0)
        mask = create_causal_mask(
            config=model.config,
            inputs_embeds=embeddings,
            attention_mask=batch["attention_mask"],
            past_key_values=None,
            position_ids=positions,
        )
        return dict(batch, attention_mask={"full_attention": mask}, position_ids=positions)


def _sft_graph(args):
    model, batch, digest, names, recipe, enabled, optimizer = _setup_sft(args)
    static_batch = _static_mask_batch(model, batch)
    with torch.no_grad():
        standard_loss = benchmark._forward(model, batch, recipe, enabled)
        static_loss = benchmark._forward(model, static_batch, recipe, enabled)
        torch.testing.assert_close(standard_loss, static_loss, rtol=1e-5, atol=1e-5)

    def forward_backward():
        optimizer.zero_grad(set_to_none=True)
        loss = benchmark._forward(model, static_batch, recipe, enabled)
        loss.backward()
        return loss

    for _ in range(8):
        forward_backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
        graph_loss = forward_backward()
    graph.replay()
    torch.cuda.synchronize()
    parameters = list(model.parameters())
    captured_gradients = [parameter.grad for parameter in parameters]
    probe = model.model.layers[0].mlp.up_proj.weight
    graph_gradient = probe.grad.detach().clone()
    eager_loss = forward_backward()
    gradient_relative_error = float((probe.grad - graph_gradient).norm() / graph_gradient.norm().clamp_min(1e-12))
    torch.testing.assert_close(graph_loss, eager_loss, rtol=1e-4, atol=1e-4)
    if gradient_relative_error > 0.02:
        raise AssertionError(f"Captured/eager gradient mismatch: {gradient_relative_error}")
    # The optimizer must continue reading the buffers written by the captured graph.
    for parameter, gradient in zip(parameters, captured_gradients, strict=True):
        parameter.grad = gradient
    reference = probe.detach().clone()
    records = []
    for _ in range(args.iterations):
        torch.cuda.synchronize()
        started = time.perf_counter()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        records.append(
            dict(
                wall_ms=(time.perf_counter() - started) * 1000,
                forward_backward_ms=start.elapsed_time(end),
                loss=float(graph_loss.detach()),
            )
        )
    delta = float((probe.detach() - reference).norm())
    if delta <= 0 or not torch.isfinite(graph_loss):
        raise AssertionError("Captured graph must update finite model weights")
    return dict(
        records=records,
        medians={key: statistics.median(row[key] for row in records) for key in records[0]},
        parameter_delta_l2=delta,
        gradient_relative_error=gradient_relative_error,
        mask_loss_absolute_error=float((standard_loss - static_loss).abs()),
        train_sha256=digest,
        te_modules=len(names),
        note="Only forward/backward captured; clipping and optimizer remain eager.",
    )


def _linear_case(args, tokens, hidden, intermediate):
    recipe = benchmark._recipe(args.mode)
    layer = te.Linear(hidden, intermediate, params_dtype=torch.float32, device="cuda", bias=False)
    inputs = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    dy = torch.randn(tokens, intermediate, dtype=torch.bfloat16, device="cuda")

    def iteration():
        layer.zero_grad(set_to_none=True)
        inputs.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16), te.autocast(enabled=args.mode != "bf16", recipe=recipe):
            output = layer(inputs)
        output.backward(dy)

    # Warm clocks as well as allocations before comparing eager and captured work.
    started = time.perf_counter()
    while time.perf_counter() - started < 1.0:
        iteration()
        torch.cuda.synchronize()
    samples = []
    for _ in range(args.iterations):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        iteration()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    stem = f"linear-{args.mode}-m{tokens}-k{hidden}-n{intermediate}"
    result = dict(
        shape=[tokens, hidden, intermediate],
        eager_ms=statistics.median(samples),
        eager_samples_ms=samples,
        profile=_profile(iteration, args.output_dir / f"{stem}-trace.json"),
    )
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            iteration()
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
        samples = []
        for _ in range(10):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iterations):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / args.iterations)
        result.update(graph_ms=statistics.median(samples), graph_samples_ms=samples)
    except RuntimeError as error:
        result["graph_error"] = str(error)
    if not torch.isfinite(layer.weight.grad).all() or not torch.isfinite(inputs.grad).all():
        raise FloatingPointError("Nonfinite gradients in linear diagnostic")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--task", choices=("sft", "sft-static", "sft-graph", "linear"), required=True)
    parser.add_argument("--mode", choices=("bf16", "mxfp8", "nvfp4"), required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError("Requires an NVIDIA Blackwell GPU")
    args.output_dir = args.output_dir or args.work_dir / "diagnostics"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261116)
    torch.set_num_threads(8)
    if args.task in ("sft", "sft-static"):
        result = _sft(args)
    elif args.task == "sft-graph":
        # Keep warmup, capture, and the eager gradient check on the same side
        # stream so retained AccumulateGrad nodes do not cross stream boundaries.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            result = _sft_graph(args)
        torch.cuda.current_stream().wait_stream(stream)
    else:
        result = dict(
            cases=[
                _linear_case(args, *shape)
                for shape in (
                    (2048, 1024, 3072),
                    (2048, 3072, 1024),
                    (8192, 1024, 3072),
                    (4096, 4096, 14336),
                )
            ]
        )
    result["arguments"] = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = args.output_dir / f"{args.task}-{args.mode}-b{args.batch_size}.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"RESULT {output}", flush=True)


if __name__ == "__main__":
    main()
