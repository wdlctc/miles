"""CUDA-only, standalone TE precision experiment; not a Miles RL launcher.

Compares BF16, MXFP8, NVFP4 and NVFP4-forward/MXFP8-backward on the same
inputs. The hybrid uses dequantized forward operands for backward, but lacks
MXFP8 stochastic rounding and Thinking Machines' proprietary MoE kernels.
This is an experimental SFT benchmark, not an implementation of their stack.

Example (see the experiment report for pinned dependencies):
    python tools/benchmark_blackwell_posttraining.py --task prepare --work-dir /workspace/experiment
    python tools/benchmark_blackwell_posttraining.py --task sft --mode hybrid --work-dir /workspace/experiment
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch
import transformer_engine.pytorch as te
from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download
from transformer_engine.common.recipe import CustomRecipe, MXFP8BlockScaling, NVFP4BlockScaling
from transformer_engine.pytorch.constants import DType
from transformer_engine.pytorch.custom_recipes.quantizer_factories import mxfp8_factory
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
from transformers import AutoModelForCausalLM, AutoTokenizer


class _BF16InputLinear(te.Linear):
    """Keep FP32 optimizer weights while meeting NVFP4 RHT's BF16 input contract."""

    def forward(self, inputs, *args, **kwargs):
        return super().forward(inputs.to(torch.bfloat16), *args, **kwargs)


def _hybrid_factory(role):
    if role is not None and role.tensor_type in ("input", "weight"):
        return te.HybridQuantizer(
            rowwise_quantizer=NVFP4Quantizer(
                fp4_dtype=DType.kFloat4E2M1,
                with_rht=False,
                with_post_rht_amax=False,
                with_2d_quantization=False,
                stochastic_rounding=False,
            ),
            columnwise_quantizer=mxfp8_factory(role),
            columnwise_source="rowwise_dequantized",
        )
    return mxfp8_factory(role)


def _recipe(mode):
    return {
        "bf16": lambda: None,
        "mxfp8": MXFP8BlockScaling,
        "nvfp4": NVFP4BlockScaling,
        "nvfp4-1d": lambda: NVFP4BlockScaling(disable_2d_quantization=True),
        "hybrid": lambda: CustomRecipe(qfactory=_hybrid_factory),
    }[mode]()


def _prepare(root):
    api = HfApi()
    model_revision = api.model_info("Qwen/Qwen3-0.6B").sha
    dataset_revision = api.dataset_info("openai/gsm8k").sha
    snapshot_download(
        "Qwen/Qwen3-0.6B",
        revision=model_revision,
        local_dir=root / "model",
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
    )
    dataset = load_dataset("openai/gsm8k", "main", revision=dataset_revision)
    for name, split, count in (("train", "train", 128), ("validation", "test", 16)):
        (root / f"{name}.json").write_text(json.dumps([dataset[split][i] for i in range(count)]))
    metadata = dict(
        model="Qwen/Qwen3-0.6B",
        model_revision=model_revision,
        dataset="openai/gsm8k",
        dataset_revision=dataset_revision,
        train_samples=128,
        validation_samples=16,
    )
    (root / "inputs.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def _batches(root, split, batch_size, sequence_length):
    tokenizer = AutoTokenizer.from_pretrained(root / "model")
    records = json.loads((root / f"{split}.json").read_text())
    rows = []
    for item in records:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": item["question"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        answer = tokenizer.encode(item["answer"] + tokenizer.eos_token, add_special_tokens=False)
        ids = (prefix + answer)[:sequence_length]
        labels = ([-100] * len(prefix) + answer)[:sequence_length]
        padding = sequence_length - len(ids)
        if not any(label != -100 for label in labels[1:]):
            raise ValueError("Sequence length truncates every answer token")
        rows.append(
            dict(
                input_ids=ids + [tokenizer.pad_token_id] * padding,
                attention_mask=[1] * len(ids) + [0] * padding,
                labels=labels + [-100] * padding,
            )
        )
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    batches = [
        {key: torch.tensor([row[key] for row in rows[i : i + batch_size]], device="cuda") for key in rows[0]}
        for i in range(0, len(rows), batch_size)
    ]
    return batches, digest


def _replace_mlp(model):
    # Last ceil(1/6) of layers, attention, embedding and LM head stay BF16 compute.
    layers = model.model.layers
    selected = len(layers) - math.ceil(len(layers) / 6)
    names = []
    for i, layer in enumerate(layers[:selected]):
        for name in ("gate_proj", "up_proj", "down_proj"):
            original = getattr(layer.mlp, name)
            full_name = f"model.layers.{i}.mlp.{name}"
            replacement = _BF16InputLinear(
                original.in_features,
                original.out_features,
                bias=original.bias is not None,
                params_dtype=torch.float32,
                device="cuda",
                name=full_name,
            )
            with torch.no_grad():
                replacement.weight.copy_(original.weight)
                if original.bias is not None:
                    replacement.bias.copy_(original.bias)
            setattr(layer.mlp, name, replacement)
            names.append(full_name)
    return names


def _forward(model, batch, recipe, enabled):
    with torch.autocast("cuda", dtype=torch.bfloat16), te.autocast(enabled=enabled, recipe=recipe):
        return model(**batch, use_cache=False).loss


@torch.no_grad()
def _evaluate(model, batches, recipe, enabled):
    model.eval()
    weighted_loss, tokens = 0.0, 0
    for batch in batches:
        count = int((batch["labels"][:, 1:] != -100).sum())
        weighted_loss += float(_forward(model, batch, recipe, enabled)) * count
        tokens += count
    model.train()
    return weighted_loss / tokens


def _sft(args, recipe):
    root = args.work_dir
    train, train_hash = _batches(root, "train", args.batch_size, args.sequence_length)
    validation, val_hash = _batches(root, "validation", args.batch_size, args.sequence_length)
    model = AutoModelForCausalLM.from_pretrained(root / "model", dtype=torch.float32, attn_implementation="sdpa").to(
        "cuda"
    )
    names = _replace_mlp(model)
    enabled = args.mode != "bf16"
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, fused=True)
    reference = model.model.layers[0].mlp.up_proj.weight.detach().clone()
    before = _evaluate(model, validation, recipe, enabled)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        _forward(model, train[0], recipe, enabled).backward()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    records = []
    for step in range(args.steps):
        batch = train[step % len(train)]
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        loss = _forward(model, batch, recipe, enabled)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        record = dict(step=step + 1, loss=float(loss.detach()), grad_norm=float(grad_norm), seconds=elapsed)
        if not math.isfinite(record["loss"]):
            raise FloatingPointError(record)
        records.append(record)
        print(json.dumps(record), flush=True)
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    timed = records[5:]
    delta = float((model.model.layers[0].mlp.up_proj.weight.detach() - reference).norm())
    if not timed or delta <= 0:
        raise AssertionError("Need more than 5 measured steps and changed model weights")
    after = _evaluate(model, validation, recipe, enabled)
    after_bf16 = _evaluate(model, validation, None, False)
    result = dict(
        inputs=json.loads((root / "inputs.json").read_text()),
        train_sha256=train_hash,
        validation_sha256=val_hash,
        te_modules=names,
        records=records,
        validation_loss_before=before,
        validation_loss_after=after,
        validation_loss_after_bf16=after_bf16,
        parameter_delta_l2=delta,
        peak_allocated_gib=peak_gib,
        median_step_ms=statistics.median(r["seconds"] for r in timed) * 1000,
        padded_tokens_per_second=args.batch_size * args.sequence_length / statistics.mean(r["seconds"] for r in timed),
        actual_tokens_per_second=sum(int(train[(r["step"] - 1) % len(train)]["attention_mask"].sum()) for r in timed)
        / sum(r["seconds"] for r in timed),
    )
    if args.save_checkpoint:
        # Native HF models do not consume TE's quantizer state (which can itself be a tensor).
        model.to(torch.bfloat16)
        state = {
            key: value
            for key, value in model.state_dict().items()
            if isinstance(value, torch.Tensor) and not key.endswith("._extra_state")
        }
        destination = root / "checkpoints" / _result_name(args)
        model.save_pretrained(destination, state_dict=state)
        AutoTokenizer.from_pretrained(root / "model").save_pretrained(destination)
        result["checkpoint"] = str(destination)
    return result


def _micro(args, recipe):
    options = dict(bias=False, params_dtype=torch.bfloat16, device="cuda")
    grouped = args.task in ("grouped", "fused-grouped")
    if args.task == "fused-grouped" and os.getenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM") != "1":
        raise RuntimeError("Set NVTE_GROUPED_LINEAR_SINGLE_PARAM=1 to enable the requested packed expert weights")
    if grouped:
        if args.tokens % args.experts:
            raise ValueError("Grouped benchmark requires evenly divisible tokens")
        layer = te.GroupedLinear(
            args.experts,
            args.hidden_size,
            args.intermediate_size,
            use_grouped_tensor=args.task == "fused-grouped",
            single_grouped_weight=args.task == "fused-grouped",
            **options,
        )
    else:
        layer = te.Linear(args.hidden_size, args.intermediate_size, **options)
    x = torch.randn(args.tokens, args.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn(args.tokens, args.intermediate_size, device="cuda", dtype=torch.bfloat16)
    splits = [args.tokens // args.experts] * args.experts if grouped else None
    if args.task == "fused-grouped":
        # Native fused kernels consume GPU-resident expert row counts.
        splits = torch.tensor(splits, dtype=torch.int64, device="cuda")

    def iteration():
        layer.zero_grad(set_to_none=True)
        x.grad = None
        with te.autocast(enabled=args.mode != "bf16", recipe=recipe):
            y = layer(x, splits) if grouped else layer(x)
        y.backward(dy)

    for _ in range(10):
        iteration()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    milliseconds = []
    for _ in range(args.steps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        iteration()
        end.record()
        end.synchronize()
        milliseconds.append(start.elapsed_time(end))
    finite = bool(torch.isfinite(x.grad).all()) and all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in layer.parameters()
    )
    if not finite:
        raise FloatingPointError("Nonfinite microbenchmark gradients")
    result = dict(
        median_step_ms=statistics.median(milliseconds),
        times_ms=milliseconds,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        finite_gradients=finite,
    )
    if grouped:
        result.update(use_grouped_tensor=layer.use_grouped_tensor, single_grouped_weight=layer.single_grouped_weight)
    if args.profile:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profiler:
            iteration()
            torch.cuda.synchronize()
        trace = args.work_dir / "results" / f"{_result_name(args)}-trace.json"
        profiler.export_chrome_trace(str(trace))
        result["trace"] = str(trace)
        result["cuda_kernels"] = sorted(
            {event.name for event in profiler.events() if event.device_type == torch.autograd.DeviceType.CUDA}
        )
    return result


def _result_name(args):
    suffix = f"-{args.run_label}" if args.run_label else ""
    return f"{args.task}-{args.mode}{suffix}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("prepare", "micro", "grouped", "fused-grouped", "sft"), required=True)
    parser.add_argument("--mode", choices=("bf16", "mxfp8", "nvfp4", "nvfp4-1d", "hybrid"), default="bf16")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=14336)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--run-label", default="")
    args = parser.parse_args()
    (args.work_dir / "results").mkdir(parents=True, exist_ok=True)
    if args.task == "prepare":
        print(json.dumps(_prepare(args.work_dir), indent=2))
        return
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError("This benchmark requires an NVIDIA Blackwell CUDA GPU")
    torch.manual_seed(20261115)
    torch.set_num_threads(8)
    started = time.time()
    result = (_sft if args.task == "sft" else _micro)(args, _recipe(args.mode))
    result.update(
        config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        gpu=torch.cuda.get_device_name(),
        cuda=torch.version.cuda,
        versions={p: importlib.metadata.version(p) for p in ("torch", "transformer-engine", "transformers")},
        start_unix=started,
        elapsed_seconds=time.time() - started,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    output = args.work_dir / "results" / f"{_result_name(args)}.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"RESULT {output}", flush=True)


if __name__ == "__main__":
    main()
