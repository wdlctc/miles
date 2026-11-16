"""Post-training quantization of an exported model, measured against its BF16 reference.

FP8 stores weights and activations as E4M3 with one weight scale per output
channel and one activation scale per token. NVFP4 stores E2M1 values with one
E4M3 scale per 16-element block and one FP32 scale per tensor; its block scale
is chosen by maximum (round-to-nearest), four-over-six or a local-Hessian
search calibrated on the reference model's own activations.

Arithmetic is simulated in floating point: every stored value is exactly
representable in the target format, but no packed low-precision kernel runs.
Fidelity is the teacher-forced KL(reference‖quantized) and top-1 agreement on
held-out sequences. It says nothing about serving speed or memory.

Example:
    python -m miles_platform.quantize --model-dir /models/mdl_x \
        --calibration calib.jsonl --heldout heldout.jsonl --report report.json
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

E4M3_MAX = 448.0
E2M1_MAX = 6.0
BLOCK = 16
GROUPS = {"attention": ("q_proj", "k_proj", "v_proj", "o_proj"), "mlp": ("gate_proj", "up_proj", "down_proj")}
# Block-scale candidates as multiples of max/6. 1.0 maps the block maximum to 6
# and 1.5 maps it to 4; smaller factors clip the maximum for a finer grid.
FACTORS = {
    "max": (1.0,),
    "four_over_six": (1.0, 1.5),
    "local_hessian": (0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6),
}


@dataclass(frozen=True)
class Recipe:
    format: str  # "fp8" or "nvfp4"
    method: str = "max"
    groups: tuple = ("attention", "mlp")
    activations: bool = True

    def __post_init__(self):
        if self.format not in {"fp8", "nvfp4"}:
            raise ValueError("format must be fp8 or nvfp4")
        if self.method not in FACTORS or (self.format == "fp8" and self.method != "max"):
            raise ValueError("fp8 supports max; nvfp4 supports max, four_over_six and local_hessian")
        if not self.groups or set(self.groups) - set(GROUPS):
            raise ValueError("groups must be chosen from attention and mlp")

    @property
    def name(self):
        return "-".join([self.format, self.method, "+".join(self.groups), "w+a" if self.activations else "w"])


def round_e4m3(x):
    """Nearest FP8 E4M3 value, ties to even, saturating at ±448."""
    magnitude = x.abs().clamp(max=E4M3_MAX)
    exponent = torch.floor(torch.log2(magnitude.clamp(min=2.0**-9))).clamp(min=-6, max=8)
    step = torch.exp2(exponent - 3)
    return torch.copysign((torch.round(magnitude / step) * step).clamp(max=E4M3_MAX), x)


def round_e2m1(x):
    """Nearest FP4 E2M1 value, one of ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}."""
    magnitude = x.abs().clamp(max=E2M1_MAX)
    step = torch.where(magnitude < 2, 0.5, torch.where(magnitude < 4, 1.0, 2.0))
    return torch.copysign(torch.round(magnitude / step) * step, x)


def _positive(scale):
    # An all-zero channel or block still needs a usable scale.
    return torch.where(scale > 0, scale, torch.ones_like(scale))


def fp8(x):
    """E4M3 with one scale per row of the last dimension: an output channel or a token."""
    value = x.float()
    scale = _positive(value.abs().amax(dim=-1, keepdim=True) / E4M3_MAX)
    return (round_e4m3(value / scale) * scale).to(x.dtype)


def nvfp4(x, method="max", hessian=None):
    """E2M1 in blocks of 16 along the last dimension, with E4M3 block scales.

    `hessian` holds one 16×16 second moment of the layer input per block. With
    it, a block's scale minimises its own contribution to the output error;
    without it, the plain squared error.
    """
    value = x.float()
    if value.shape[-1] % BLOCK:
        raise ValueError("NVFP4 needs a last dimension divisible by 16")
    blocks = value.reshape(*value.shape[:-1], -1, BLOCK)
    tensor_scale = value.abs().max() / (E2M1_MAX * E4M3_MAX)
    if tensor_scale == 0:
        return x.clone()
    largest = blocks.abs().amax(dim=-1, keepdim=True)
    best, lowest = None, None
    for factor in FACTORS[method]:
        scale = _positive(round_e4m3(largest * factor / (E2M1_MAX * tensor_scale)) * tensor_scale)
        candidate = round_e2m1(blocks / scale) * scale
        if len(FACTORS[method]) == 1:
            best = candidate
            break
        error = blocks - candidate
        if hessian is None:
            cost = error.square().sum(dim=-1, keepdim=True)
        else:
            cost = torch.einsum("...bi,bij,...bj->...b", error, hessian, error).unsqueeze(-1)
        if best is None:
            best, lowest = candidate, cost
        else:
            better = cost < lowest
            best, lowest = torch.where(better, candidate, best), torch.where(better, cost, lowest)
    return best.reshape(value.shape).to(x.dtype)


def targets(model, groups):
    names = {name for group in groups for name in GROUPS[group]}
    return {
        path: module
        for path, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and path.rsplit(".", 1)[-1] in names
    }


@torch.no_grad()
def collect_hessians(model, modules, batches):
    """Per-block input second moments, taken from the unquantized model.

    Every layer therefore sees reference activations, never those of an
    already quantized layer.
    """
    sums, counts, hooks = {}, {}, []

    def observer(path):
        def hook(module, inputs):
            blocks = inputs[0].float().reshape(-1, inputs[0].shape[-1] // BLOCK, BLOCK)
            moment = torch.einsum("tbi,tbj->bij", blocks, blocks)
            sums[path] = sums.get(path, 0) + moment
            counts[path] = counts.get(path, 0) + blocks.shape[0]

        return hook

    for path, module in modules.items():
        hooks.append(module.register_forward_pre_hook(observer(path)))
    try:
        for batch in batches:
            model(**batch)
    finally:
        for hook in hooks:
            hook.remove()
    return {path: sums[path] / counts[path] for path in sums}


@torch.no_grad()
def apply(model, recipe, hessians=None):
    """Quantize the model in place and return each layer's relative weight error."""
    if recipe.method == "local_hessian" and hessians is None:
        raise ValueError("local_hessian needs calibration statistics")
    errors = {}
    for path, module in targets(model, recipe.groups).items():
        weight = module.weight.data
        if recipe.format == "fp8":
            quantized = fp8(weight)
        else:
            # Only the calibrated search may see activation statistics.
            quantized = nvfp4(weight, recipe.method, hessians[path] if recipe.method == "local_hessian" else None)
        errors[path] = ((quantized.float() - weight.float()).norm() / weight.float().norm().clamp(min=1e-12)).item()
        module.weight.data = quantized
        if recipe.activations:
            convert = fp8 if recipe.format == "fp8" else nvfp4
            module.register_forward_pre_hook(lambda module, inputs, convert=convert: (convert(inputs[0]),))
    return errors


@torch.no_grad()
def fidelity(reference, quantized, batches):
    """Teacher-forced KL(reference‖quantized) in nats per token and top-1 agreement."""
    divergence, agree, tokens = 0.0, 0, 0
    for batch in batches:
        reference_logits, quantized_logits = reference(**batch).logits, quantized(**batch).logits
        # One sequence at a time: FP32 log-probabilities over the vocabulary are large.
        for row, mask in enumerate(batch["attention_mask"].bool()):
            expected = torch.log_softmax(reference_logits[row][mask].float(), dim=-1)
            observed = torch.log_softmax(quantized_logits[row][mask].float(), dim=-1)
            divergence += (expected.exp() * (expected - observed)).sum().item()
            agree += (expected.argmax(dim=-1) == observed.argmax(dim=-1)).sum().item()
            tokens += int(mask.sum())
    return {"mean_kl": divergence / tokens, "top1_agreement": agree / tokens, "tokens": tokens}


def read_batches(tokenizer, path, device, max_tokens, batch_size):
    """Rows are {"text": ...} or {"prompt": [chat messages], "response": optional text}."""
    texts = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "text" in row:
            texts.append(row["text"])
        else:
            messages = list(row["prompt"])
            if row.get("response"):
                messages.append({"role": "assistant", "content": row["response"]})
            texts.append(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=not row.get("response"), enable_thinking=False
                )
            )
    if not texts:
        raise ValueError(f"{path} has no rows")
    tokenizer.padding_side = "right"
    batches = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=max_tokens
        )
        batches.append({key: encoded[key].to(device) for key in ("input_ids", "attention_mask")})
    return batches


def parse_recipe(text):
    """format[:method[:groups[:w]]], for example nvfp4:local_hessian:mlp or fp8:max:attention+mlp:w."""
    parts = text.split(":")
    return Recipe(
        parts[0],
        parts[1] if len(parts) > 1 else "max",
        tuple(parts[2].split("+")) if len(parts) > 2 else ("attention", "mlp"),
        not (len(parts) > 3 and parts[3] == "w"),
    )


def run(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    def load():
        return AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16).to(device).eval()

    reference = load()
    calibration = read_batches(tokenizer, args.calibration, device, args.max_tokens, args.batch_size)
    heldout = read_batches(tokenizer, args.heldout, device, args.max_tokens, args.batch_size)
    hessians = None
    results = []
    for recipe in [parse_recipe(text) for text in args.recipe]:
        if recipe.method == "local_hessian" and hessians is None:
            hessians = collect_hessians(reference, targets(reference, ("attention", "mlp")), calibration)
        model = load()
        errors = apply(model, recipe, hessians)
        measured = fidelity(reference, model, heldout)
        result = {
            "recipe": recipe.name,
            **asdict(recipe),
            **measured,
            "quantized_layers": len(errors),
            "mean_relative_weight_error": sum(errors.values()) / len(errors),
        }
        print(json.dumps(result), flush=True)
        results.append(result)
        if args.save and recipe.name == parse_recipe(args.save_recipe or args.recipe[0]).name:
            # Stored values are exactly representable in the target format but
            # kept as BF16 tensors, so any engine that loads the reference loads this.
            model.save_pretrained(args.save)
            tokenizer.save_pretrained(args.save)
            (Path(args.save) / "quantization.json").write_text(json.dumps(result, indent=2))
        del model
    report = {
        "model": str(args.model_dir),
        "simulation": "values are representable in the target format; arithmetic runs in floating point",
        "calibration_tokens": sum(int(b["attention_mask"].sum()) for b in calibration),
        "heldout_tokens": sum(int(b["attention_mask"].sum()) for b in heldout),
        "results": results,
    }
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--calibration", required=True, help="JSONL used only for local-Hessian statistics")
    parser.add_argument("--heldout", required=True, help="JSONL disjoint from calibration, used for fidelity")
    parser.add_argument("--recipe", action="append", help="format[:method[:groups[:w]]]; repeatable")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report")
    parser.add_argument("--save", help="directory for the weights of --save-recipe (default: the first recipe)")
    parser.add_argument("--save-recipe")
    args = parser.parse_args()
    args.recipe = args.recipe or ["fp8", "nvfp4:max:mlp", "nvfp4:four_over_six:mlp", "nvfp4:local_hessian:mlp"]
    run(args)


if __name__ == "__main__":
    main()
