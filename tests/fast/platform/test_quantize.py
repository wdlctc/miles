"""Number formats, scale searches and fidelity accounting on CPU; no model download."""

import copy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from miles_platform.quantize import (  # noqa: E402
    BLOCK,
    Recipe,
    apply,
    collect_hessians,
    fidelity,
    fp8,
    nvfp4,
    parse_recipe,
    round_e2m1,
    round_e4m3,
    targets,
)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(50, 32)
        self.q_proj = torch.nn.Linear(32, 32, bias=False)
        self.up_proj = torch.nn.Linear(32, 64, bias=False)
        self.down_proj = torch.nn.Linear(64, 32, bias=False)
        self.lm_head = torch.nn.Linear(32, 50, bias=False)

    def forward(self, input_ids, attention_mask):
        hidden = self.embed(input_ids)
        hidden = hidden + self.q_proj(hidden)
        hidden = hidden + self.down_proj(torch.nn.functional.silu(self.up_proj(hidden)))
        return SimpleNamespace(logits=self.lm_head(hidden))


def batches(count=3):
    generator = torch.Generator().manual_seed(1)
    ids = [torch.randint(0, 50, (4, 12), generator=generator) for _ in range(count)]
    return [{"input_ids": item, "attention_mask": torch.ones_like(item)} for item in ids]


def test_e4m3_rounds_to_representable_values_and_saturates():
    values = torch.tensor([448.0, 1000.0, -1000.0, 0.3, 1.0625, 2.0**-9, 2.0**-11, 0.0])
    assert round_e4m3(values).tolist() == [448.0, 448.0, -448.0, 0.3125, 1.0, 2.0**-9, 0.0, 0.0]
    sample = torch.randn(4096, generator=torch.Generator().manual_seed(0)) * 50
    assert torch.equal(round_e4m3(round_e4m3(sample)), round_e4m3(sample))


def test_e2m1_uses_only_the_eight_fp4_magnitudes():
    values = torch.tensor([7.0, 5.0, 2.6, 2.4, -0.74, 0.2, 0.0])
    assert round_e2m1(values).tolist() == [6.0, 4.0, 3.0, 2.0, -0.5, 0.0, 0.0]
    sample = torch.randn(4096, generator=torch.Generator().manual_seed(0)) * 3
    assert set(round_e2m1(sample).abs().unique().tolist()) <= {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}


def test_fp8_keeps_each_channel_maximum_and_handles_zero_rows():
    weight = torch.randn(8, 64, generator=torch.Generator().manual_seed(0))
    weight[3] = 0
    quantized = fp8(weight)
    assert torch.allclose(quantized.abs().amax(dim=1), weight.abs().amax(dim=1), rtol=1e-6)
    assert torch.equal(quantized[3], torch.zeros(64))
    assert (quantized - weight).norm() / weight.norm() < 0.03


def test_nvfp4_scale_searches_never_increase_the_error_they_minimise():
    generator = torch.Generator().manual_seed(0)
    weight = torch.randn(24, 64, generator=generator) * torch.rand(24, 1, generator=generator)
    inputs = torch.randn(500, 64 // BLOCK, BLOCK, generator=generator)
    hessian = torch.einsum("tbi,tbj->bij", inputs, inputs) / 500

    def squared(quantized):
        return (quantized - weight).square().sum().item()

    def output(quantized):
        error = (quantized - weight).reshape(24, -1, BLOCK)
        return torch.einsum("obi,bij,obj->", error, hessian, error).item()

    maximum, four_six, local = (nvfp4(weight, m) for m in ("max", "four_over_six", "local_hessian"))
    assert squared(four_six) <= squared(maximum) and squared(local) <= squared(four_six)
    calibrated = nvfp4(weight, "local_hessian", hessian)
    assert output(calibrated) <= output(four_six) and output(calibrated) <= output(local)
    assert torch.equal(nvfp4(torch.zeros(2, 32)), torch.zeros(2, 32))
    with pytest.raises(ValueError, match="divisible by 16"):
        nvfp4(torch.zeros(2, 20))


def test_fidelity_is_exact_for_the_reference_and_orders_formats():
    torch.manual_seed(0)
    reference = Tiny().eval()
    data = batches()
    same = fidelity(reference, reference, data)
    assert same["mean_kl"] == pytest.approx(0, abs=1e-7) and same["top1_agreement"] == 1 and same["tokens"] == 144
    measured = {}
    for recipe in (Recipe("fp8"), Recipe("nvfp4")):
        model = copy.deepcopy(reference)
        errors = apply(model, recipe)
        assert set(errors) == {"q_proj", "up_proj", "down_proj"}
        measured[recipe.format] = fidelity(reference, model, data)["mean_kl"]
    assert 0 < measured["fp8"] < measured["nvfp4"]


def test_local_hessian_calibrates_on_reference_activations_only():
    torch.manual_seed(0)
    reference = Tiny().eval()
    before = copy.deepcopy(reference.state_dict())
    selected = targets(reference, ("mlp",))
    assert set(selected) == {"up_proj", "down_proj"}
    hessians = collect_hessians(reference, selected, batches())
    assert hessians["up_proj"].shape == (2, BLOCK, BLOCK) and hessians["down_proj"].shape == (4, BLOCK, BLOCK)
    assert all(torch.equal(before[key], value) for key, value in reference.state_dict().items())
    assert not reference.up_proj._forward_pre_hooks
    model = copy.deepcopy(reference)
    with pytest.raises(ValueError, match="calibration"):
        apply(model, Recipe("nvfp4", "local_hessian", ("mlp",)))
    apply(model, Recipe("nvfp4", "local_hessian", ("mlp",), activations=False), hessians)
    uncalibrated = [copy.deepcopy(reference) for _ in range(2)]
    apply(uncalibrated[0], Recipe("nvfp4", "four_over_six", ("mlp",)))
    apply(uncalibrated[1], Recipe("nvfp4", "four_over_six", ("mlp",)), hessians)
    assert torch.equal(uncalibrated[0].up_proj.weight, uncalibrated[1].up_proj.weight)
    assert torch.equal(model.q_proj.weight, reference.q_proj.weight)
    assert not torch.equal(model.up_proj.weight, reference.up_proj.weight)


def test_recipe_parsing_and_validation():
    assert parse_recipe("fp8") == Recipe("fp8")
    recipe = parse_recipe("nvfp4:local_hessian:mlp:w")
    assert recipe == Recipe("nvfp4", "local_hessian", ("mlp",), False) and recipe.name == "nvfp4-local_hessian-mlp-w"
    for bad in ("int8", "fp8:local_hessian", "nvfp4:max:embeddings"):
        with pytest.raises(ValueError):
            parse_recipe(bad)
