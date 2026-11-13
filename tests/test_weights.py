"""Weight layer: classification, rounding limits, lost-update ratio."""

from __future__ import annotations

import torch
from conftest import copy_dir, edit_json

from lora_mergecheck import check_merge
from lora_mergecheck.weights import classify_module, ulp

O_PROJ = {"model.layers.0.self_attn.o_proj", "model.layers.1.self_attn.o_proj"}


def weights_only(cases, adapter, merged, **kw):
    return check_merge(cases / "base", cases / adapter, cases / merged, layers="weights", **kw)


def test_fp32_merge_is_exact(cases):
    report = weights_only(cases, "adapter", "merged_fp32_normal")
    counts = report.weights["counts"]
    assert counts == {"exact": 8, "rounding": 0, "mismatch": 0, "not_applied": 0, "missing": 0}
    assert report.weights["max_residual"] <= 1e-6
    assert report.verdict == "PASS"
    assert report.causes == []


def test_dropped_module_is_named(cases):
    report = weights_only(cases, "adapter", "merged_dropped_o_proj")
    cause = report.cause("update_not_applied")
    assert report.verdict == "FAIL"
    assert cause is not None
    assert {m.removesuffix(".weight") for m in cause.modules} == O_PROJ
    assert report.weights["counts"]["exact"] == 6


def test_bf16_rounding_is_not_mismatch(cases):
    report = weights_only(cases, "adapter", "merged_bf16_normal")
    assert report.weights["merged_dtype"] == "bfloat16"
    assert report.weights["counts"]["mismatch"] == 0
    assert report.weights["counts"]["rounding"] == 8
    assert report.cause("non_lora_changed") is None  # base is cast to bf16 before comparing
    assert report.verdict == "PASS"


def test_bf16_small_update_warns(cases):
    report = weights_only(cases, "adapter_small", "merged_bf16_small_update")
    cause = report.cause("low_precision_update_loss")
    assert cause is not None
    assert report.verdict == "WARN"
    assert report.weights["lost_update_ratio"] >= 0.25
    assert report.weights["counts"]["mismatch"] == 0  # the merge itself followed bf16 rules


def test_warn_threshold_is_configurable(cases):
    report = weights_only(cases, "adapter_small", "merged_bf16_small_update",
                          warn_lost_ratio=1.0)
    # the 0.9 retention rule still fires, so the warning does not vanish silently
    assert report.cause("low_precision_update_loss") is not None


def test_rslora_scale_is_honoured(cases, work):
    adapter = copy_dir(cases / "adapter", work / "adapter")
    edit_json(adapter / "adapter_config.json", use_rslora=True)
    report = check_merge(cases / "base", adapter, cases / "merged_fp32_normal", layers="weights")
    assert report.weights["counts"]["mismatch"] == 8  # s = alpha/sqrt(r) differs from alpha/r
    assert report.cause("update_mismatch") is not None


def test_alpha_pattern_targets_matching_modules(cases, work):
    adapter = copy_dir(cases / "adapter", work / "adapter")
    edit_json(adapter / "adapter_config.json", alpha_pattern={"q_proj": 32})
    report = check_merge(cases / "base", adapter, cases / "merged_fp32_normal", layers="weights")
    bad = {m.removesuffix(".weight") for m in report.cause("update_mismatch").modules}
    assert bad == {"model.layers.0.self_attn.q_proj", "model.layers.1.self_attn.q_proj"}


def test_fan_in_fan_out_transposes_the_update(cases, work):
    adapter = copy_dir(cases / "adapter", work / "adapter")
    edit_json(adapter / "adapter_config.json", fan_in_fan_out=True)
    report = check_merge(cases / "base", adapter, cases / "merged_fp32_normal", layers="weights")
    assert report.verdict == "FAIL"
    assert report.cause("shape_mismatch") is not None  # non-square layers cannot match when transposed


def test_ulp_matches_dtype_spacing():
    one = torch.tensor([1.0, 0.0])
    assert ulp(one, torch.float32)[0].item() == 2.0**-23
    assert ulp(one, torch.bfloat16)[0].item() == 2.0**-7
    assert ulp(one, torch.float16)[0].item() == 2.0**-10
    assert ulp(torch.tensor([3.0]), torch.bfloat16)[0].item() == 2.0**-6
    assert ulp(one, torch.float16)[1].item() == 2.0**-24  # zero uses the smallest spacing


def test_classify_module_unit_cases():
    torch.manual_seed(0)
    w = torch.randn(16, 16)
    delta = torch.randn(16, 16) * 0.1
    assert classify_module(w, w + delta, delta)["status"] == "exact"
    assert classify_module(w, w.clone(), delta)["status"] == "not_applied"
    assert classify_module(w, w + 2 * delta, delta)["status"] == "mismatch"
    bf16 = (w.bfloat16().float() + delta).bfloat16()
    assert classify_module(w, bf16, delta)["status"] == "rounding"
    zero = classify_module(w, w.clone(), torch.zeros_like(w))
    assert zero["status"] == "exact" and zero["update_retention"] == 1.0
