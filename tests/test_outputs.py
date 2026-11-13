"""Output layer: the verdict is relative to the adapter effect, not a fixed threshold."""

from __future__ import annotations

import pytest
import torch
from conftest import write_state

from lora_mergecheck import InputError, check_merge, demo
from lora_mergecheck.outputs import compare_logits, judge


def outputs_only(base, adapter, merged, **kw):
    return check_merge(base, adapter, merged, layers="outputs", **kw)


def test_correct_merge_passes(cases):
    report = outputs_only(cases / "base", cases / "adapter", cases / "merged_fp32_normal")
    out = report.outputs
    assert report.verdict == "PASS"
    assert out["relative_drift"] <= 0.10
    assert out["adapter_effect_kl"] > 1000 * out["numeric_floor_kl"]
    assert out["top1_agree"] == 1.0 and out["greedy_agree"] == 1.0
    assert out["n_prompts"] == 8


def _merged_with_noise(root, name, b_scale, noise_std):
    """Correct fp32 merge of an adapter of strength ``b_scale`` plus fixed lm_head noise."""
    base = demo.make_base()
    base_dir = root / "base"
    if not base_dir.exists():
        base.save_pretrained(str(base_dir))
    adapter_dir = root / f"adapter_{name}"
    demo.make_adapter(base, adapter_dir, b_scale=b_scale)
    merged = demo._load_peft(base_dir, adapter_dir, torch.float32).merge_and_unload()
    state = {k: v.clone() for k, v in merged.state_dict().items()}
    gen = torch.Generator().manual_seed(5)
    head = state["lm_head.weight"]
    state["lm_head.weight"] = head + torch.randn(head.shape, generator=gen) * noise_std
    return base_dir, adapter_dir, write_state(state, root / f"merged_{name}", base_dir)


def test_threshold_scales_with_adapter_effect(tmp_path):
    """The same absolute perturbation passes for a strong adapter and fails for a weak one."""
    strong = outputs_only(*_merged_with_noise(tmp_path, "strong", 1.0, 0.003))
    weak = outputs_only(*_merged_with_noise(tmp_path, "weak", 0.003, 0.003))
    s, w = strong.outputs, weak.outputs
    assert 0.3 < s["drift_kl_mean"] / w["drift_kl_mean"] < 3.0  # same absolute drift
    assert strong.verdict == "PASS"
    assert weak.verdict == "FAIL" and weak.cause_codes() == ["output_drift"]
    assert s["adapter_effect_kl"] > 1000 * w["adapter_effect_kl"]


def test_tiny_effect_is_inconclusive(tmp_path):
    report = outputs_only(*_merged_with_noise(tmp_path, "tiny", 1e-6, 0.0))
    assert report.verdict == "INCONCLUSIVE"
    assert report.cause_codes() == ["adapter_effect_too_small"]


def test_drift_is_reported_for_a_broken_merge(cases):
    report = outputs_only(cases / "base", cases / "adapter", cases / "merged_dropped_o_proj")
    assert report.verdict == "FAIL" and report.cause_codes() == ["output_drift"]
    assert report.outputs["greedy_agree"] < 1.0


def test_weights_failure_with_structure_problem_skips_outputs(cases):
    report = check_merge(cases / "base", cases / "adapter", cases / "merged_leftover_lora")
    assert report.outputs == {"status": "skipped", "skipped_reason": "structure"}


def test_prompts_file_needs_a_tokenizer(cases, work):
    prompts = work / "prompts.txt"
    prompts.write_text("hello world\n")
    with pytest.raises(InputError) as info:
        outputs_only(cases / "base", cases / "adapter", cases / "merged_fp32_normal", prompts=prompts)
    assert info.value.code == "bad_prompts"


def test_gen_tokens_zero_skips_generation(cases):
    report = outputs_only(cases / "base", cases / "adapter", cases / "merged_fp32_normal", gen_tokens=0)
    assert report.outputs["greedy_agree"] is None


def test_judge_unit_thresholds():
    def stats(kl, top1=1.0):
        return {"kl_mean": kl, "top1_agree": top1, "n_tokens": 100}

    ok, nums = judge(stats(0.01), stats(1.0), stats(1e-9), max_relative_drift=0.1)
    assert ok == [] and nums["relative_drift"] == pytest.approx(0.01)
    bad, _ = judge(stats(0.01), stats(0.01), stats(1e-9), max_relative_drift=0.1)
    assert [c.code for c in bad] == ["output_drift"]
    flat, _ = judge(stats(0.01), stats(1e-9), stats(1e-9), max_relative_drift=0.1)
    assert [c.code for c in flat] == ["adapter_effect_too_small"]
    flips, _ = judge(stats(1e-6, top1=0.5), stats(1.0, top1=0.9), stats(1e-9), max_relative_drift=0.1)
    assert [c.code for c in flips] == ["output_drift"]  # top-1 flips alone can fail the check


def test_compare_logits_identical_inputs():
    logits = [torch.randn(5, 11), torch.randn(3, 11)]
    result = compare_logits(logits, logits)
    assert result["kl_mean"] == 0.0 and result["top1_agree"] == 1.0 and result["n_tokens"] == 8
    assert result["max_logit_diff"] == 0.0



def test_reference_model_is_freed_before_the_merged_model_loads(cases):
    """Peak memory is one fp32 model at a time: the reference must be gone when the merged load starts."""
    import gc
    import weakref

    from peft import PeftModel

    refs: list[weakref.ref] = []
    real = PeftModel.from_pretrained.__func__

    def spy(cls, *args, **kwargs):
        model = real(cls, *args, **kwargs)
        refs.extend([weakref.ref(model), weakref.ref(model.get_base_model())])
        return model

    alive_at_merged_load: list[bool] = []

    def progress(message: str) -> None:
        if message.startswith("loading merged"):
            gc.collect()
            alive_at_merged_load.extend(r() is not None for r in refs)

    PeftModel.from_pretrained = classmethod(spy)
    try:
        outputs_only(cases / "base", cases / "adapter", cases / "merged_fp32_normal", progress=progress)
    finally:
        PeftModel.from_pretrained = classmethod(real)
    assert len(alive_at_merged_load) == 2 and not any(alive_at_merged_load)
