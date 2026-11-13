"""Practice mock: build a random tiny Llama, its adapters and five merged checkpoints.

Nothing is downloaded. Every case is produced with the real PEFT calls
(``get_peft_model``, ``save_pretrained``, ``merge_and_unload``) and then checked
by the real ``check_merge`` path, so the lessons are real even though the
model is random and its numbers say nothing about real-model quality.
"""

from __future__ import annotations

import copy
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import save_file

from .check import check_merge
from .errors import InputError
from .outputs import quiet_transformers
from .verdict import Report

# Offline: the demo never talks to the Hub (set before transformers is imported).
os.environ.setdefault("HF_HUB_OFFLINE", "1")

#: random models have flat logits; a larger init makes the adapter effect measurable
INITIALIZER_RANGE = 0.3
TARGETS = ["q_proj", "v_proj", "o_proj", "gate_proj"]
_SMALL_UPDATE = 0.01  # B scale of the "small update" adapter: about half of it rounds away in bf16


@dataclass(frozen=True)
class CaseSpec:
    """One demo case: ``expected`` is a cause code, or None for a clean PASS."""

    key: str
    expected: str | None
    adapter: str
    merged: str


CASES = [
    CaseSpec("fp32_normal", None, "adapter", "merged_fp32_normal"),
    CaseSpec("dropped_o_proj", "update_not_applied", "adapter", "merged_dropped_o_proj"),
    CaseSpec("bf16_small_update", "low_precision_update_loss", "adapter_small", "merged_bf16_small_update"),
    CaseSpec("leftover_lora", "leftover_lora_keys", "adapter", "merged_leftover_lora"),
    CaseSpec("mts_not_merged", "modules_to_save_not_merged", "adapter_mts", "merged_mts_not_merged"),
]


@dataclass
class CaseResult:
    spec: CaseSpec
    report: Report
    ok: bool


def make_base(seed: int = 0):
    """Random 2-layer Llama (hidden 64, vocab 512) with untied lm_head."""
    from transformers import LlamaConfig, LlamaForCausalLM

    quiet_transformers()
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        tie_word_embeddings=False, initializer_range=INITIALIZER_RANGE,
    )
    return LlamaForCausalLM(cfg).eval()


def make_adapter(base, out: Path, *, b_scale: float = 1.0, modules_to_save: list[str] | None = None,
                 seed: int = 1) -> None:
    """Save a LoRA adapter with non-zero B (so the update is not a no-op)."""
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    cfg = LoraConfig(r=8, lora_alpha=16, target_modules=TARGETS, init_lora_weights=False,
                     modules_to_save=modules_to_save)
    peft_model = get_peft_model(copy.deepcopy(base), cfg)
    with torch.no_grad():
        for name, p in peft_model.named_parameters():
            if "lora_B" in name:
                p.mul_(b_scale)
            if modules_to_save and "modules_to_save" in name:
                p.add_(torch.randn_like(p) * 0.05)  # pretend lm_head was trained
    peft_model.save_pretrained(str(out))


def _load_peft(base_dir: Path, adapter_dir: Path, dtype: torch.dtype):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=dtype, local_files_only=True)
    return PeftModel.from_pretrained(base, str(adapter_dir), local_files_only=True).eval()


def _save_state(state: dict[str, torch.Tensor], out: Path, base_dir: Path) -> None:
    out.mkdir(parents=True)
    save_file({k: v.detach().contiguous() for k, v in state.items()}, str(out / "model.safetensors"),
              metadata={"format": "pt"})
    shutil.copy(base_dir / "config.json", out / "config.json")


def build_cases(out: Path, progress: Callable[[str], None] | None = None) -> Path:
    """Create ``base/``, three adapters and five merged folders under ``out``; returns ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    base = make_base()
    base_dir = out / "base"
    base.save_pretrained(str(base_dir), safe_serialization=True)
    make_adapter(base, out / "adapter", b_scale=1.0)
    make_adapter(base, out / "adapter_small", b_scale=_SMALL_UPDATE)
    make_adapter(base, out / "adapter_mts", b_scale=1.0, modules_to_save=["lm_head"])

    # 1. correct fp32 merge
    merged = _load_peft(base_dir, out / "adapter", torch.float32).merge_and_unload()
    good_state = {k: v.clone() for k, v in merged.state_dict().items()}
    _save_state(good_state, out / "merged_fp32_normal", base_dir)

    # 2. the o_proj update was never added (partial merge / wrong target modules)
    base_state = base.state_dict()
    dropped = {k: (base_state[k] if "o_proj" in k else v) for k, v in good_state.items()}
    _save_state(dropped, out / "merged_dropped_o_proj", base_dir)

    # 3. merged in bf16 with a small update: the update mostly rounds away
    small = _load_peft(base_dir, out / "adapter_small", torch.bfloat16).merge_and_unload()
    _save_state(small.state_dict(), out / "merged_bf16_small_update", base_dir)

    # 4. merge_and_unload() result not used: the wrapper still holds lora_ and base_layer keys
    wrapper = _load_peft(base_dir, out / "adapter", torch.float32)
    wrapper.merge_adapter()
    _save_state(wrapper.base_model.model.state_dict(), out / "merged_leftover_lora", base_dir)

    # 5. lm_head was trained (modules_to_save) but the merged file keeps the base lm_head
    with_mts = _load_peft(base_dir, out / "adapter_mts", torch.float32).merge_and_unload()
    mts_state = {k: (base_state[k] if k == "lm_head.weight" else v) for k, v in with_mts.state_dict().items()}
    _save_state(mts_state, out / "merged_mts_not_merged", base_dir)
    return out


def _matches(spec: CaseSpec, report: Report) -> bool:
    codes = report.cause_codes()
    if spec.expected is None:
        return report.verdict == "PASS" and not codes
    if spec.expected == "low_precision_update_loss":
        return spec.expected in codes and report.verdict in ("WARN", "FAIL")
    return spec.expected in codes and report.verdict == "FAIL"


def run_demo(out: Path | None = None, progress: Callable[[str], None] | None = None,
             on_case: Callable[[int, int, str], None] | None = None) -> tuple[list[CaseResult], Path]:
    """Build the five cases (in ``out`` or a temp folder), check them, return results.

    Raises ``InputError`` when ``out`` exists and is not empty. When ``out`` is None
    the temporary folder is removed before returning (the returned path then no longer exists).
    """
    import tempfile

    if out is not None and out.exists() and any(out.iterdir()):
        raise InputError("demo_out_not_empty", path=str(out))
    tmp = None
    if out is None:
        tmp = tempfile.TemporaryDirectory(prefix="lora-mergecheck-demo-")
        root = Path(tmp.name)
    else:
        root = out
    try:
        build_cases(root, progress)
        results: list[CaseResult] = []
        for i, spec in enumerate(CASES, start=1):
            if on_case:
                on_case(i, len(CASES), spec.key)
            report = check_merge(root / "base", root / spec.adapter, root / spec.merged)
            results.append(CaseResult(spec, report, _matches(spec, report)))
        return results, root
    finally:
        if tmp is not None:
            tmp.cleanup()
