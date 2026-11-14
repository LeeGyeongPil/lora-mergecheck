"""Output layer: do the merged weights answer like the adapter model?

Three reference points are computed (models are loaded one after another and
only the logits are kept):

* ``ref``    base + adapter, unmerged, fp32 (the behaviour we want to preserve)
* ``base``   the same model with the adapter switched off (how far the adapter moves the model)
* ``floor``  an in-memory fp32 ``merge_and_unload()`` (unavoidable numeric noise)

Then the merged checkpoint is loaded (upcast to fp32 so only the stored values matter).
The verdict compares the drift ``KL(ref || merged)`` with the *adapter effect*
``KL(ref || base)``: the same absolute drift can be harmless for a strong
adapter and fatal for a weak one, so there is no fixed absolute threshold.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F

from .errors import InputError
from .verdict import Cause

#: the adapter effect must exceed the numeric floor by this factor to be judged at all
EFFECT_OVER_FLOOR = 10.0
_TINY = 1e-12
_PREFIX_TOKENS = 8
_SEED = 0
#: any one of these files in the base folder means a tokenizer can be loaded
TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
    "vocab.json", "merges.txt", "spiece.model",
)

Progress = Callable[[str], None]


def quiet_transformers() -> None:
    """Silence weight-loading progress bars and deprecation chatter on stderr."""
    import warnings

    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    warnings.filterwarnings("ignore", category=FutureWarning)


def compare_logits(ref: list[torch.Tensor], other: list[torch.Tensor]) -> dict:
    """Token-level comparison of two lists of ``(tokens, vocab)`` logits."""
    kls: list[torch.Tensor] = []
    agree = total = 0
    max_diff = 0.0
    for a, b in zip(ref, other, strict=True):
        lp_a = F.log_softmax(a.float(), dim=-1)
        lp_b = F.log_softmax(b.float(), dim=-1)
        kls.append((lp_a.exp() * (lp_a - lp_b)).sum(-1).clamp(min=0))
        agree += int((a.argmax(-1) == b.argmax(-1)).sum())
        total += a.shape[0]
        max_diff = max(max_diff, float((a.float() - b.float()).abs().max()))
    kl = torch.cat(kls)
    return {"kl_mean": float(kl.mean()), "kl_max": float(kl.max()), "max_logit_diff": max_diff,
            "top1_agree": agree / total, "n_tokens": total}


def judge(
    drift: dict, effect: dict, floor: dict, *, max_relative_drift: float
) -> tuple[list[Cause], dict]:
    """Turn the three comparisons into causes and the relative numbers.

    ``drift`` = ref vs merged, ``effect`` = ref vs base, ``floor`` = ref vs fp32 in-memory merge.
    """
    effect_kl = effect["kl_mean"]
    floor_kl = floor["kl_mean"]
    relative = drift["kl_mean"] / max(effect_kl, _TINY)
    n = drift["n_tokens"]
    flips = (1.0 - drift["top1_agree"]) / max(1.0 - effect["top1_agree"], 1.0 / n)
    numbers = {"relative_drift": relative, "top1_flip_ratio": flips,
               "max_relative_drift": max_relative_drift}
    details = {"adapter_effect_kl": effect_kl, "numeric_floor_kl": floor_kl,
               "drift_kl_mean": drift["kl_mean"], **numbers}
    if effect_kl <= EFFECT_OVER_FLOOR * max(floor_kl, _TINY):
        return [Cause("adapter_effect_too_small", [], details)], numbers
    if relative > max_relative_drift or flips > max_relative_drift:
        return [Cause("output_drift", [], details)], numbers
    return [], numbers


def _load_prompts(
    base: Path, prompts_file: Path | None, num_prompts: int, prompt_len: int, vocab: int
) -> list[torch.Tensor]:
    if prompts_file is None:
        gen = torch.Generator().manual_seed(_SEED)
        ids = torch.randint(0, vocab, (num_prompts, prompt_len), generator=gen)
        return [row for row in ids]
    # Check for tokenizer files first, so a missing tokenizer is reported as such
    # instead of as a misleading "install sentencepiece" error from transformers.
    if not any((base / name).is_file() for name in TOKENIZER_FILES):
        raise InputError("no_tokenizer", path=str(base))
    try:
        from transformers import AutoTokenizer

        lines = [x for x in prompts_file.read_text(encoding="utf-8").splitlines() if x.strip()]
        tok = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
        out = [torch.tensor(tok(x)["input_ids"][:256]) for x in lines]
    except Exception as exc:  # unreadable file, missing tokenizer, ...
        raise InputError("bad_prompts", path=str(prompts_file), reason=str(exc)) from exc
    if not out:
        raise InputError("bad_prompts", path=str(prompts_file), reason="no non-empty lines")
    return out


@torch.no_grad()
def _logits(model, prompts: list[torch.Tensor], device: str) -> list[torch.Tensor]:
    return [model(input_ids=p[None].to(device)).logits[0].float().cpu() for p in prompts]


@torch.no_grad()
def _generate(model, prompts: list[torch.Tensor], n_new: int, device: str) -> list[list[int]]:
    outs = []
    for p in prompts:
        prefix = p[: min(_PREFIX_TOKENS, len(p))][None].to(device)
        gen = model.generate(input_ids=prefix, attention_mask=torch.ones_like(prefix),
                             max_new_tokens=n_new, do_sample=False, pad_token_id=0)
        outs.append(gen[0].tolist())
    return outs


def _free_memory(device: str) -> None:
    """Collect garbage after the caller dropped its model references (``del``), then trim device caches."""
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.startswith("mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def check_outputs(
    base: Path,
    adapter: Path,
    merged: Path,
    *,
    prompts_file: Path | None = None,
    num_prompts: int = 8,
    prompt_len: int = 32,
    gen_tokens: int = 16,
    max_relative_drift: float = 0.10,
    device: str = "cpu",
    progress: Progress | None = None,
) -> tuple[dict, list[Cause]]:
    """Run the output layer. Raises ``InputError`` for unusable prompt files."""
    say = progress or (lambda _msg: None)
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM

    quiet_transformers()
    vocab = int(AutoConfig.from_pretrained(str(base), local_files_only=True).vocab_size)
    prompt_ids = _load_prompts(base, prompts_file, num_prompts, prompt_len, vocab)

    def load(path: Path):
        model = AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.float32, local_files_only=True)
        return model.eval().to(device)

    say("loading base + adapter ...")
    peft_model = PeftModel.from_pretrained(load(base), str(adapter), local_files_only=True).eval()
    ref_logits = _logits(peft_model, prompt_ids, device)
    with peft_model.disable_adapter():
        base_logits = _logits(peft_model, prompt_ids, device)
    ref_gen = _generate(peft_model, prompt_ids, gen_tokens, device) if gen_tokens else None
    say("merging in memory (fp32 floor) ...")
    in_memory = peft_model.merge_and_unload()
    floor_logits = _logits(in_memory, prompt_ids, device)
    del in_memory, peft_model  # drop the last references so only one fp32 model is alive at a time
    _free_memory(device)

    say("loading merged checkpoint ...")
    merged_model = load(merged)
    merged_logits = _logits(merged_model, prompt_ids, device)
    merged_gen = _generate(merged_model, prompt_ids, gen_tokens, device) if gen_tokens else None
    del merged_model
    _free_memory(device)

    drift = compare_logits(ref_logits, merged_logits)
    effect = compare_logits(ref_logits, base_logits)
    floor = compare_logits(ref_logits, floor_logits)
    causes, numbers = judge(drift, effect, floor, max_relative_drift=max_relative_drift)
    greedy = None
    if ref_gen is not None and merged_gen is not None:
        greedy = sum(a == b for a, b in zip(ref_gen, merged_gen, strict=True)) / len(ref_gen)
    result = {
        "status": "checked",
        "n_prompts": len(prompt_ids),
        "n_tokens": drift["n_tokens"],
        "device": device,
        "adapter_effect_kl": effect["kl_mean"],
        "numeric_floor_kl": floor["kl_mean"],
        "drift_kl_mean": drift["kl_mean"],
        "drift_kl_max": drift["kl_max"],
        "relative_drift": numbers["relative_drift"],
        "top1_agree": drift["top1_agree"],
        "top1_flip_ratio": numbers["top1_flip_ratio"],
        "greedy_agree": greedy,
        "max_logit_diff": drift["max_logit_diff"],
        "max_relative_drift": max_relative_drift,
    }
    return result, causes
