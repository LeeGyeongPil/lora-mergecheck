"""Weight layer: compare the merged checkpoint with ``base + s*B*A`` tensor by tensor.

No model is built and ``transformers`` is never imported here: only safetensors
headers and one tensor at a time are read.
"""

from __future__ import annotations

import math

import torch

from .adapter import Adapter, LoraModule
from .checkpoint import Checkpoint
from .verdict import Cause

#: mantissa bits per floating dtype (spacing between neighbours = 2**(exponent - bits))
_MANTISSA = {torch.float32: 23, torch.bfloat16: 7, torch.float16: 10}
_EXACT_TOLERANCE = 1e-6
_NOT_APPLIED_SHARE = 0.99
_MIN_RETENTION = 0.9
_VOCAB_LAYERS = ("embed_tokens", "lm_head", "wte", "word_embeddings")
_LISTED_MAX = 20  # modules named per cause in the report


def ulp(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Spacing between ``|x|`` and the next larger number representable in ``dtype``."""
    bits = _MANTISSA.get(dtype, 23)
    emin = int(round(math.log2(torch.finfo(dtype).tiny))) if dtype in _MANTISSA else -126
    _, exp = torch.frexp(x.abs().float())
    exp = torch.where(x == 0, torch.full_like(exp, emin), exp - 1)  # frexp mantissa is in [0.5, 1)
    exp = torch.clamp(exp, min=emin) - bits
    return torch.ldexp(torch.ones_like(x, dtype=torch.float32), exp)


def _norm(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x.double()))


def classify_module(
    base_w: torch.Tensor, merged_w: torch.Tensor, delta: torch.Tensor
) -> dict:
    """Compare one LoRA layer and return its status and statistics.

    ``base_w`` is cast to the merged dtype first (what a merge in that dtype
    starts from), then ``delta`` (fp32) is added in fp32 to form the expectation.
    """
    dtype = merged_w.dtype
    w_c = base_w.to(dtype).float()
    merged = merged_w.float()
    expected = w_c + delta
    resid = merged - expected
    abs_r = resid.abs()
    max_resid = float(abs_r.max()) if abs_r.numel() else 0.0
    ref = max(1.0, float(expected.abs().max()))

    nonzero = delta != 0
    n_nonzero = int(nonzero.sum())
    lost = 0.0
    if n_nonzero:
        lost = float(((expected.to(dtype).float() == w_c) & nonzero).sum()) / n_nonzero
    d_norm = _norm(delta)
    retention = 1.0 if d_norm == 0 else min(1.0, 1.0 - _norm(resid) / d_norm)
    w_norm = _norm(w_c)

    required = delta.abs() >= ulp(w_c, dtype)
    n_required = int(required.sum())
    unchanged = float((merged == w_c)[required].float().mean()) if n_required else 0.0

    if max_resid <= _EXACT_TOLERANCE * ref:
        status = "exact"
    elif n_required and unchanged >= _NOT_APPLIED_SHARE:
        status = "not_applied"
    elif bool((abs_r <= 2 * ulp(torch.maximum(expected.abs(), w_c.abs()), dtype)).all()):
        status = "rounding"
    else:
        status = "mismatch"
    return {
        "status": status,
        "max_resid": max_resid,
        "lost_update_ratio": lost,
        "update_retention": retention,
        "rel_update_norm": d_norm / w_norm if w_norm else 0.0,
    }


def _dtype_name(t: torch.Tensor) -> str:
    return str(t.dtype).replace("torch.", "")


def _unsupported_weight_keys(adapter: Adapter) -> set[str]:
    """Base ``.weight`` keys of layers whose adapter type is not supported."""
    skip: set[str] = set()
    for key in adapter.unsupported:
        name = key[len("base_model.model."):] if key.startswith("base_model.model.") else key
        for marker in (".lora_", ".trainable_tokens", ".magnitude"):
            if marker in name:
                skip.add(name.split(marker)[0] + ".weight")
                break
    return skip


def _cap(names: list[str]) -> list[str]:
    return sorted(names)[:_LISTED_MAX]


def check_weights(
    base: Checkpoint,
    adapter: Adapter,
    merged: Checkpoint,
    *,
    warn_lost_ratio: float = 0.25,
) -> tuple[dict, list[Cause]]:
    """Run the weight layer. Returns ``(weights_dict, causes)``."""
    causes: list[Cause] = []
    result: dict = {"status": "checked", "skipped_reason": None, "merged_dtype": None, "counts": {},
                    "lost_update_ratio": None, "update_retention": None, "max_residual": None,
                    "modules": [], "saved_layers": [], "non_lora": {"checked": 0, "changed": []},
                    "quantization": []}

    leftover = [k for k in merged.keys() if ".lora_" in k or ".base_layer." in k or ".modules_to_save." in k]
    if leftover:
        causes.append(Cause("leftover_lora_keys", _cap(leftover), {"count": len(leftover)}))
        result.update(status="skipped", skipped_reason="leftover")
        return result, causes

    quant = base.quantization_evidence() + merged.quantization_evidence()
    if quant:
        causes.append(Cause("quantized_checkpoint", [], {"evidence": quant}))
        result.update(status="skipped", skipped_reason="quantized", quantization=quant)
        return result, causes

    if adapter.unsupported:
        causes.append(Cause("unsupported_layer", _cap(adapter.unsupported), {"count": len(adapter.unsupported)}))
    if not adapter.modules and not adapter.saved_layers:
        result.update(status="skipped", skipped_reason="no_lora")
        return result, causes

    rows: list[dict] = []
    missing: list[str] = []
    shape_bad: list[str] = []
    vocab_bad: list[str] = []
    lora_keys: set[str] = set()
    dtypes: list[str] = []

    for module in adapter.modules.values():
        key = module.weight_key
        lora_keys.add(key)
        if key not in base or key not in merged:
            missing.append(key)
            rows.append({"module": module.name, "status": "missing"})
            continue
        if base.shape(key) != merged.shape(key):
            shape_bad.append(key)
            continue
        merged_w = merged.get(key)
        delta = adapter.delta_weight(module)
        if tuple(delta.shape) != tuple(merged_w.shape):
            shape_bad.append(key)
            continue
        stats = classify_module(base.get(key), merged_w, delta)
        dtypes.append(_dtype_name(merged_w))
        rows.append({"module": module.name, "rank": module.rank, "scale": module.scale, **stats})

    counts = {s: 0 for s in ("exact", "rounding", "mismatch", "not_applied", "missing")}
    for row in rows:
        counts[row["status"]] += 1
    result["counts"] = counts
    result["modules"] = rows
    result["merged_dtype"] = max(set(dtypes), key=dtypes.count) if dtypes else None

    def modules_with(status: str) -> list[str]:
        return [r["module"] + ".weight" for r in rows if r["status"] == status]

    if counts["not_applied"]:
        causes.append(Cause("update_not_applied", _cap(modules_with("not_applied"))))
    if counts["mismatch"]:
        causes.append(Cause("update_mismatch", _cap(modules_with("mismatch"))))

    ok_rows = [r for r in rows if r["status"] in ("exact", "rounding")]
    if ok_rows:
        lost = sum(r["lost_update_ratio"] for r in ok_rows) / len(ok_rows)
        retention = sum(r["update_retention"] for r in ok_rows) / len(ok_rows)
        result["lost_update_ratio"] = lost
        result["update_retention"] = retention
        result["max_residual"] = max(r["max_resid"] for r in ok_rows)
        if lost >= warn_lost_ratio or retention < _MIN_RETENTION:
            causes.append(Cause("low_precision_update_loss", [],
                                {"lost_update_ratio": lost, "update_retention": retention,
                                 "merged_dtype": result["merged_dtype"], "threshold": warn_lost_ratio}))

    # modules_to_save: the merged layer must hold the adapter's saved values.
    saved_bad: list[str] = []
    skip_keys = _unsupported_weight_keys(adapter)
    for key, a_key in sorted(adapter.saved_layers.items()):
        lora_keys.add(key)
        if key not in merged:
            missing.append(key)
            continue
        saved = adapter.weights.get(a_key)
        merged_w = merged.get(key)
        if tuple(saved.shape) != tuple(merged_w.shape):
            saved_bad.append(key)
            result["saved_layers"].append({"key": key, "status": "not_merged"})
            continue
        if torch.equal(saved.to(merged_w.dtype), merged_w):
            result["saved_layers"].append({"key": key, "status": "ok"})
        else:
            saved_bad.append(key)
            result["saved_layers"].append({"key": key, "status": "not_merged"})
    if saved_bad:
        causes.append(Cause("modules_to_save_not_merged", _cap(saved_bad)))

    # everything else must equal the base cast to the merged dtype.
    changed: list[str] = []
    checked = 0
    for key in base.keys():
        if key in lora_keys or key in skip_keys:
            continue
        if key not in merged:
            missing.append(key)
            continue
        if base.shape(key) != merged.shape(key):
            if any(v in key for v in _VOCAB_LAYERS):
                vocab_bad.append(key)
            else:
                shape_bad.append(key)
            continue
        merged_w = merged.get(key)
        checked += 1
        if not torch.equal(base.get(key).to(merged_w.dtype), merged_w):
            changed.append(key)
    result["non_lora"] = {"checked": checked, "changed": _cap(changed)}

    if changed:
        causes.append(Cause("non_lora_changed", _cap(changed), {"count": len(changed)}))
    if vocab_bad:
        causes.append(Cause("vocab_resized_without_saved_embeddings", _cap(vocab_bad),
                            {"merged_shapes": {k: list(merged.shape(k)) for k in _cap(vocab_bad)}}))
    if shape_bad:
        causes.append(Cause("shape_mismatch", _cap(shape_bad)))
    if missing:
        causes.append(Cause("missing_tensor", _cap(missing), {"count": len(missing)}))
    return result, causes


__all__ = ["check_weights", "classify_module", "ulp", "LoraModule"]
