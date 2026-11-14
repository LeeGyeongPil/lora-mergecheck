"""``check_merge``: the one public entry point (the CLI calls this too)."""

from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path

import torch

from .adapter import load_adapter
from .checkpoint import Checkpoint
from .errors import InputError
from .verdict import Cause, Report, combine, sort_causes
from .weights import check_weights

LAYERS = ("all", "weights", "outputs")
#: weight-layer causes after which the model cannot be loaded meaningfully
_BLOCKING = {"leftover_adapter_config", "leftover_lora_keys", "missing_tensor", "shape_mismatch", "vocab_resized_without_saved_embeddings"}


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _validate(layers: str, num_prompts: int, prompt_len: int, gen_tokens: int,
              max_relative_drift: float, warn_lost_ratio: float) -> None:
    problems = []
    if layers not in LAYERS:
        problems.append(f"layers must be one of {', '.join(LAYERS)}, got {layers!r}")
    if num_prompts < 1:
        problems.append("num_prompts must be >= 1")
    if prompt_len < 2:
        problems.append("prompt_len must be >= 2")
    if gen_tokens < 0:
        problems.append("gen_tokens must be >= 0")
    if max_relative_drift <= 0:
        problems.append("max_relative_drift must be > 0")
    if not 0 < warn_lost_ratio <= 1:
        problems.append("warn_lost_ratio must be in (0, 1]")
    if problems:
        raise InputError("bad_option", reason="; ".join(problems))


def _validate_device(device: str) -> None:
    """Refuse an unusable ``--device`` before any work (a CI job must not pass without the output layer)."""
    try:
        dev = torch.device(device)
        if dev.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available on this machine")
            if dev.index is not None and dev.index >= torch.cuda.device_count():
                raise RuntimeError(f"only {torch.cuda.device_count()} CUDA device(s) found")
        elif dev.type == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS is not available on this machine")
        elif dev.type != "cpu":
            torch.zeros(1, device=dev)  # any other backend: it must at least hold a tensor
    except (RuntimeError, ValueError, TypeError, AssertionError) as exc:
        raise InputError("bad_device", device=device, reason=str(exc).splitlines()[0]) from exc


def check_merge(
    base: str | Path,
    adapter: str | Path,
    merged: str | Path,
    *,
    layers: str = "all",
    prompts: str | Path | None = None,
    num_prompts: int = 8,
    prompt_len: int = 32,
    gen_tokens: int = 16,
    max_relative_drift: float = 0.10,
    warn_lost_ratio: float = 0.25,
    device: str = "cpu",
    progress=None,
) -> Report:
    """Check that ``merged`` equals ``base`` + ``adapter``.

    ``base`` and ``merged`` are folders (or single ``.safetensors`` files for the
    weight layer); ``adapter`` is a PEFT LoRA folder. ``layers`` is ``"weights"``
    (no model is loaded), ``"outputs"`` or ``"all"``.

    Raises:
        InputError: unusable inputs or options (the CLI exits with code 2).
    """
    _validate(layers, num_prompts, prompt_len, gen_tokens, max_relative_drift, warn_lost_ratio)
    base_ckpt = Checkpoint(base)
    merged_ckpt = Checkpoint(merged)
    lora = load_adapter(adapter)
    run_outputs = layers in ("all", "outputs")
    if run_outputs:
        _validate_device(device)
    if run_outputs:
        for ckpt in (base_ckpt, merged_ckpt):
            if ckpt.is_file:
                raise InputError("file_for_outputs", path=str(ckpt.path))
            if ckpt.config() is None:
                raise InputError("no_config", path=str(ckpt.path))

    causes: list[Cause] = []
    # A leftover adapter_config.json makes transformers treat the folder as an adapter and fail to load it.
    if not merged_ckpt.is_file and (merged_ckpt.root / "adapter_config.json").is_file():
        causes.append(Cause("leftover_adapter_config", ["adapter_config.json"], {"file": "adapter_config.json"}))
    weights: dict = {"status": "skipped", "skipped_reason": "layers"}
    if layers in ("all", "weights"):
        weights, w_causes = check_weights(base_ckpt, lora, merged_ckpt, warn_lost_ratio=warn_lost_ratio)
        causes.extend(w_causes)

    outputs: dict | None = None
    if run_outputs:
        blocked = [c.code for c in causes if c.code in _BLOCKING]
        if blocked:
            outputs = {"status": "skipped", "skipped_reason": "structure"}
        else:
            from .outputs import check_outputs  # imported late: weights-only runs never import transformers

            try:
                outputs, o_causes = check_outputs(
                    Path(base), Path(adapter), Path(merged),
                    prompts_file=Path(prompts) if prompts else None, num_prompts=num_prompts,
                    prompt_len=prompt_len, gen_tokens=gen_tokens, max_relative_drift=max_relative_drift,
                    device=device, progress=progress)
                causes.extend(o_causes)
            except InputError:
                raise
            except Exception as exc:
                if layers == "outputs":
                    raise InputError("output_load", reason=str(exc)) from exc
                outputs = {"status": "skipped", "skipped_reason": "error", "error": str(exc)}
                causes.append(Cause("outputs_unavailable", [], {"error": str(exc)}))

    causes = sort_causes(causes)
    versions = {"lora_mergecheck": _version("lora-mergecheck"), "python": sys.version.split()[0],
                "torch": _version("torch"), "safetensors": _version("safetensors")}
    if "transformers" in sys.modules:
        versions["transformers"] = _version("transformers")
        versions["peft"] = _version("peft")
    inputs = {"base": str(base), "adapter": str(adapter), "merged": str(merged), "layers": layers}
    return Report(combine(causes), causes, weights, outputs, inputs, versions)
