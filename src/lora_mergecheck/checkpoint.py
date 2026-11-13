"""Read safetensors checkpoints tensor by tensor (single file or sharded)."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import SafetensorError, safe_open

from .errors import InputError

# Key suffixes that only appear in bitsandbytes / GPTQ / AWQ style quantized files.
_QUANT_SUFFIXES = (".SCB", ".absmax", ".quant_map", ".quant_state", ".qweight", ".qzeros", ".g_idx")


class Checkpoint:
    """A set of safetensors files addressed as one key -> tensor mapping.

    Headers are read once; tensors are loaded one at a time with :meth:`get`
    so a large model is never held in memory.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise InputError("path_not_found", path=str(path))
        self.root = self.path if self.path.is_dir() else self.path.parent
        self.is_file = self.path.is_file()
        self._files = self._find_files()
        self._key_file: dict[str, Path] = {}
        self._dtype: dict[str, str] = {}
        self._shape: dict[str, tuple[int, ...]] = {}
        for file in self._files:
            try:
                with safe_open(str(file), "pt") as handle:
                    for key in handle.keys():
                        sl = handle.get_slice(key)
                        self._key_file[key] = file
                        self._dtype[key] = sl.get_dtype()
                        self._shape[key] = tuple(sl.get_shape())
            except (SafetensorError, OSError) as exc:  # truncated save, plain text, unreadable file
                raise InputError("bad_safetensors", path=str(file), reason=str(exc)) from exc

    def _find_files(self) -> list[Path]:
        if self.is_file:
            return [self.path]
        index = self.path / "model.safetensors.index.json"
        if index.is_file():
            try:
                weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
                names = sorted(set(weight_map.values()))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise InputError("bad_index", path=str(index), reason=str(exc)) from exc
            files = [self.path / name for name in names]
            missing = [f for f in files if not f.is_file()]
            if missing:
                raise InputError("bad_index", path=str(index), reason=f"missing shard {missing[0].name}")
            return files
        single = self.path / "model.safetensors"
        if single.is_file():
            return [single]
        files = sorted(self.path.glob("*.safetensors"))
        if not files:
            raise InputError("no_safetensors", path=str(self.path))
        return files

    # -- tensor access -------------------------------------------------------
    def keys(self) -> list[str]:
        return list(self._key_file)

    def __contains__(self, key: object) -> bool:
        return key in self._key_file

    def dtype_name(self, key: str) -> str:
        """safetensors dtype name such as ``F32``, ``BF16``, ``F16``, ``I8``."""
        return self._dtype[key]

    def shape(self, key: str) -> tuple[int, ...]:
        return self._shape[key]

    def get(self, key: str) -> torch.Tensor:
        with safe_open(str(self._key_file[key]), "pt") as handle:
            return handle.get_tensor(key)

    # -- config / quantization ----------------------------------------------
    def config(self) -> dict | None:
        """Parsed ``config.json`` next to the weights, or None."""
        cfg = self.root / "config.json"
        if not cfg.is_file():
            return None
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def quantization_evidence(self) -> list[str]:
        """Short reasons why this checkpoint looks quantized (empty when it does not)."""
        reasons: list[str] = []
        cfg = self.config()
        if cfg and cfg.get("quantization_config"):
            method = cfg["quantization_config"].get("quant_method", "unknown")
            reasons.append(f"config.json quantization_config ({method})")
        suffix_hits = sorted({s for key in self._key_file for s in _QUANT_SUFFIXES if key.endswith(s)})
        if suffix_hits:
            reasons.append("quantization tensors: " + ", ".join(suffix_hits))
        int_weights = [k for k, d in self._dtype.items() if k.endswith(".weight") and d in ("I8", "U8")]
        if int_weights:
            reasons.append(f"integer weight tensors (e.g. {int_weights[0]})")
        return reasons
