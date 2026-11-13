"""Read a PEFT LoRA adapter folder: config, key mapping and the expected update ``s*B*A``."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .checkpoint import Checkpoint
from .errors import InputError, UnsupportedAdapterError

_PREFIX = "base_model.model."
_LORA_KEY = re.compile(r"^(?P<mod>.+)\.lora_(?P<ab>[AB])(?:\.[^.]+)?\.weight$")
_MTS = re.compile(r"\.modules_to_save\.[^.]+")


@dataclass(frozen=True)
class LoraModule:
    """One LoRA-adapted Linear layer."""

    name: str  # base module name, e.g. "model.layers.0.self_attn.q_proj"
    a_key: str
    b_key: str
    rank: int
    alpha: float
    scale: float
    fan_in_fan_out: bool

    @property
    def weight_key(self) -> str:
        return f"{self.name}.weight"


@dataclass
class Adapter:
    """Parsed adapter: LoRA modules, fully saved layers and unsupported keys."""

    path: Path
    config: dict
    weights: Checkpoint
    modules: dict[str, LoraModule] = field(default_factory=dict)
    saved_layers: dict[str, str] = field(default_factory=dict)  # base key -> adapter key
    unsupported: list[str] = field(default_factory=list)

    def delta_weight(self, module: LoraModule) -> torch.Tensor:
        """Expected update ``s*B@A`` in float32 (transposed for fan_in_fan_out layers)."""
        a = self.weights.get(module.a_key).float()
        b = self.weights.get(module.b_key).float()
        delta = module.scale * (b @ a)
        return delta.T.contiguous() if module.fan_in_fan_out else delta

    def saved_tensor(self, base_key: str) -> torch.Tensor:
        return self.weights.get(self.saved_layers[base_key])


def _pattern_value(patterns: dict, module: str, default: float) -> float:
    for pattern, value in patterns.items():
        if re.match(rf"(.*\.)?({pattern})$", module):
            return float(value)
    return default


def load_adapter(path: str | Path) -> Adapter:
    """Load ``adapter_config.json`` and index ``adapter_model.safetensors``."""
    root = Path(path)
    if not root.exists():
        raise InputError("path_not_found", path=str(path))
    if not root.is_dir():
        raise InputError("no_adapter_config", path=str(path))
    cfg_path = root / "adapter_config.json"
    if not cfg_path.is_file():
        raise InputError("no_adapter_config", path=str(root))
    weights_path = root / "adapter_model.safetensors"
    if not weights_path.is_file():
        raise InputError("no_adapter_weights", path=str(root))
    try:
        config = json.loads(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("top-level value is not an object")
        lora_alpha = float(config.get("lora_alpha", 8))
    except (OSError, ValueError, TypeError) as exc:
        raise InputError("bad_adapter_config", reason=str(exc)) from exc

    peft_type = str(config.get("peft_type", "LORA"))
    if peft_type.upper() != "LORA":
        raise InputError("not_lora", peft_type=peft_type)
    if config.get("use_dora"):
        raise UnsupportedAdapterError("unsupported_dora")

    adapter = Adapter(path=root, config=config, weights=Checkpoint(weights_path))
    use_rslora = bool(config.get("use_rslora"))
    fan_in_fan_out = bool(config.get("fan_in_fan_out"))
    alpha_pattern = config.get("alpha_pattern") or {}

    pairs: dict[str, dict[str, str]] = {}
    for key in adapter.weights.keys():
        name = key[len(_PREFIX):] if key.startswith(_PREFIX) else key
        match = _LORA_KEY.match(name)
        if match:
            pairs.setdefault(match["mod"], {})[match["ab"]] = key
        elif "lora_" in name or "trainable_tokens" in name or "magnitude" in name:
            adapter.unsupported.append(key)
        elif name.endswith(".weight") or name.endswith(".bias"):
            adapter.saved_layers[_MTS.sub("", name)] = key
        else:
            adapter.unsupported.append(key)

    for module, ab in sorted(pairs.items()):
        a_key, b_key = ab.get("A"), ab.get("B")
        if a_key is None or b_key is None:
            adapter.unsupported.extend(k for k in (a_key, b_key) if k)
            continue
        if len(adapter.weights.shape(a_key)) != 2 or len(adapter.weights.shape(b_key)) != 2:
            adapter.unsupported.extend([a_key, b_key])  # Conv LoRA
            continue
        rank = adapter.weights.shape(a_key)[0]
        alpha = _pattern_value(alpha_pattern, module, lora_alpha)
        scale = alpha / math.sqrt(rank) if use_rslora else alpha / rank
        adapter.modules[module] = LoraModule(module, a_key, b_key, rank, alpha, scale, fan_in_fan_out)
    return adapter
