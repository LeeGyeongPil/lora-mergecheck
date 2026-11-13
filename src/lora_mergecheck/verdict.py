"""Cause codes, severities, the combined verdict and the ``Report`` dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field

from .i18n import t

SCHEMA_VERSION = 1

#: severity of each cause code. FAIL > WARN > INCONCLUSIVE.
SEVERITY: dict[str, str] = {
    "update_not_applied": "FAIL",
    "update_mismatch": "FAIL",
    "leftover_lora_keys": "FAIL",
    "non_lora_changed": "FAIL",
    "modules_to_save_not_merged": "FAIL",
    "vocab_resized_without_saved_embeddings": "FAIL",
    "shape_mismatch": "FAIL",
    "missing_tensor": "FAIL",
    "output_drift": "FAIL",
    "quantized_checkpoint": "WARN",
    "low_precision_update_loss": "WARN",
    "unsupported_layer": "WARN",
    "outputs_unavailable": "WARN",
    "adapter_effect_too_small": "INCONCLUSIVE",
}
_RANK = {"FAIL": 0, "WARN": 1, "INCONCLUSIVE": 2}


@dataclass
class Cause:
    """One reason the verdict is not a clean PASS."""

    code: str
    modules: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    @property
    def severity(self) -> str:
        return SEVERITY[self.code]


def combine(causes: list[Cause]) -> str:
    """PASS when there are no causes, else the most severe cause level."""
    if not causes:
        return "PASS"
    return min((c.severity for c in causes), key=lambda s: _RANK[s])


def sort_causes(causes: list[Cause]) -> list[Cause]:
    return sorted(causes, key=lambda c: _RANK[c.severity])


@dataclass
class Report:
    """Result of :func:`lora_mergecheck.check_merge`."""

    verdict: str
    causes: list[Cause]
    weights: dict
    outputs: dict | None
    inputs: dict
    versions: dict

    def to_dict(self, lang: str = "en") -> dict:
        """JSON-ready dict; recommendation and meaning texts use ``lang``."""
        codes = [c.code for c in self.causes]
        return {
            "schema_version": SCHEMA_VERSION,
            "verdict": self.verdict,
            "causes": [
                {
                    "code": c.code,
                    "severity": c.severity,
                    "meaning": t(f"cause.{c.code}", lang),
                    "modules": c.modules,
                    "details": c.details,
                }
                for c in self.causes
            ],
            "weights": self.weights,
            "outputs": self.outputs,
            "recommendations": [{"code": code, "text": t(f"rec.{code}", lang)} for code in codes],
            "inputs": self.inputs,
            "versions": self.versions,
        }

    def cause_codes(self) -> list[str]:
        return [c.code for c in self.causes]

    def cause(self, code: str) -> Cause | None:
        return next((c for c in self.causes if c.code == code), None)
