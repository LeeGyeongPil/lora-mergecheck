"""Plain fixed-width text rendering (no colour) and JSON serialisation."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from .i18n import t
from .verdict import Report

_MODULES_SHOWN = 8


def _width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def pad(text: str, width: int) -> str:
    """Left-align ``text`` in ``width`` terminal columns (Korean characters are two columns wide)."""
    return text + " " * max(0, width - _width(text))


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _sci(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2e}"


def _num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_report(report: Report, lang: str = "en") -> str:
    """Human-readable result: layers, verdict, causes with module names, recommendations."""
    lines = [t("report.inputs", lang, base=report.inputs["base"], adapter=report.inputs["adapter"],
               merged=report.inputs["merged"])]
    w = report.weights
    if w.get("status") == "checked":
        lines.append(t("report.weights_header", lang, n=sum(w["counts"].values()),
                       dtype=w.get("merged_dtype") or "n/a"))
        lines.append(t("report.weights_counts", lang, **{k: w["counts"].get(k, 0) for k in
                       ("exact", "rounding", "mismatch", "not_applied", "missing")}))
        lines.append(t("report.weights_loss", lang, lost=_pct(w["lost_update_ratio"]),
                       retention=_pct(w["update_retention"]), resid=_sci(w["max_residual"])))
    else:
        lines.append(t("report.weights_skipped", lang, reason=_skip_reason(w, lang)))
    o = report.outputs
    if o is not None and o.get("status") == "checked":
        lines.append(t("report.outputs_header", lang, n_prompts=o["n_prompts"], n_tokens=o["n_tokens"],
                       device=o["device"]))
        lines.append(t("report.outputs_effect", lang, effect=_sci(o["adapter_effect_kl"]),
                       floor=_sci(o["numeric_floor_kl"])))
        lines.append(t("report.outputs_drift", lang, drift=_sci(o["drift_kl_mean"]),
                       drift_max=_sci(o["drift_kl_max"]), rel=_num(o["relative_drift"])))
        lines.append(t("report.outputs_agree", lang, top1=_pct(o["top1_agree"]),
                       greedy=_pct(o["greedy_agree"]), logit=_sci(o["max_logit_diff"])))
    elif o is not None:
        lines.append(t("report.outputs_skipped", lang, reason=_skip_reason(o, lang)))
    lines.append("")
    lines.append(t("report.verdict", lang, verdict=t(f"verdict.{report.verdict}", lang)))
    if report.causes:
        lines.append(t("report.causes", lang))
        for cause in report.causes:
            lines.append(f"  {cause.severity} {cause.code}: {t(f'cause.{cause.code}', lang)}")
            for name in cause.modules[:_MODULES_SHOWN]:
                lines.append(f"      {name}")
            extra = int(cause.details.get("count", len(cause.modules))) - min(len(cause.modules), _MODULES_SHOWN)
            if extra > 0:
                lines.append(t("report.modules_more", lang, n=extra))
        lines.append(t("report.recommendations", lang))
        for cause in report.causes:
            lines.append(f"  - {cause.code}: {t(f'rec.{cause.code}', lang)}")
    return "\n".join(lines)


def _skip_reason(section: dict, lang: str) -> str:
    reason = section.get("skipped_reason") or "error"
    key = f"report.skip.{reason}"
    from .i18n import EN

    if key in EN:
        return t(key, lang, reason=section.get("error", ""))
    return reason


def render_demo(results, lang: str = "en") -> str:
    """Table with one row per demo case."""
    head = (pad(t("demo.col.case", lang), 30) + pad(t("demo.col.expected", lang), 34)
            + pad(t("demo.col.verdict", lang), 48) + t("demo.col.match", lang))
    rows = [head, "-" * _width(head)]
    for r in results:
        expected = t(f"demo.expect.{r.spec.expected or 'PASS'}", lang)
        codes = r.report.cause_codes()
        verdict = r.report.verdict + (" " + ", ".join(codes) if codes else "")
        rows.append(pad(t(f"demo.case.{r.spec.key}", lang), 30) + pad(expected, 34)
                    + pad(verdict, 48) + ("ok" if r.ok else "MISMATCH"))
    return "\n".join(rows)


def write_json(report: Report, path: Path, lang: str = "en") -> None:
    """Write the report as UTF-8 JSON, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(lang), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
