"""English / Korean messages: same keys, language precedence, both languages on screen."""

from __future__ import annotations

import re
import string
from pathlib import Path

import pytest

from lora_mergecheck import i18n
from lora_mergecheck.cli import main
from lora_mergecheck.report import render_report
from lora_mergecheck.verdict import SEVERITY

SRC = Path(__file__).resolve().parent.parent / "src" / "lora_mergecheck"
HANGUL = re.compile(r"[가-힣]")


def test_message_keys_match():
    assert set(i18n.EN) == set(i18n.KO)
    assert all(i18n.EN[k].strip() and i18n.KO[k].strip() for k in i18n.EN)


def test_placeholders_match():
    def fields(text: str) -> set[str]:
        return {name for _, name, _, _ in string.Formatter().parse(text) if name}

    mismatched = [k for k in i18n.EN if fields(i18n.EN[k]) != fields(i18n.KO[k])]
    assert mismatched == []


def test_every_cause_has_text_in_both_languages():
    for code in SEVERITY:
        assert f"cause.{code}" in i18n.EN and f"rec.{code}" in i18n.EN


def test_keys_used_in_code_exist():
    used = set()
    for path in SRC.glob("*.py"):
        used |= set(re.findall(r'"((?:cli|err|hint|report|demo|verdict|error)\.[a-zA-Z_.]+)"', path.read_text()))
    assert used - set(i18n.EN) == set()


def test_unused_messages_do_not_pile_up():
    text = "\n".join(p.read_text() for p in SRC.glob("*.py") if p.name != "i18n.py")
    dynamic = ("cause.", "rec.", "demo.case.", "demo.expect.", "verdict.", "report.skip.", "err.", "hint.")
    unused = [k for k in i18n.EN if not k.startswith(dynamic) and f'"{k}"' not in text and f"'{k}'" not in text]
    assert unused == []


def test_lang_precedence():
    env = {"LORA_MERGECHECK_LANG": "ko", "LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"}
    assert i18n.resolve_lang("en", env) == "en"  # --lang wins
    assert i18n.resolve_lang(None, env) == "ko"  # then the tool variable
    assert i18n.resolve_lang(None, {"LC_ALL": "ko_KR.UTF-8", "LANG": "en_US.UTF-8"}) == "ko"
    assert i18n.resolve_lang(None, {"LC_ALL": "en_US.UTF-8", "LC_MESSAGES": "ko_KR"}) == "en"  # LC_ALL first
    assert i18n.resolve_lang(None, {"LC_MESSAGES": "ko_KR.UTF-8", "LANG": "en_US"}) == "ko"
    assert i18n.resolve_lang(None, {"LANG": "ko_KR.UTF-8"}) == "ko"
    assert i18n.resolve_lang(None, {"LANG": "fr_FR.UTF-8"}) == "en"  # not Korean -> English
    assert i18n.resolve_lang(None, {"LORA_MERGECHECK_LANG": "xx", "LANG": "ko_KR"}) == "ko"  # invalid is ignored
    assert i18n.resolve_lang(None, {}) == "en"


def help_text(argv, capsys, env=None, monkeypatch=None):
    with pytest.raises(SystemExit) as info:
        main(argv)
    assert info.value.code == 0
    return capsys.readouterr().out


def test_help_in_both_languages(capsys, monkeypatch):
    for var in ("LORA_MERGECHECK_LANG", "LC_ALL", "LC_MESSAGES", "LANG"):
        monkeypatch.delenv(var, raising=False)
    ko = help_text(["--lang", "ko", "check", "--help"], capsys)
    en = help_text(["--lang", "en", "check", "--help"], capsys)
    assert HANGUL.search(ko) and not HANGUL.search(en)
    assert "합친 체크포인트" in ko and "Merged checkpoint" in en
    assert HANGUL.search(help_text(["check", "--help", "--lang=ko"], capsys))  # flag after the subcommand
    monkeypatch.setenv("LORA_MERGECHECK_LANG", "ko")
    assert HANGUL.search(help_text(["--help"], capsys))
    assert not HANGUL.search(help_text(["--lang", "en", "--help"], capsys))  # --lang beats the variable


def test_result_table_in_both_languages(cases, capsys):
    argv = ["check", str(cases / "base"), str(cases / "adapter"), str(cases / "merged_dropped_o_proj"),
            "--layers", "weights"]
    assert main(["--lang", "ko", *argv]) == 1
    ko = capsys.readouterr().out
    assert main(["--lang", "en", *argv]) == 1
    en = capsys.readouterr().out
    assert "판정" in ko and "Verdict" in en and "Verdict" not in ko
    assert "update_not_applied" in ko and "update_not_applied" in en  # codes stay stable
    assert not HANGUL.search(en)


def test_errors_in_both_languages(work, capsys):
    argv = ["check", str(work / "x"), str(work / "y"), str(work / "z")]
    assert main(["--lang", "ko", *argv]) == 2
    assert capsys.readouterr().err.startswith("오류:")
    assert main(["--lang", "en", *argv]) == 2
    assert capsys.readouterr().err.startswith("error:")


def test_report_to_dict_language(cases):
    from lora_mergecheck import check_merge

    report = check_merge(cases / "base", cases / "adapter", cases / "merged_dropped_o_proj", layers="weights")
    assert HANGUL.search(report.to_dict("ko")["recommendations"][0]["text"])
    assert not HANGUL.search(report.to_dict("en")["recommendations"][0]["text"])
    assert render_report(report, "ko") != render_report(report, "en")


def test_korean_help_and_usage_errors_have_no_english_argparse_text(capsys):
    ko = help_text(["--lang", "ko", "check", "--help"], capsys)
    top = help_text(["--lang", "ko", "--help"], capsys)
    for text in (ko, top):
        for english in ("usage:", "positional arguments", "options:", "show this help message"):
            assert english not in text
        assert "사용법:" in text and "위치 인자" in text and "옵션:" in text
    with pytest.raises(SystemExit) as info:
        main(["--lang", "ko", "check"])
    err = capsys.readouterr().err
    assert info.value.code == 2
    assert "사용법 오류" in err and "the following arguments" not in err and "error:" not in err
    with pytest.raises(SystemExit):
        main(["--lang", "ko", "check", "a", "b", "c", "--layers", "x"])
    assert "허용되지 않는 값" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["--lang", "en", "check"])
    assert "the following arguments are required" in capsys.readouterr().err
