"""CLI: exit codes, input/output safety, JSON report, demo, version."""

from __future__ import annotations

import json
import re
import tomllib
from importlib import metadata
from pathlib import Path

import pytest
from conftest import copy_dir

import lora_mergecheck
from lora_mergecheck import demo
from lora_mergecheck.cli import main

ROOT = Path(__file__).resolve().parent.parent


def run(argv, capsys):
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def check_args(cases, merged="merged_fp32_normal", adapter="adapter", *extra):
    return ["--lang", "en", "check", str(cases / "base"), str(cases / adapter), str(cases / merged),
            "--layers", "weights", *extra]


def test_pass_exit_code_0(cases, capsys):
    code, out, _ = run(check_args(cases), capsys)
    assert code == 0 and "Verdict: PASS" in out


def test_fail_exit_code_1(cases, capsys):
    code, out, _ = run(check_args(cases, "merged_dropped_o_proj"), capsys)
    assert code == 1
    assert "update_not_applied" in out
    assert "model.layers.0.self_attn.o_proj.weight" in out
    assert "model.layers.1.self_attn.o_proj.weight" in out


def test_strict_turns_warn_into_exit_1(cases, capsys):
    args = check_args(cases, "merged_bf16_small_update", "adapter_small")
    code, out, _ = run(args, capsys)
    assert code == 0 and "WARN" in out and "low_precision_update_loss" in out
    code, _, _ = run([*args, "--strict"], capsys)
    assert code == 1


def test_report_path_equal_to_input_is_refused(cases, work, capsys):
    merged = copy_dir(cases / "merged_fp32_normal", work / "merged")
    weights_file = merged / "model.safetensors"
    before = weights_file.read_bytes()
    argv = ["check", str(cases / "base" / "model.safetensors"), str(cases / "adapter"), str(weights_file),
            "--layers", "weights", "--json", str(weights_file)]
    code, _, err = run(argv, capsys)
    assert code == 2 and "error:" in err
    assert weights_file.read_bytes() == before
    # also the folder itself
    code, _, _ = run(["check", str(cases / "base"), str(cases / "adapter"), str(merged),
                      "--layers", "weights", "--json", str(merged)], capsys)
    assert code == 2


@pytest.mark.parametrize("which", ["base", "adapter", "merged"])
def test_report_path_inside_input_is_refused(cases, work, capsys, which):
    dirs = {name: copy_dir(cases / folder, work / name)
            for name, folder in (("base", "base"), ("adapter", "adapter"), ("merged", "merged_fp32_normal"))}
    target = dirs[which] / "report.json"
    argv = ["check", str(dirs["base"]), str(dirs["adapter"]), str(dirs["merged"]),
            "--layers", "weights", "--json", str(target)]
    code, _, err = run(argv, capsys)
    assert code == 2 and "error:" in err and "hint:" in err
    assert not target.exists()
    # a path that merely reaches inside through ".." is refused as well
    sneaky = dirs[which] / "sub" / ".." / "report.json"
    code, _, _ = run([*argv[:-1], str(sneaky)], capsys)
    assert code == 2 and not target.exists()


def test_missing_input_exit_2(cases, work, capsys):
    code, _, err = run(["check", str(work / "nope"), str(cases / "adapter"), str(cases / "merged_fp32_normal")],
                       capsys)
    assert code == 2 and "path not found" in err
    no_cfg = copy_dir(cases / "adapter", work / "adapter")
    (no_cfg / "adapter_config.json").unlink()
    code, _, err = run(["check", str(cases / "base"), str(no_cfg), str(cases / "merged_fp32_normal"),
                        "--layers", "weights"], capsys)
    assert code == 2 and "adapter_config.json" in err


def test_output_layer_needs_folders_with_config(cases, work, capsys):
    merged = copy_dir(cases / "merged_fp32_normal", work / "merged")
    (merged / "config.json").unlink()
    code, _, err = run(["check", str(cases / "base"), str(cases / "adapter"), str(merged)], capsys)
    assert code == 2 and "config.json" in err
    code, _, err = run(["check", str(cases / "base"), str(cases / "adapter"),
                        str(cases / "merged_fp32_normal" / "model.safetensors")], capsys)
    assert code == 2 and "single file" in err


@pytest.mark.parametrize("lang, phrase", [("en", "no tokenizer files"), ("ko", "토크나이저 파일이 없습니다")])
def test_prompts_without_tokenizer_in_base_is_exit_2(cases, work, capsys, lang, phrase):
    prompts = work / "p.txt"
    prompts.write_text("hello world\n")
    argv = ["--lang", lang, "check", str(cases / "base"), str(cases / "adapter"),
            str(cases / "merged_fp32_normal"), "--layers", "outputs", "--prompts", str(prompts)]
    code, _, err = run(argv, capsys)
    assert code == 2 and phrase in err
    assert "sentencepiece" not in err and "tiktoken" not in err


def test_bad_option_value_exit_2(cases, capsys):
    code, _, err = run(check_args(cases, "merged_fp32_normal", "adapter", "--warn-lost-ratio", "5"), capsys)
    assert code == 2 and "warn_lost_ratio" in err


def test_json_report_fields(cases, work, capsys):
    path = work / "out" / "report.json"
    code, _, _ = run(check_args(cases, "merged_dropped_o_proj", "adapter", "--json", str(path)), capsys)
    assert code == 1
    data = json.loads(path.read_text(encoding="utf-8"))
    for field in ("schema_version", "verdict", "causes", "weights", "outputs", "recommendations", "versions"):
        assert field in data, field
    assert data["schema_version"] == 1 and data["verdict"] == "FAIL"
    assert data["causes"][0]["code"] == "update_not_applied"
    assert data["recommendations"][0]["code"] == "update_not_applied"
    assert data["versions"]["lora_mergecheck"] == lora_mergecheck.__version__
    assert data["weights"]["counts"]["not_applied"] == 2


def test_full_check_runs_both_layers(cases, work, capsys):
    path = work / "report.json"
    code, out, err = run(["--lang", "en", "check", str(cases / "base"), str(cases / "adapter"),
                          str(cases / "merged_fp32_normal"), "--json", str(path)], capsys)
    assert code == 0
    assert "Outputs layer" in out and "relative drift" in out
    assert "loading merged checkpoint" in err  # progress goes to stderr, not stdout
    data = json.loads(path.read_text())
    assert data["outputs"]["status"] == "checked" and "transformers" in data["versions"]


def test_demo_cases_match_expected(capsys):
    results, _ = demo.run_demo()
    assert [r.spec.key for r in results] == [
        "fp32_normal", "dropped_o_proj", "bf16_small_update", "leftover_lora", "mts_not_merged"]
    assert all(r.ok for r in results), [(r.spec.key, r.report.verdict, r.report.cause_codes()) for r in results]
    by_key = {r.spec.key: r.report for r in results}
    assert by_key["fp32_normal"].verdict == "PASS"
    assert by_key["dropped_o_proj"].causes[0].code == "update_not_applied"
    assert by_key["bf16_small_update"].cause("low_precision_update_loss") is not None
    assert by_key["leftover_lora"].causes[0].code == "leftover_lora_keys"
    assert by_key["mts_not_merged"].causes[0].code == "modules_to_save_not_merged"


def test_demo_command_output_and_out_dir(work, capsys):
    out_dir = work / "demo"
    code, out, _ = run(["--lang", "en", "demo", "--out", str(out_dir)], capsys)
    assert code == 0
    assert "practice mock" in out
    assert len(re.findall(r"\bok$", out, flags=re.M)) == 5
    assert (out_dir / "merged_dropped_o_proj" / "model.safetensors").is_file()
    code, _, err = run(["demo", "--out", str(out_dir)], capsys)  # not empty any more
    assert code == 2 and "not empty" in err


def test_version_matches_package(capsys):
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0
    printed = capsys.readouterr().out.strip()
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert printed == f"lora-mergecheck {declared}"
    assert metadata.version("lora-mergecheck") == declared
    assert lora_mergecheck.__version__ == declared


# ---- input-error regressions (broken files, unwritable report, unusable device) ----
def test_empty_safetensors_is_input_error(cases, work, capsys):
    empty = work / "empty.safetensors"
    empty.write_bytes(b"")
    code, _, err = run(check_args(cases, str(empty)), capsys)
    assert code == 2 and "empty.safetensors" in err and "hint:" in err and "Traceback" not in err


def test_garbage_safetensors_in_folder_is_input_error(cases, work, capsys):
    bad = work / "merged"
    bad.mkdir()
    (bad / "model.safetensors").write_text("this is plain text, not safetensors " * 10)
    code, _, err = run(check_args(cases, str(bad)), capsys)
    assert code == 2 and "safe_serialization=True" in err


def test_broken_adapter_weights_is_input_error(cases, work, capsys):
    adapter = copy_dir(cases / "adapter", work / "adapter")
    (adapter / "adapter_model.safetensors").write_bytes(b"\x00\x01")
    code, _, err = run(check_args(cases, "merged_fp32_normal", str(adapter)), capsys)
    assert code == 2 and "adapter_model.safetensors" in err


def test_json_path_that_is_a_folder_is_refused_before_the_check(cases, work, capsys):
    code, out, err = run(check_args(cases, "merged_fp32_normal", "adapter", "--json", str(work)), capsys)
    assert code == 2 and out == "" and "folder" in err


def test_json_path_in_unwritable_place_is_refused(cases, capsys):
    code, out, err = run(check_args(cases, "merged_fp32_normal", "adapter", "--json", "/nonexistent_dir/r.json"),
                         capsys)
    assert code == 2 and out == "" and "hint:" in err


def test_json_write_failure_after_check_is_exit_2(cases, work, capsys, monkeypatch):
    def boom(*_a, **_k):
        raise PermissionError("denied")

    monkeypatch.setattr("lora_mergecheck.cli.write_json", boom)
    code, _, err = run(check_args(cases, "merged_fp32_normal", "adapter", "--json", str(work / "r.json")), capsys)
    assert code == 2 and "denied" in err


@pytest.mark.parametrize("layers", ["all", "outputs"])
def test_unusable_device_is_exit_2_for_every_layer_choice(cases, capsys, layers):
    argv = ["--lang", "en", "check", str(cases / "base"), str(cases / "adapter"), str(cases / "merged_fp32_normal"),
            "--layers", layers, "--device", "nonsense:9"]
    code, out, err = run(argv, capsys)
    assert code == 2 and "nonsense:9" in err and "Verdict" not in out
