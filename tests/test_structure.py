"""Structure checks: leftover keys, modules_to_save, vocab, quantization, shards, DoRA."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
import torch
from conftest import copy_dir, edit_json, read_state, write_state

from lora_mergecheck import InputError, UnsupportedAdapterError, check_merge


def weights_only(base, adapter, merged, **kw):
    return check_merge(base, adapter, merged, layers="weights", **kw)


def test_leftover_lora_keys(cases):
    report = weights_only(cases / "base", cases / "adapter", cases / "merged_leftover_lora")
    cause = report.cause("leftover_lora_keys")
    assert report.verdict == "FAIL"
    assert cause is not None and all(".lora_" in m or ".base_layer." in m for m in cause.modules)
    assert report.weights["status"] == "skipped"
    assert report.causes[0].code == "leftover_lora_keys"
    assert len(report.causes) == 1  # no flood of follow-up errors


def test_modules_to_save_ok_and_missing(cases):
    ok = weights_only(cases / "base", cases / "adapter_mts", cases / "merged_mts_ok")
    assert ok.verdict == "PASS"
    assert ok.weights["saved_layers"] == [{"key": "lm_head.weight", "status": "ok"}]
    bad = weights_only(cases / "base", cases / "adapter_mts", cases / "merged_mts_not_merged")
    assert bad.verdict == "FAIL"
    assert [m for m in bad.cause("modules_to_save_not_merged").modules] == ["lm_head.weight"]
    assert bad.cause("non_lora_changed") is None  # lm_head is judged against the adapter, not the base


def test_quantized_checkpoint_detected(cases, work):
    merged = copy_dir(cases / "merged_fp32_normal", work / "merged")
    edit_json(merged / "config.json", quantization_config={"quant_method": "bitsandbytes", "load_in_4bit": True})
    report = weights_only(cases / "base", cases / "adapter", merged)
    cause = report.cause("quantized_checkpoint")
    assert cause is not None
    assert report.verdict == "WARN"
    assert report.weights["status"] == "skipped" and report.weights["skipped_reason"] == "quantized"
    assert "bitsandbytes" in json.dumps(cause.details)
    assert "quantized_checkpoint" in [r["code"] for r in report.to_dict("en")["recommendations"]]


def test_quantized_tensor_keys_detected(cases, work):
    state = read_state(cases / "merged_fp32_normal")
    state["model.layers.0.self_attn.q_proj.weight.absmax"] = torch.zeros(4)
    merged = write_state(state, work / "merged", cases / "base")
    report = weights_only(cases / "base", cases / "adapter", merged)
    assert report.cause("quantized_checkpoint") is not None


def _shard(folder, n_shards=2):
    state = read_state(folder)
    keys = sorted(state)
    weight_map = {}
    (folder / "model.safetensors").unlink()
    from safetensors.torch import save_file

    for i in range(n_shards):
        name = f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors"
        part = {k: state[k].contiguous() for k in keys[i::n_shards]}
        save_file(part, str(folder / name))
        weight_map.update({k: name for k in part})
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


def test_sharded_equals_single(cases, work):
    single = weights_only(cases / "base", cases / "adapter", cases / "merged_bf16_normal")
    sharded_dir = copy_dir(cases / "merged_bf16_normal", work / "sharded")
    _shard(sharded_dir)
    assert not (sharded_dir / "model.safetensors").exists()
    sharded = weights_only(cases / "base", cases / "adapter", sharded_dir)
    assert sharded.weights == single.weights
    assert sharded.cause_codes() == single.cause_codes()
    # the base can be sharded too
    base_dir = copy_dir(cases / "base", work / "base_sharded")
    _shard(base_dir, 3)
    both = weights_only(base_dir, cases / "adapter", sharded_dir)
    assert both.weights == single.weights


def test_single_safetensors_file_input(cases):
    report = weights_only(cases / "base" / "model.safetensors", cases / "adapter",
                          cases / "merged_fp32_normal" / "model.safetensors")
    assert report.verdict == "PASS"


def test_dora_is_rejected(cases, work):
    adapter = copy_dir(cases / "adapter", work / "adapter")
    edit_json(adapter / "adapter_config.json", use_dora=True)
    with pytest.raises(UnsupportedAdapterError):
        weights_only(cases / "base", adapter, cases / "merged_fp32_normal")
    assert issubclass(UnsupportedAdapterError, InputError)


def test_vocab_resized_without_saved_embeddings(cases, work):
    state = read_state(cases / "merged_fp32_normal")
    emb = state["model.embed_tokens.weight"]
    state["model.embed_tokens.weight"] = torch.cat([emb, torch.zeros(4, emb.shape[1])])
    merged = write_state(state, work / "merged", cases / "base")
    report = weights_only(cases / "base", cases / "adapter", merged)
    assert report.cause("vocab_resized_without_saved_embeddings") is not None
    assert report.cause("shape_mismatch") is None


def test_missing_tensor_and_non_lora_change(cases, work):
    state = read_state(cases / "merged_fp32_normal")
    del state["model.layers.1.mlp.down_proj.weight"]
    state["model.norm.weight"] = state["model.norm.weight"] + 0.5
    merged = write_state(state, work / "merged", cases / "base")
    report = weights_only(cases / "base", cases / "adapter", merged)
    assert report.cause("missing_tensor").modules == ["model.layers.1.mlp.down_proj.weight"]
    assert report.cause("non_lora_changed").modules == ["model.norm.weight"]
    assert report.verdict == "FAIL"


def test_unsupported_adapter_layers_are_not_silently_passed(cases, work):
    from safetensors import safe_open
    from safetensors.torch import save_file

    adapter = copy_dir(cases / "adapter", work / "adapter")
    path = adapter / "adapter_model.safetensors"
    with safe_open(str(path), "pt") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    tensors["base_model.model.model.embed_tokens.lora_embedding_A"] = torch.zeros(8, 512)
    save_file(tensors, str(path))
    report = weights_only(cases / "base", adapter, cases / "merged_fp32_normal")
    assert report.cause("unsupported_layer") is not None
    assert report.verdict == "WARN"


def test_weights_layer_does_not_import_transformers(cases):
    code = (
        "import json, sys\n"
        "from lora_mergecheck.cli import main\n"
        "rc = main(['check', sys.argv[1], sys.argv[2], sys.argv[3], '--layers', 'weights'])\n"
        "print(json.dumps({'rc': rc, 'transformers': 'transformers' in sys.modules,\n"
        "                  'peft': 'peft' in sys.modules}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-I", "-c", code, str(cases / "base"), str(cases / "adapter"),
         str(cases / "merged_fp32_normal")],
        capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode == 0, proc.stderr
    last = json.loads(proc.stdout.strip().splitlines()[-1])
    assert last == {"rc": 0, "transformers": False, "peft": False}
