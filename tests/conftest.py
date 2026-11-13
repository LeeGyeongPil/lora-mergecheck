"""Shared fixtures: a random tiny Llama, its adapters and merged checkpoints (CPU, offline)."""

from __future__ import annotations

import json
import os
import shutil
import socket
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Never talk to the Hub, even if a code path forgets local_files_only.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from lora_mergecheck import demo  # noqa: E402


@pytest.fixture(autouse=True)
def english_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must not depend on the developer's locale."""
    for var in ("LORA_MERGECHECK_LANG", "LC_ALL", "LC_MESSAGES", "LANG"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if a test tries to open a network connection."""

    def refuse(*_args, **_kwargs):
        raise AssertionError("network access attempted in a test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture(scope="session")
def cases(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The five demo cases plus a correct bf16 merge and a correct modules_to_save merge."""
    root = tmp_path_factory.mktemp("cases")
    demo.build_cases(root)
    base_dir = root / "base"
    bf16 = demo._load_peft(base_dir, root / "adapter", torch.bfloat16).merge_and_unload()
    demo._save_state(bf16.state_dict(), root / "merged_bf16_normal", base_dir)
    mts = demo._load_peft(base_dir, root / "adapter_mts", torch.float32).merge_and_unload()
    demo._save_state(mts.state_dict(), root / "merged_mts_ok", base_dir)
    return root


@pytest.fixture()
def work(tmp_path: Path) -> Path:
    return tmp_path


def copy_dir(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


def edit_json(path: Path, **changes) -> None:
    data = json.loads(path.read_text())
    data.update(changes)
    path.write_text(json.dumps(data))


def read_state(folder: Path) -> dict[str, torch.Tensor]:
    out = {}
    with safe_open(str(folder / "model.safetensors"), "pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def write_state(state: dict[str, torch.Tensor], folder: Path, config_from: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in state.items()}, str(folder / "model.safetensors"))
    shutil.copy(config_from / "config.json", folder / "config.json")
    return folder
