# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.1] - 2026-11-14

### Fixed
- A merged folder that still contains `adapter_config.json` now fails with the new cause
  `leftover_adapter_config` (all layer modes, exit 1 without `--strict`). Before, `--layers weights`
  passed and the default run only warned, although transformers cannot load such a folder.

## [0.1.0] - 2026-11-14

### Added
- `lora-mergecheck check BASE ADAPTER MERGED`: weight layer (per-module `s*B*A` residuals,
  dtype ulp limits, lost-update ratio, leftover LoRA keys, `modules_to_save`, vocabulary and
  shape checks, quantization detection, sharded safetensors) and output layer (drift relative to
  the adapter effect, numeric floor, top-1 and greedy agreement).
- Verdicts `PASS`, `WARN`, `FAIL`, `INCONCLUSIVE` with cause codes, recommended actions, a JSON
  report (`--json`), exit codes 0/1/2 and `--strict`.
- `lora-mergecheck demo`: five practice cases on a random tiny Llama (offline).
- English and Korean output (`--lang`, `LORA_MERGECHECK_LANG`, locale variables).
- Python API `lora_mergecheck.check_merge`.
