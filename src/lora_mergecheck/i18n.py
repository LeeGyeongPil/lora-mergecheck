"""Message tables (English / Korean) and language resolution.

All user-visible text lives here. Both tables must have exactly the same keys
(checked by ``tests/test_i18n.py``).

Language precedence: ``--lang`` > ``LORA_MERGECHECK_LANG`` > ``LC_ALL`` >
``LC_MESSAGES`` > ``LANG``. Anything that is not Korean falls back to English.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

LANGS = ("en", "ko")
ENV_VAR = "LORA_MERGECHECK_LANG"

EN: dict[str, str] = {
    # ---- CLI help -------------------------------------------------------
    "cli.description": "Verify that a merged LoRA checkpoint matches the adapter model "
    "(weight layer + output layer). Exit codes: 0 pass, 1 fail, 2 input error.",
    "cli.lang_help": "Display language: en or ko (default: $LORA_MERGECHECK_LANG, then LC_ALL/LC_MESSAGES/LANG, else en).",
    "cli.version_help": "Show the version and exit.",
    "cli.check_help": "Check a merged checkpoint against its base model and adapter.",
    "cli.demo_help": "Run five practice cases on a random tiny model (offline, no download).",
    "cli.base": "Base model folder (or a single .safetensors file) before merging.",
    "cli.adapter": "LoRA adapter folder (adapter_config.json + adapter_model.safetensors).",
    "cli.merged": "Merged checkpoint folder (or a single .safetensors file).",
    "cli.layers": "Which layers to run: weights (no model load), outputs, or all (default).",
    "cli.prompts": "Text file with one prompt per line for the output layer (default: random token ids).",
    "cli.num_prompts": "Number of random prompts for the output layer (default: 8).",
    "cli.prompt_len": "Tokens per random prompt (default: 32).",
    "cli.gen_tokens": "Tokens for the greedy-generation comparison, 0 to skip (default: 16).",
    "cli.max_relative_drift": "Largest allowed output drift relative to the adapter effect (default: 0.10).",
    "cli.warn_lost_ratio": "Warn when this share of update elements vanished in rounding (default: 0.25).",
    "cli.device": "Device for the output layer (default: cpu).",
    "cli.json": "Write the full report as JSON to this path (must not be inside an input).",
    "cli.strict": "Exit 1 on WARN and INCONCLUSIVE too.",
    "cli.demo_out": "Keep the generated checkpoints in this new/empty folder (default: temporary folder).",
    "cli.help_help": "Show this help message and exit.",
    "cli.usage": "usage: ",
    "cli.positionals": "positional arguments",
    "cli.optionals": "options",
    "cli.usage_error": "usage error",
    "argp.required": "the following arguments are required: {names}",
    "argp.unrecognized": "unrecognized arguments: {names}",
    "argp.expected_one": "{option}: expected one argument",
    "argp.invalid_choice": "argument {name}: invalid choice: {value} (choose from {choices})",
    "argp.invalid_value": "argument {name}: invalid {kind} value: {value}",
    # ---- errors ---------------------------------------------------------
    "error.prefix": "error",
    "hint.prefix": "hint",
    "err.path_not_found": "path not found: {path}",
    "hint.path_not_found": "pass an existing folder or .safetensors file.",
    "err.no_safetensors": "no .safetensors weights found in {path}",
    "hint.no_safetensors": "save the checkpoint with safe_serialization=True (model.safetensors or "
    "model.safetensors.index.json).",
    "err.bad_index": "cannot read the shard index {path}: {reason}",
    "hint.bad_index": "re-save the checkpoint so model.safetensors.index.json is valid JSON with a weight_map.",
    "err.no_adapter_config": "adapter_config.json not found in {path}",
    "hint.no_adapter_config": "pass the folder written by PeftModel.save_pretrained().",
    "err.no_adapter_weights": "adapter_model.safetensors not found in {path}",
    "hint.no_adapter_weights": "save the adapter with safe_serialization=True.",
    "err.bad_adapter_config": "adapter_config.json is invalid: {reason}",
    "hint.bad_adapter_config": "re-save the adapter with PeftModel.save_pretrained().",
    "err.not_lora": "adapter type {peft_type} is not LORA",
    "hint.not_lora": "only LoRA adapters are supported (peft_type == LORA).",
    "err.unsupported_dora": "DoRA adapters (use_dora=true) are not supported yet",
    "hint.unsupported_dora": "check this adapter with a LoRA-only run, or wait for DoRA support.",
    "err.report_inside_input": "--json path {path} is the same as or inside the input {input}",
    "hint.report_inside_input": "write the report outside the checkpoint folders, e.g. ./report.json.",
    "err.no_config": "config.json not found in {path}, which the output layer needs",
    "hint.no_config": "pass a folder with config.json, or run with --layers weights.",
    "err.file_for_outputs": "the output layer needs folders, but {path} is a single file",
    "hint.file_for_outputs": "pass model folders, or run with --layers weights.",
    "err.bad_prompts": "cannot read prompts from {path}: {reason}",
    "hint.bad_prompts": "pass a UTF-8 text file with one prompt per line and a tokenizer in the base folder.",
    "err.output_load": "could not load a model for the output layer: {reason}",
    "hint.output_load": "check that the folders are complete, or run with --layers weights.",
    "err.bad_option": "invalid option value: {reason}",
    "hint.bad_option": "see --help for the allowed values.",
    "err.demo_out_not_empty": "demo output folder {path} exists and is not empty",
    "hint.demo_out_not_empty": "pass a new or empty folder, or omit --out to use a temporary folder.",
    "err.bad_safetensors": "cannot read {path} as safetensors: {reason}",
    "hint.bad_safetensors": "the file is truncated or not safetensors; save it again with safe_serialization=True.",
    "err.report_unwritable": "cannot write the --json report to {path}: {reason}",
    "hint.report_unwritable": "pass a file path (not a folder) inside a folder you can write to.",
    "err.bad_device": "device {device!r} is not usable: {reason}",
    "hint.bad_device": "use cpu, or a device this machine has (for example cuda, cuda:0, mps).",
    # ---- report ---------------------------------------------------------
    "report.inputs": "base: {base}  adapter: {adapter}  merged: {merged}",
    "report.weights_header": "Weights layer: {n} LoRA modules, merged dtype {dtype}",
    "report.weights_skipped": "Weights layer: module comparison skipped ({reason})",
    "report.weights_counts": "  exact {exact}  rounding {rounding}  mismatch {mismatch}  "
    "not_applied {not_applied}  missing {missing}",
    "report.weights_loss": "  lost-update ratio {lost}  update retention {retention}  "
    "max residual {resid}",
    "report.outputs_header": "Outputs layer: {n_prompts} prompts, {n_tokens} tokens, device {device}",
    "report.outputs_skipped": "Outputs layer: skipped ({reason})",
    "report.outputs_effect": "  adapter effect (KL ref||base) {effect}  numeric floor {floor}",
    "report.outputs_drift": "  drift (KL ref||merged) mean {drift} max {drift_max}  relative drift {rel}",
    "report.outputs_agree": "  top-1 agreement {top1}  greedy agreement {greedy}  max logit diff {logit}",
    "report.verdict": "Verdict: {verdict}",
    "report.causes": "Causes:",
    "report.recommendations": "What to do:",
    "report.modules_more": "      ... and {n} more",
    "report.skip.layers": "--layers weights",
    "report.skip.leftover": "merged file still holds LoRA keys",
    "report.skip.quantized": "quantized checkpoint",
    "report.skip.structure": "structure problem found first",
    "report.skip.no_lora": "adapter has no LoRA modules",
    "report.skip.error": "{reason}",
    # ---- demo -----------------------------------------------------------
    "demo.banner": "practice mock - random tiny model, offline; numbers are not real-model quality",
    "demo.building": "Building the random base model and five merged checkpoints ...",
    "demo.checking": "Checking case {i}/{n}: {name} ...",
    "demo.col.case": "case",
    "demo.col.expected": "expected",
    "demo.col.verdict": "verdict",
    "demo.col.match": "match",
    "demo.all_match": "All {n} cases matched the expected verdicts.",
    "demo.some_mismatch": "{n} of {total} cases did not match the expected verdicts.",
    "demo.kept": "Checkpoints kept in {path}; try: lora-mergecheck check {base} {adapter} {merged}",
    "demo.case.fp32_normal": "1 fp32 merge, correct",
    "demo.case.dropped_o_proj": "2 o_proj update dropped",
    "demo.case.bf16_small_update": "3 bf16 merge, small update",
    "demo.case.leftover_lora": "4 merge result not used",
    "demo.case.mts_not_merged": "5 lm_head not merged",
    "demo.expect.PASS": "PASS",
    "demo.expect.update_not_applied": "FAIL update_not_applied",
    "demo.expect.low_precision_update_loss": "WARN+ low_precision_update_loss",
    "demo.expect.leftover_lora_keys": "FAIL leftover_lora_keys",
    "demo.expect.modules_to_save_not_merged": "FAIL modules_to_save_not_merged",
    # ---- verdicts -------------------------------------------------------
    "verdict.PASS": "PASS",
    "verdict.WARN": "WARN",
    "verdict.FAIL": "FAIL",
    "verdict.INCONCLUSIVE": "INCONCLUSIVE",
    # ---- cause meanings -------------------------------------------------
    "cause.update_not_applied": "The LoRA update was not added to these modules (the merged weight equals the base).",
    "cause.update_mismatch": "The merged weight differs from base + s*B*A by more than dtype rounding allows.",
    "cause.leftover_lora_keys": "The merged file still contains lora_/base_layer keys: it is not a merged checkpoint.",
    "cause.non_lora_changed": "Layers without a LoRA adapter changed compared with the base model.",
    "cause.modules_to_save_not_merged": "Layers the adapter saved in full (modules_to_save) do not hold the adapter values.",
    "cause.vocab_resized_without_saved_embeddings": "The vocabulary grew but the adapter did not save the embedding layers.",
    "cause.shape_mismatch": "Tensor shapes differ between the base and merged checkpoints.",
    "cause.missing_tensor": "Tensors present in the base are missing from the merged checkpoint.",
    "cause.quantized_checkpoint": "The checkpoint is quantized; weight comparison was skipped.",
    "cause.low_precision_update_loss": "A large share of the LoRA update vanished in low-precision rounding.",
    "cause.unsupported_layer": "Adapter layers of an unsupported type were left out of the verdict.",
    "cause.adapter_effect_too_small": "On these prompts the adapter effect is at numeric-noise level; the output check cannot judge.",
    "cause.output_drift": "Merged outputs drift from the adapter model by more than allowed relative to the adapter effect.",
    "cause.outputs_unavailable": "The output layer could not run.",
    # ---- recommendations ------------------------------------------------
    "rec.update_not_applied": "Re-run merge_and_unload() on all target modules and save the returned model.",
    "rec.update_mismatch": "Confirm base, adapter and merged belong together (same base, same lora_alpha/r).",
    "rec.leftover_lora_keys": "Save the model RETURNED by merge_and_unload(), not the PEFT wrapper.",
    "rec.non_lora_changed": "Check that the merged model started from the same base weights.",
    "rec.modules_to_save_not_merged": "Merge with the adapter's modules_to_save layers included, then save.",
    "rec.vocab_resized_without_saved_embeddings": "Resize the base embeddings first and save embed_tokens/lm_head via modules_to_save.",
    "rec.shape_mismatch": "Make sure the merged checkpoint was built from this base model.",
    "rec.missing_tensor": "Save the complete model (all shards) after merging.",
    "rec.quantized_checkpoint": "Merge into a non-quantized base first, then quantize; or serve the adapter separately.",
    "rec.low_precision_update_loss": "Save the merged model in fp32, serve the adapter separately, or confirm the output layer.",
    "rec.unsupported_layer": "Verify those layers by another route; this tool supports plain LoRA Linear layers only.",
    "rec.adapter_effect_too_small": "Re-run with real prompts (--prompts FILE) where the adapter changes the answer.",
    "rec.output_drift": "Compare per-module results; merge in higher precision or serve the adapter separately.",
    "rec.outputs_unavailable": "Run with --layers weights, or check that both folders load with transformers.",
}

KO: dict[str, str] = {
    "cli.description": "합친 LoRA 체크포인트가 어댑터 모델과 같은지 확인합니다 "
    "(가중치 층 + 출력 층). 종료 코드: 0 통과, 1 실패, 2 입력 오류.",
    "cli.lang_help": "화면 언어 en 또는 ko (기본: $LORA_MERGECHECK_LANG, 다음 LC_ALL/LC_MESSAGES/LANG, 그 외 en).",
    "cli.version_help": "버전을 보여 주고 끝냅니다.",
    "cli.check_help": "합친 체크포인트를 기본 모델·어댑터와 대조합니다.",
    "cli.demo_help": "무작위 소형 모델로 연습 사례 5개를 돌립니다(오프라인, 다운로드 없음).",
    "cli.base": "합치기 전 기본 모델 폴더(또는 .safetensors 파일 하나).",
    "cli.adapter": "LoRA 어댑터 폴더(adapter_config.json + adapter_model.safetensors).",
    "cli.merged": "합친 체크포인트 폴더(또는 .safetensors 파일 하나).",
    "cli.layers": "실행할 층: weights(모델 로드 없음), outputs, all(기본).",
    "cli.prompts": "출력 층에 쓸 프롬프트 텍스트 파일(한 줄에 하나, 기본: 무작위 토큰).",
    "cli.num_prompts": "출력 층의 무작위 프롬프트 수(기본: 8).",
    "cli.prompt_len": "무작위 프롬프트당 토큰 수(기본: 32).",
    "cli.gen_tokens": "탐욕 생성 비교에 쓸 토큰 수, 0 이면 생략(기본: 16).",
    "cli.max_relative_drift": "어댑터 효과 대비 허용하는 출력 드리프트 상한(기본: 0.10).",
    "cli.warn_lost_ratio": "갱신 원소가 반올림으로 사라진 비율이 이 값 이상이면 경고(기본: 0.25).",
    "cli.device": "출력 층을 돌릴 장치(기본: cpu).",
    "cli.json": "전체 보고서를 이 경로에 JSON 으로 저장(입력 경로 안이면 안 됨).",
    "cli.strict": "WARN·INCONCLUSIVE 도 종료 코드 1 로 처리합니다.",
    "cli.demo_out": "만든 체크포인트를 이 새(빈) 폴더에 남깁니다(기본: 임시 폴더).",
    "cli.help_help": "이 도움말을 보여 주고 끝냅니다.",
    "cli.usage": "사용법: ",
    "cli.positionals": "위치 인자",
    "cli.optionals": "옵션",
    "cli.usage_error": "사용법 오류",
    "argp.required": "다음 인자가 필요합니다: {names}",
    "argp.unrecognized": "알 수 없는 인자: {names}",
    "argp.expected_one": "{option}: 값이 하나 필요합니다",
    "argp.invalid_choice": "인자 {name}: 허용되지 않는 값: {value} (선택지: {choices})",
    "argp.invalid_value": "인자 {name}: {kind} 값이 올바르지 않습니다: {value}",
    "error.prefix": "오류",
    "hint.prefix": "힌트",
    "err.path_not_found": "경로가 없습니다: {path}",
    "hint.path_not_found": "있는 폴더나 .safetensors 파일을 넘기세요.",
    "err.no_safetensors": "{path} 에 .safetensors 가중치가 없습니다",
    "hint.no_safetensors": "safe_serialization=True 로 저장하세요(model.safetensors 또는 "
    "model.safetensors.index.json).",
    "err.bad_index": "샤드 색인 {path} 를 읽을 수 없습니다: {reason}",
    "hint.bad_index": "model.safetensors.index.json 이 weight_map 이 있는 올바른 JSON 이 되게 다시 저장하세요.",
    "err.no_adapter_config": "{path} 에 adapter_config.json 이 없습니다",
    "hint.no_adapter_config": "PeftModel.save_pretrained() 가 만든 폴더를 넘기세요.",
    "err.no_adapter_weights": "{path} 에 adapter_model.safetensors 가 없습니다",
    "hint.no_adapter_weights": "safe_serialization=True 로 어댑터를 저장하세요.",
    "err.bad_adapter_config": "adapter_config.json 이 올바르지 않습니다: {reason}",
    "hint.bad_adapter_config": "PeftModel.save_pretrained() 로 어댑터를 다시 저장하세요.",
    "err.not_lora": "어댑터 종류 {peft_type} 는 LORA 가 아닙니다",
    "hint.not_lora": "LoRA 어댑터(peft_type == LORA)만 지원합니다.",
    "err.unsupported_dora": "DoRA 어댑터(use_dora=true)는 아직 지원하지 않습니다",
    "hint.unsupported_dora": "LoRA 전용 어댑터로 검사하거나 DoRA 지원을 기다리세요.",
    "err.report_inside_input": "--json 경로 {path} 가 입력 {input} 와 같거나 그 안에 있습니다",
    "hint.report_inside_input": "보고서는 체크포인트 폴더 밖에 쓰세요(예: ./report.json).",
    "err.no_config": "출력 층에 필요한 config.json 이 {path} 에 없습니다",
    "hint.no_config": "config.json 이 있는 폴더를 넘기거나 --layers weights 로 실행하세요.",
    "err.file_for_outputs": "출력 층은 폴더가 필요한데 {path} 는 파일 하나입니다",
    "hint.file_for_outputs": "모델 폴더를 넘기거나 --layers weights 로 실행하세요.",
    "err.bad_prompts": "{path} 에서 프롬프트를 읽을 수 없습니다: {reason}",
    "hint.bad_prompts": "한 줄에 프롬프트 하나인 UTF-8 텍스트 파일을 넘기고, 기본 모델 폴더에 토크나이저를 두세요.",
    "err.output_load": "출력 층에 쓸 모델을 불러오지 못했습니다: {reason}",
    "hint.output_load": "폴더가 완전한지 확인하거나 --layers weights 로 실행하세요.",
    "err.bad_option": "옵션 값이 올바르지 않습니다: {reason}",
    "hint.bad_option": "허용 값은 --help 를 보세요.",
    "err.demo_out_not_empty": "데모 출력 폴더 {path} 가 있고 비어 있지 않습니다",
    "hint.demo_out_not_empty": "새 폴더나 빈 폴더를 넘기거나 --out 을 빼서 임시 폴더를 쓰세요.",
    "err.bad_safetensors": "{path} 를 safetensors 로 읽을 수 없습니다: {reason}",
    "hint.bad_safetensors": "파일이 잘렸거나 safetensors 가 아닙니다. safe_serialization=True 로 다시 저장하세요.",
    "err.report_unwritable": "--json 보고서를 {path} 에 쓸 수 없습니다: {reason}",
    "hint.report_unwritable": "쓸 수 있는 폴더 안의 파일 경로를 넘기세요(폴더 자체는 안 됩니다).",
    "err.bad_device": "장치 {device!r} 를 쓸 수 없습니다: {reason}",
    "hint.bad_device": "cpu 를 쓰거나 이 컴퓨터에 있는 장치(예: cuda, cuda:0, mps)를 넘기세요.",
    "report.inputs": "기본: {base}  어댑터: {adapter}  합친 것: {merged}",
    "report.weights_header": "가중치 층: LoRA 모듈 {n}개, 합친 dtype {dtype}",
    "report.weights_skipped": "가중치 층: 모듈 비교 생략({reason})",
    "report.weights_counts": "  exact {exact}  rounding {rounding}  mismatch {mismatch}  "
    "not_applied {not_applied}  missing {missing}",
    "report.weights_loss": "  갱신 소실 비율 {lost}  갱신 보존율 {retention}  최대 잔차 {resid}",
    "report.outputs_header": "출력 층: 프롬프트 {n_prompts}개, 토큰 {n_tokens}개, 장치 {device}",
    "report.outputs_skipped": "출력 층: 생략({reason})",
    "report.outputs_effect": "  어댑터 효과(KL 기준||기본) {effect}  수치 바닥 {floor}",
    "report.outputs_drift": "  드리프트(KL 기준||합친 것) 평균 {drift} 최대 {drift_max}  상대 드리프트 {rel}",
    "report.outputs_agree": "  top-1 일치율 {top1}  탐욕 생성 일치율 {greedy}  최대 로짓 차 {logit}",
    "report.verdict": "판정: {verdict}",
    "report.causes": "원인:",
    "report.recommendations": "권장 조치:",
    "report.modules_more": "      ... 외 {n}개",
    "report.skip.layers": "--layers weights",
    "report.skip.leftover": "합친 파일에 LoRA 키가 남아 있음",
    "report.skip.quantized": "양자화 체크포인트",
    "report.skip.structure": "구조 문제가 먼저 발견됨",
    "report.skip.no_lora": "어댑터에 LoRA 모듈이 없음",
    "report.skip.error": "{reason}",
    "demo.banner": "연습용 목(mock, practice mock) - 무작위 소형 모델, 오프라인; 숫자는 실제 모델 품질이 아님",
    "demo.building": "무작위 기본 모델과 합친 체크포인트 5개를 만드는 중 ...",
    "demo.checking": "사례 {i}/{n} 검사 중: {name} ...",
    "demo.col.case": "사례",
    "demo.col.expected": "기대",
    "demo.col.verdict": "판정",
    "demo.col.match": "일치",
    "demo.all_match": "사례 {n}개가 모두 기대한 판정과 같았습니다.",
    "demo.some_mismatch": "사례 {total}개 중 {n}개가 기대한 판정과 달랐습니다.",
    "demo.kept": "체크포인트를 {path} 에 남겼습니다. 직접 해 보세요: lora-mergecheck check {base} {adapter} {merged}",
    "demo.case.fp32_normal": "1 fp32 로 정상 합침",
    "demo.case.dropped_o_proj": "2 o_proj 갱신 누락",
    "demo.case.bf16_small_update": "3 bf16 합침, 작은 갱신",
    "demo.case.leftover_lora": "4 합친 결과를 안 받음",
    "demo.case.mts_not_merged": "5 lm_head 미합침",
    "demo.expect.PASS": "PASS",
    "demo.expect.update_not_applied": "FAIL update_not_applied",
    "demo.expect.low_precision_update_loss": "WARN+ low_precision_update_loss",
    "demo.expect.leftover_lora_keys": "FAIL leftover_lora_keys",
    "demo.expect.modules_to_save_not_merged": "FAIL modules_to_save_not_merged",
    "verdict.PASS": "통과(PASS)",
    "verdict.WARN": "경고(WARN)",
    "verdict.FAIL": "실패(FAIL)",
    "verdict.INCONCLUSIVE": "판단 불가(INCONCLUSIVE)",
    "cause.update_not_applied": "이 모듈들에 LoRA 갱신이 더해지지 않았습니다(합친 가중치가 기본과 같음).",
    "cause.update_mismatch": "합친 가중치가 기본 + s*B*A 와 dtype 반올림 범위를 넘어 다릅니다.",
    "cause.leftover_lora_keys": "합친 파일에 lora_/base_layer 키가 남아 있습니다: 합친 체크포인트가 아닙니다.",
    "cause.non_lora_changed": "LoRA 가 없는 층이 기본 모델과 달라졌습니다.",
    "cause.modules_to_save_not_merged": "어댑터가 통째로 저장한 층(modules_to_save)에 어댑터 값이 들어 있지 않습니다.",
    "cause.vocab_resized_without_saved_embeddings": "어휘가 늘었는데 어댑터가 임베딩 층을 저장하지 않았습니다.",
    "cause.shape_mismatch": "기본과 합친 체크포인트의 텐서 모양이 다릅니다.",
    "cause.missing_tensor": "기본에 있는 텐서가 합친 체크포인트에 없습니다.",
    "cause.quantized_checkpoint": "양자화된 체크포인트라 가중치 비교를 건너뛰었습니다.",
    "cause.low_precision_update_loss": "LoRA 갱신의 상당 부분이 저정밀 반올림으로 사라졌습니다.",
    "cause.unsupported_layer": "지원하지 않는 종류의 어댑터 층은 판정에서 뺐습니다.",
    "cause.adapter_effect_too_small": "이 프롬프트에서는 어댑터 효과가 수치 잡음 수준이라 출력 층이 판단할 수 없습니다.",
    "cause.output_drift": "합친 모델의 출력이 어댑터 효과 대비 허용치보다 어댑터 모델에서 벗어납니다.",
    "cause.outputs_unavailable": "출력 층을 실행하지 못했습니다.",
    "rec.update_not_applied": "모든 대상 모듈에 merge_and_unload() 를 다시 돌리고 반환된 모델을 저장하세요.",
    "rec.update_mismatch": "기본·어댑터·합친 것이 서로 맞는지(같은 기본, 같은 lora_alpha/r) 확인하세요.",
    "rec.leftover_lora_keys": "PEFT 래퍼가 아니라 merge_and_unload() 가 반환한 모델을 저장하세요.",
    "rec.non_lora_changed": "합친 모델이 같은 기본 가중치에서 시작했는지 확인하세요.",
    "rec.modules_to_save_not_merged": "어댑터의 modules_to_save 층까지 포함해 합친 뒤 저장하세요.",
    "rec.vocab_resized_without_saved_embeddings": "기본 임베딩을 먼저 늘리고 embed_tokens/lm_head 를 modules_to_save 로 저장하세요.",
    "rec.shape_mismatch": "합친 체크포인트가 이 기본 모델에서 만들어졌는지 확인하세요.",
    "rec.missing_tensor": "합친 뒤 모델 전체(모든 샤드)를 저장하세요.",
    "rec.quantized_checkpoint": "양자화하지 않은 기본에 먼저 합친 뒤 양자화하거나, 어댑터를 분리해 서빙하세요.",
    "rec.low_precision_update_loss": "fp32 로 저장하거나 어댑터를 분리해 서빙하고, 출력 층 결과를 확인하세요.",
    "rec.unsupported_layer": "그 층은 다른 방법으로 확인하세요. 이 도구는 일반 LoRA Linear 층만 지원합니다.",
    "rec.adapter_effect_too_small": "어댑터가 답을 바꾸는 실제 프롬프트(--prompts FILE)로 다시 돌리세요.",
    "rec.output_drift": "모듈별 결과를 비교하고, 더 높은 정밀도로 합치거나 어댑터를 분리해 서빙하세요.",
    "rec.outputs_unavailable": "--layers weights 로 실행하거나 두 폴더가 transformers 로 불러와지는지 확인하세요.",
}

TABLES: dict[str, dict[str, str]] = {"en": EN, "ko": KO}


def normalize(value: str | None) -> str | None:
    """Map a language tag such as ``ko_KR.UTF-8`` to ``ko``/``en``; unknown -> None."""
    if not value:
        return None
    head = value.strip().lower().replace("-", "_")
    if head.startswith("ko"):
        return "ko"
    if head.startswith("en"):
        return "en"
    return None


def resolve_lang(cli_value: str | None = None, env: Mapping[str, str] | None = None) -> str:
    """Pick the display language: --lang > LORA_MERGECHECK_LANG > LC_ALL > LC_MESSAGES > LANG."""
    env = os.environ if env is None else env
    if cli_value in LANGS:
        return str(cli_value)
    tool = normalize(env.get(ENV_VAR))
    if tool:
        return tool
    for name in ("LC_ALL", "LC_MESSAGES", "LANG"):
        raw = env.get(name)
        if raw:  # POSIX: the first variable that is set decides
            return normalize(raw) or "en"
    return "en"


def t(key: str, lang: str = "en", **params: object) -> str:
    """Look up ``key`` in the table for ``lang`` and fill ``{placeholders}``."""
    table = TABLES.get(lang, EN)
    text = table.get(key, EN[key])
    return text.format(**params) if params else text
