"""Command line interface: ``lora-mergecheck check`` and ``lora-mergecheck demo``.

Exit codes: 0 pass, 1 fail (or WARN/INCONCLUSIVE with --strict), 2 input or usage error.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from . import __version__
from .check import LAYERS, check_merge
from .errors import InputError
from .i18n import LANGS, resolve_lang, t
from .report import render_demo, render_report, write_json

EXIT_OK, EXIT_FAIL, EXIT_INPUT = 0, 1, 2


def _scan_lang(argv: list[str]) -> str | None:
    """Find ``--lang X`` / ``--lang=X`` anywhere in argv (needed before the parser is built)."""
    for i, arg in enumerate(argv):
        if arg == "--lang" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--lang="):
            return arg.split("=", 1)[1]
    return None


def _translate_usage_error(message: str, lang: str) -> str:
    """Translate argparse's own error texts (they are English only); unknown texts stay as they are."""
    rules = (
        (r"the following arguments are required: (?P<names>.+)", "argp.required"),
        (r"unrecognized arguments: (?P<names>.+)", "argp.unrecognized"),
        (r"(?P<option>\S+): expected one argument", "argp.expected_one"),
        (r"argument (?P<name>.+?): invalid choice: (?P<value>.+?) \(choose from (?P<choices>.+)\)",
         "argp.invalid_choice"),
        (r"argument (?P<name>.+?): invalid (?P<kind>\w+) value: (?P<value>.+)", "argp.invalid_value"),
    )
    for pattern, key in rules:
        match = re.fullmatch(pattern, message)
        if match:
            return t(key, lang, **match.groupdict())
    return message


class _Parser(argparse.ArgumentParser):
    """ArgumentParser whose usage prefix, help option and error text follow the display language."""

    def __init__(self, *args, lang: str = "en", **kwargs) -> None:
        self._lang = lang

        class Formatter(argparse.HelpFormatter):
            def add_usage(self, usage, actions, groups, prefix=None):
                # argparse itself calls this with prefix="" to build the sub-command's prog name
                return super().add_usage(usage, actions, groups, prefix=t("cli.usage", lang) if prefix is None else prefix)

        kwargs.setdefault("formatter_class", Formatter)
        super().__init__(*args, add_help=False, **kwargs)
        self._positionals.title = t("cli.positionals", lang)
        self._optionals.title = t("cli.optionals", lang)
        self.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS, help=t("cli.help_help", lang))

    def error(self, message: str):  # type: ignore[override]
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: {t('cli.usage_error', self._lang)}: "
                     f"{_translate_usage_error(message, self._lang)}\n")


def build_parser(lang: str) -> argparse.ArgumentParser:
    def lang_option(p: argparse.ArgumentParser, default) -> None:
        p.add_argument("--lang", choices=LANGS, default=default, help=t("cli.lang_help", lang))

    parser = _Parser(prog="lora-mergecheck", description=t("cli.description", lang), lang=lang)
    parser.add_argument("--version", action="version", version=f"lora-mergecheck {__version__}",
                        help=t("cli.version_help", lang))
    lang_option(parser, None)
    sub = parser.add_subparsers(dest="command", required=True, metavar="{check,demo}",
                                parser_class=_Parser)

    check = sub.add_parser("check", help=t("cli.check_help", lang), description=t("cli.check_help", lang),
                         lang=lang)
    check.add_argument("base", metavar="BASE", help=t("cli.base", lang))
    check.add_argument("adapter", metavar="ADAPTER", help=t("cli.adapter", lang))
    check.add_argument("merged", metavar="MERGED", help=t("cli.merged", lang))
    check.add_argument("--layers", choices=LAYERS, default="all", help=t("cli.layers", lang))
    check.add_argument("--prompts", metavar="FILE", help=t("cli.prompts", lang))
    check.add_argument("--num-prompts", type=int, default=8, help=t("cli.num_prompts", lang))
    check.add_argument("--prompt-len", type=int, default=32, help=t("cli.prompt_len", lang))
    check.add_argument("--gen-tokens", type=int, default=16, help=t("cli.gen_tokens", lang))
    check.add_argument("--max-relative-drift", type=float, default=0.10, help=t("cli.max_relative_drift", lang))
    check.add_argument("--warn-lost-ratio", type=float, default=0.25, help=t("cli.warn_lost_ratio", lang))
    check.add_argument("--device", default="cpu", help=t("cli.device", lang))
    check.add_argument("--json", metavar="PATH", help=t("cli.json", lang))
    check.add_argument("--strict", action="store_true", help=t("cli.strict", lang))
    lang_option(check, argparse.SUPPRESS)

    demo = sub.add_parser("demo", help=t("cli.demo_help", lang), description=t("cli.demo_help", lang),
                        lang=lang)
    demo.add_argument("--out", metavar="DIR", help=t("cli.demo_out", lang))
    lang_option(demo, argparse.SUPPRESS)
    return parser


def _print_error(exc: InputError, lang: str) -> None:
    print(f"{t('error.prefix', lang)}: {exc.message(lang)}", file=sys.stderr)
    print(f"{t('hint.prefix', lang)}: {exc.hint(lang)}", file=sys.stderr)


def _refuse_report_path(report_path: Path, inputs: list[str]) -> None:
    """The report must never land on, or inside, a checkpoint input."""
    target = report_path.resolve()
    for raw in inputs:
        src = Path(raw).resolve()
        if target == src or src in target.parents:
            raise InputError("report_inside_input", path=str(report_path), input=raw)


def _check_report_writable(report_path: Path) -> None:
    """Fail before the (possibly long) check when the --json file cannot be written."""
    reason = None
    if report_path.is_dir():
        reason = "it is a folder"
    else:
        ancestor = report_path.resolve().parent
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        if not ancestor.is_dir():
            reason = f"{ancestor} is not a folder"
        elif not os.access(ancestor, os.W_OK | os.X_OK):
            reason = f"no permission to write in {ancestor}"
    if reason:
        raise InputError("report_unwritable", path=str(report_path), reason=reason)


def _run_check(args: argparse.Namespace, lang: str) -> int:
    report_path = Path(args.json) if args.json else None
    try:
        if report_path is not None:
            _refuse_report_path(report_path, [args.base, args.adapter, args.merged])
            _check_report_writable(report_path)

        def progress(message: str) -> None:
            print(f"  {message}", file=sys.stderr)

        report = check_merge(
            args.base, args.adapter, args.merged, layers=args.layers, prompts=args.prompts,
            num_prompts=args.num_prompts, prompt_len=args.prompt_len, gen_tokens=args.gen_tokens,
            max_relative_drift=args.max_relative_drift, warn_lost_ratio=args.warn_lost_ratio,
            device=args.device, progress=progress)
    except InputError as exc:
        _print_error(exc, lang)
        return EXIT_INPUT
    print(render_report(report, lang))
    if report_path is not None:
        try:
            write_json(report, report_path, lang)
        except OSError as exc:
            _print_error(InputError("report_unwritable", path=str(report_path), reason=str(exc)), lang)
            return EXIT_INPUT
    if report.verdict == "FAIL" or (args.strict and report.verdict != "PASS"):
        return EXIT_FAIL
    return EXIT_OK


def _run_demo(args: argparse.Namespace, lang: str) -> int:
    from .demo import CASES, run_demo  # lazy: building the models needs transformers

    print(t("demo.banner", lang))
    print(t("demo.building", lang), file=sys.stderr)

    def on_case(i: int, n: int, key: str) -> None:
        print(t("demo.checking", lang, i=i, n=n, name=t(f"demo.case.{key}", lang)), file=sys.stderr)

    out = Path(args.out) if args.out else None
    try:
        results, root = run_demo(out, on_case=on_case)
    except InputError as exc:
        _print_error(exc, lang)
        return EXIT_INPUT
    print(render_demo(results, lang))
    bad = sum(1 for r in results if not r.ok)
    print(t("demo.some_mismatch", lang, n=bad, total=len(CASES)) if bad
          else t("demo.all_match", lang, n=len(CASES)))
    if out is not None:
        print(t("demo.kept", lang, path=root, base=root / "base", adapter=root / "adapter",
                merged=root / "merged_dropped_o_proj"))
    print(t("demo.banner", lang))
    return EXIT_FAIL if bad else EXIT_OK


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    lang = resolve_lang(_scan_lang(argv))
    args = build_parser(lang).parse_args(argv)
    lang = resolve_lang(args.lang)
    if args.command == "check":
        return _run_check(args, lang)
    return _run_demo(args, lang)


if __name__ == "__main__":
    sys.exit(main())
