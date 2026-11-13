"""Public exceptions. ``InputError`` maps to CLI exit code 2."""

from __future__ import annotations

from .i18n import t


class InputError(Exception):
    """The inputs or options are unusable (missing path, bad adapter, ...).

    ``code`` selects the translated message (``err.<code>`` / ``hint.<code>``);
    ``str(exc)`` is always the English text with the hint appended.
    """

    def __init__(self, code: str, **params: object) -> None:
        self.code = code
        self.params = params
        super().__init__(self.message("en"))

    def message(self, lang: str) -> str:
        return t(f"err.{self.code}", lang, **self.params)

    def hint(self, lang: str) -> str:
        return t(f"hint.{self.code}", lang, **self.params)

    def __str__(self) -> str:
        return f"{self.message('en')} ({self.hint('en')})"


class UnsupportedAdapterError(InputError):
    """The adapter uses a feature this version cannot verify (for example DoRA)."""
