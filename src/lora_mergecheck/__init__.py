"""lora-mergecheck: verify that a merged LoRA checkpoint matches the adapter model.

Public API: :func:`check_merge`, :class:`Report`, :class:`InputError`,
:class:`UnsupportedAdapterError`. Everything else is internal.
"""

from __future__ import annotations

from importlib import metadata

from .check import check_merge
from .errors import InputError, UnsupportedAdapterError
from .verdict import Cause, Report

try:
    __version__ = metadata.version("lora-mergecheck")
except metadata.PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0+unknown"

__all__ = ["check_merge", "Report", "Cause", "InputError", "UnsupportedAdapterError", "__version__"]
