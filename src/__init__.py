"""Compatibility shim for src.* imports.

Allows importing modules from the repository root via `src.*` when the
code is laid out flat instead of under a src/ directory.
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in __path__:
    __path__.append(str(_ROOT))
