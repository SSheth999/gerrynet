"""Pytest configuration: ensure repo root is importable.

The ``graph`` package has no ``__init__.py`` (it is a namespace package), so
we explicitly put the project root on ``sys.path`` to keep ``from graph
import features`` working regardless of where pytest is invoked from.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
