"""Test-session compatibility for workspaces containing stale locked temp folders."""

from __future__ import annotations

import tempfile
from pathlib import Path


def pytest_configure(config):
    # pytest.ini in older releases pinned one repository-local directory. On
    # Windows, an interrupted process can leave that directory inaccessible.
    # Give every session a fresh system-temp base before tmp_path is created.
    config.option.basetemp = str(Path(tempfile.mkdtemp(prefix="lyriccap-pytest-")) / "run")
