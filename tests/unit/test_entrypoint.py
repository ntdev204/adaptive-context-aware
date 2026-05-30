from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_entrypoint(tmp_path: Path, *, require_engine: bool) -> subprocess.CompletedProcess[str]:
    if os.name == "nt":
        pytest.skip("entrypoint.sh is exercised in the Linux container runtime")
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is required to exercise entrypoint.sh")

    env = os.environ.copy()
    env.update(
        {
            "CTX_RUNTIME_BACKEND": "engine",
            "CTX_ENGINE_CACHE_DIR": str(tmp_path / "engines"),
            "CTX_ENGINE_MODEL_PATH": str(tmp_path / "missing.engine"),
            "CTX_PT_MODEL_PATH": str(tmp_path / "best.pt"),
            "CTX_REQUIRE_ENGINE_AT_BOOT": "true" if require_engine else "false",
        }
    )
    return subprocess.run(
        [bash, "scripts/entrypoint.sh", sys.executable, "-c", "print('boot-ok')"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_entrypoint_allows_degraded_boot_when_engine_is_missing(tmp_path: Path) -> None:
    result = _run_entrypoint(tmp_path, require_engine=False)

    assert result.returncode == 0
    assert "boot-ok" in result.stdout
    assert "WARN: missing TensorRT engine" in result.stderr
    assert "Runtime will start degraded" in result.stderr


def test_entrypoint_can_require_engine_at_boot(tmp_path: Path) -> None:
    result = _run_entrypoint(tmp_path, require_engine=True)

    assert result.returncode == 1
    assert "boot-ok" not in result.stdout
    assert "refusing to start" in result.stderr
