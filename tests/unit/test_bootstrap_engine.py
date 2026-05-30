from __future__ import annotations

from pathlib import Path

from scripts.bootstrap_engine import ensure_engine


def test_existing_bootstrap_engine_syncs_to_codebase_path(tmp_path: Path) -> None:
    engine_dir = tmp_path / "container" / "engines"
    engine_path = engine_dir / "best.engine"
    meta_path = engine_dir / "best.json"
    engine_dir.mkdir(parents=True)
    engine_path.write_bytes(b"fake-engine")
    meta_path.write_text("{}", encoding="utf-8")

    codebase_engine_path = tmp_path / "codebase" / "models" / "engines" / "best.engine"
    result = ensure_engine(
        engine_dir,
        model_path=tmp_path / "missing.pt",
        engine_output=engine_path,
        codebase_engine_output=codebase_engine_path,
    )

    assert result == engine_path
    assert codebase_engine_path.read_bytes() == b"fake-engine"
    assert codebase_engine_path.with_suffix(".json").read_text(encoding="utf-8") == "{}"
