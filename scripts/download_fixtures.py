from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    root = ROOT
    for name in ("coco_person_100", "mot17_subset"):
        path = root / "tests" / "fixtures" / name
        path.mkdir(parents=True, exist_ok=True)
        marker = path / "README.txt"
        if not marker.exists():
            marker.write_text(
                "Placeholder directory. Download integration is not implemented in this baseline.\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
