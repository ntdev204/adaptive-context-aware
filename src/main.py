from __future__ import annotations

from config import load_config


def main() -> None:
    config = load_config("dev")
    print(f"adaptive-context-aware booting in {config.environment} mode")


if __name__ == "__main__":
    main()
