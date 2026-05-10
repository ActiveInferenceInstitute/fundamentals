"""Allow ``python -m active_inference.menu``."""

from .tui import main

if __name__ == "__main__":
    raise SystemExit(main())
