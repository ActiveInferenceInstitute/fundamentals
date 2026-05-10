"""Allow ``python -m active_inference.web``."""

from .server import main

if __name__ == "__main__":
    raise SystemExit(main())
