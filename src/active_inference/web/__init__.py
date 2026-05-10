"""Local web UI for the chapter orchestrators.

Spins up a small ``http.server``-backed site at ``http://127.0.0.1:8765/``
with one tab per chapter. Each tab shows the rendered figures and
animations from ``output/figures/chapter_<N>/``, a list of every script
(with a "Render" button that POSTs back to the server to regenerate the
figure), and links to the corresponding documentation pages.

The web module mirrors :mod:`active_inference.menu`: stdlib only, no
third-party dependencies. The same discovery layer
(``active_inference.menu.runner``) is reused so the two UIs always stay
in sync.

Entry points::

    ./run.sh --web
    uv run python -m active_inference.web
    uv run active-inference-web                    # PEP 621 console script
"""

from .server import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    launch,
    main,
    run_server,
)

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "launch",
    "main",
    "run_server",
]
