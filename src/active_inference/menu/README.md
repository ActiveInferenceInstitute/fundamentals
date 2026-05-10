# `active_inference.menu` — chapter runner TUI

Tiny stdlib-only text menu used by the repo-root [`run.sh`](../../../run.sh)
to invoke chapter orchestrators in `chapters/chapter_<N>/`. No domain logic
lives here — this is a thin user-interface layer on top of subprocess.

For a browser-based alternative, see the sister subpackage
[`active_inference.web`](../web/README.md) (launched with
`./run.sh --web`). Both UIs share the discovery layer in
[`runner.py`](runner.py) so adding a chapter / script wires both at once.

## Module layout

| File         | Role |
|--------------|------|
| `runner.py`  | Chapter / script discovery and headless execution helpers. |
| `tui.py`     | Argparse + interactive loop. Exposes `main()`. |
| `__main__.py`| Lets `python -m active_inference.menu` work. |

## Entry points

```bash
# Interactive (default)
./run.sh

# Or invoke the module directly
uv run python -m active_inference.menu
python -m active_inference.menu

# Non-interactive
python -m active_inference.menu --all                  # every chapter
python -m active_inference.menu --chapter 2            # one chapter
python -m active_inference.menu --script chapters/chapter_03/example_3_5_bayesian_linear_regression.py
python -m active_inference.menu --script example_2_2   # filename fragment OK
python -m active_inference.menu --list                 # print menu and exit

# Flags
--no-animations           # skip animation_*.py scripts
--no-visualizations       # skip visualize_*.py diagnostic scripts
--keep-going              # don't abort on first failure
--no-save                 # don't pass --save (opens GUI windows)
```

## Programmatic surface

```python
from active_inference.menu import (
    discover_chapters,
    discover_scripts,
    run_chapter,
    run_all_chapters,
    run_script,
    ChapterEntry,
    ScriptEntry,
)

for chapter in discover_chapters():
    print(chapter.title, len(chapter.scripts))
    for script in chapter.scripts:
        print(" ", script.relative, script.kind)

# Or run them
run_chapter(2, keep_going=True)
run_all_chapters(include_animations=False)
```

## Behavior

* Discovery walks `chapters/chapter_<NN>/` and treats every top-level `.py`
  file as a runnable orchestrator, except files containing `interactive`
  (slider windows that block until the user closes them).
* Each script runs with `MPLBACKEND=Agg` and `--save`. Figures land in
  `output/figures/chapter_<NN>/`.
* `PYTHONPATH` is set to the repo's `src/` so `import active_inference`
  works without an editable install. (You should still install the package
  via `uv pip install -e .` to pick up the `[project.scripts]` entry point.)
* No third-party deps — only `argparse`, `subprocess`, `pathlib`, `shutil`.

## Design constraint

The TUI must remain stdlib-only. If you find yourself reaching for
`rich`, `prompt_toolkit`, or `click`, push the logic into `runner.py`
instead so other front ends (a Jupyter widget, a web UI) can reuse it.
