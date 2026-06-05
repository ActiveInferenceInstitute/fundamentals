# `active_inference.menu` — chapter and extras runner TUI

Tiny stdlib-only text menu used by the repo-root [`run.sh`](../../../run.sh)
to invoke chapter orchestrators in `chapters/chapter_<N>/` and topic
orchestrators in `extras/<topic>/`. No domain logic lives here — this is a
thin user-interface layer on top of subprocess.

For a browser-based alternative, see the sister subpackage
[`active_inference.web`](../web/README.md) (launched with
`./run.sh --web`). Both UIs share the discovery layer in
[`runner.py`](runner.py) so adding a chapter, extras topic, or script wires
both at once.

## Module layout

| File         | Role |
|--------------|------|
| `runner.py`  | Chapter/extras discovery and headless execution helpers. |
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
python -m active_inference.menu --extras               # every extras topic
python -m active_inference.menu --extra entropy        # one extras topic
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
    discover_extras,
    discover_scripts,
    discover_extra_scripts,
    run_chapter,
    run_extra_topic,
    run_all_chapters,
    run_all_extras,
    run_script,
    ChapterEntry,
    ExtraTopicEntry,
    ScriptEntry,
)

for chapter in discover_chapters():
    print(chapter.title, len(chapter.scripts))
    for script in chapter.scripts:
        print(" ", script.relative, script.kind)

# Or run them
run_chapter(2, keep_going=True)
run_all_chapters(include_animations=False)
run_extra_topic("entropy")
run_all_extras()
```

## Behavior

* Discovery walks `chapters/chapter_<NN>/` and `extras/<topic>/`, treating
  every top-level `.py` file as a runnable orchestrator except files
  containing `interactive` (slider windows that block until the user closes
  them). Pass `include_interactive=True` to `discover_scripts` or
  `discover_extra_scripts` when a UI needs launchable slider rows.
* Each script runs with `MPLBACKEND=Agg` and `--save`. Figures land in
  `output/figures/chapter_<NN>/` or `output/figures/extras/<topic>/`.
* `PYTHONPATH` is set to the repo's `src/` so `import active_inference`
  works without an editable install. (You should still install the package
  via `uv pip install -e .` to pick up the `[project.scripts]` entry point.)
* No third-party deps — only `argparse`, `subprocess`, `pathlib`, `shutil`.

## Design constraint

The TUI must remain stdlib-only. If you find yourself reaching for
`rich`, `prompt_toolkit`, or `click`, push the logic into `runner.py`
instead so other front ends (a Jupyter widget, a web UI) can reuse it.
