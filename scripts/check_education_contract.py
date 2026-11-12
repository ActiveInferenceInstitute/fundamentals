#!/usr/bin/env python3
"""Validate this member's pointer and bundles with the vendored Education checker.

The unmodified checker modules come from ActiveInferenceInstitute/courses;
see docs/gate.md for the source revision and the migration's content blockers.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vendor"))

from education_contract.registry import ADOPTED, MemberSpec, check_member  # noqa: E402


def main() -> int:
    """Report this member's adoption state and fail on every contract violation."""
    state = check_member(
        ROOT.parent,
        MemberSpec(ROOT.name, "ActiveInferenceInstitute/fundamentals"),
        canonical_schema=ROOT / "specs" / "education" / "schema.json",
    )
    print(f"fundamentals: {state.state} -- {state.detail}")
    for issue in state.issues:
        print(f"  {issue}")
    return 0 if state.state == ADOPTED else 1


if __name__ == "__main__":
    raise SystemExit(main())
