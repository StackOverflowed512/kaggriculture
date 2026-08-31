"""build_submission.py -- package the agent into ``submission.tar.gz``.

Release step 2 of the design doc's workflow: build the submission artifact,
*refusing to ship if the embedded rules disagree with the source*. In this
architecture the agent has no second, hand-copied rule table baked into
``main.py`` -- it loads ``rules_validated.json`` through :class:`RulesLoader`.
So the "embedded vs source" check here verifies the exact bytes we are about to
bundle round-trip, through the real loader, to the same validated object the
running agent will use. If a future change ever inlines a stale copy of the
rules, ``embedded_rules()`` is the single hook to point at it and the guard
keeps a mismatched pair from ever being packaged.

Usage:
    python build_submission.py                 # verify, then build submission.tar.gz
    python build_submission.py --check-only     # run the checks, build nothing
    python build_submission.py -o out/agent.tar.gz

Exit code 0 on success, non-zero if a required file is missing or the rules
fail to validate / disagree with the source (the tarball is not written).
"""

import argparse
import json
import os
import sys
import tarfile

from rules_loader import RulesLoader

# Runtime modules that make up the agent. Test suites, docs and plans are
# deliberately excluded -- only what the engine needs to run the season ships.
RUNTIME_MODULES = (
    "main.py",
    "scheduler.py",
    "rules_loader.py",
    "action_emitter.py",
    "telemetry.py",
    "market_model.py",
    "care_monitor.py",
    "forecaster.py",
)
RULES_FILE = "rules_validated.json"
DEFAULT_OUTPUT = "submission.tar.gz"


def submission_members(root="."):
    """Return the ordered list of files to bundle (modules + rules)."""
    return [os.path.join(root, name) for name in RUNTIME_MODULES + (RULES_FILE,)]


def missing_files(root="."):
    """Return any expected submission member that is absent on disk."""
    return [path for path in submission_members(root) if not os.path.isfile(path)]


def source_rules(root="."):
    """The raw ``rules_validated.json`` parse -- the source of truth on disk."""
    with open(os.path.join(root, RULES_FILE), "r", encoding="utf-8") as handle:
        return json.load(handle)


def embedded_rules(root="."):
    """The rules the *packaged agent* will actually load and run on.

    Today that is ``RulesLoader`` reading the same file we bundle, so this both
    exercises the mandatory-key validation and yields the object main.py uses.
    Kept as its own function so that if the wrapper ever inlines a rule copy,
    only this hook changes and the source-match guard below still bites.
    """
    return RulesLoader(os.path.join(root, RULES_FILE)).load_rules()


def verify_rules(root="."):
    """Validate the rules and confirm embedded == source.

    Returns ``(ok, messages)``. ``ok`` is False when the loader rejects the
    file (missing mandatory keys, bad JSON) or the packaged/embedded rules
    differ from the on-disk source -- either way the build must not proceed.
    """
    messages = []
    try:
        source = source_rules(root)
    except (OSError, ValueError) as exc:
        return False, [f"cannot read {RULES_FILE}: {exc}"]

    try:
        embedded = embedded_rules(root)
    except (OSError, ValueError) as exc:
        # RulesLoader raises ValueError on bad JSON or missing mandatory keys.
        return False, [f"rules failed validation: {exc}"]

    if embedded != source:
        return False, ["embedded rules do not match source rules_validated.json"]

    messages.append(f"rules validated ({len(source.get('constants', {}))} constants, "
                    f"embedded copy matches source)")
    return True, messages


def build_archive(output=DEFAULT_OUTPUT, root="."):
    """Write the gzip tarball with every member at the archive root."""
    with tarfile.open(output, "w:gz") as tar:
        for path in submission_members(root):
            tar.add(path, arcname=os.path.basename(path))
    return output


def build_submission(output=DEFAULT_OUTPUT, root=".", check_only=False):
    """Run all release checks, then (unless check_only) write the tarball.

    Returns ``(ok, messages)``; on failure the archive is never created.
    """
    absent = missing_files(root)
    if absent:
        return False, ["missing required files: " + ", ".join(os.path.basename(p) for p in absent)]

    ok, messages = verify_rules(root)
    if not ok:
        return False, messages

    if check_only:
        messages.append("check-only: all checks passed, archive not written")
        return True, messages

    archive = build_archive(output, root)
    size = os.path.getsize(archive)
    messages.append(f"wrote {archive} ({size} bytes, "
                    f"{len(RUNTIME_MODULES) + 1} files)")
    return True, messages


def main(argv=None):
    parser = argparse.ArgumentParser(description="Package the Kaggriculture agent for submission.")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT,
                        help=f"output tarball path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--root", default=".", help="project root to package from")
    parser.add_argument("--check-only", action="store_true",
                        help="run verification only; do not write the archive")
    args = parser.parse_args(argv)

    ok, messages = build_submission(args.output, args.root, args.check_only)
    stream = sys.stdout if ok else sys.stderr
    for message in messages:
        print(("OK: " if ok else "FAIL: ") + message, file=stream)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
