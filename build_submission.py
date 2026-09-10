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
import ast
import importlib.util
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

# Dependency-ordered concatenation for the single-file bundle: every module
# appears after each local module it imports, so names are defined before use.
# (rules_loader/action_emitter/telemetry/market_model/care_monitor have no local
# deps; forecaster and scheduler import market_model; main imports them all.)
SINGLE_FILE_ORDER = (
    "rules_loader.py",
    "action_emitter.py",
    "telemetry.py",
    "market_model.py",
    "care_monitor.py",
    "forecaster.py",
    "scheduler.py",
    "main.py",
)
DEFAULT_SINGLE_FILE = "submission_single_file.py"
LOCAL_MODULE_NAMES = frozenset(name[:-3] for name in RUNTIME_MODULES)


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


# --------------------------------------------------------------------------- #
# Single-file bundle (opt-in) -- for environments that require one main.py.
#
# The modular package remains the canonical source (CLAUDE.md Rule #4 forbids a
# monolith). This produces a *derived* artifact: the runtime modules concatenated
# in dependency order, with intra-package imports dropped and stdlib imports
# hoisted+de-duplicated once. It relies on rules_loader's EMBEDDED_RULES fallback,
# so the single file runs even with no rules_validated.json beside it.
# --------------------------------------------------------------------------- #
def _import_statement_text(node):
    """Reconstruct a top-level import node as a single source line."""
    if isinstance(node, ast.Import):
        parts = [a.name + (f" as {a.asname}" if a.asname else "") for a in node.names]
        return "import " + ", ".join(parts)
    names = ", ".join(a.name + (f" as {a.asname}" if a.asname else "") for a in node.names)
    return f"from {node.module} import {names}"


def _is_local_import(node):
    """True if an import refers to one of our own runtime modules."""
    if isinstance(node, ast.ImportFrom):
        return (node.module or "").split(".")[0] in LOCAL_MODULE_NAMES
    return any(a.name.split(".")[0] in LOCAL_MODULE_NAMES for a in node.names)


def bundle_single_file_source(root="."):
    """Return the concatenated single-file agent source as a string.

    Drops local imports, hoists+de-dupes external imports, and keeps every other
    line (comments and formatting) verbatim. Modules are emitted in
    :data:`SINGLE_FILE_ORDER` so each name is defined before it is referenced
    (notably ``SHED_TILES``, read in the agent's class body).
    """
    external_imports = []           # ordered, de-duplicated
    seen = set()
    sections = []

    for module in SINGLE_FILE_ORDER:
        with open(os.path.join(root, module), "r", encoding="utf-8") as handle:
            src = handle.read()
        src_lines = src.splitlines()
        tree = ast.parse(src, filename=module)

        strip = set()
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                end = node.end_lineno or node.lineno
                strip.update(range(node.lineno, end + 1))
                if not _is_local_import(node):
                    text = _import_statement_text(node)
                    if text not in seen:
                        seen.add(text)
                        external_imports.append(text)

        body = "\n".join(line for i, line in enumerate(src_lines, start=1)
                         if i not in strip).strip("\n")
        sections.append((module, body))

    out = [
        '"""Kaggriculture agent -- single-file submission bundle.',
        "",
        "AUTO-GENERATED by `build_submission.py --single-file`. Do NOT edit by hand:",
        "edit the modular sources and rebuild. The canonical architecture is the",
        "modular package (CLAUDE.md Rule #4 forbids a monolith); this file is only a",
        "submission artifact for environments that require a single main.py. It carries",
        "rules_loader's EMBEDDED_RULES fallback, so it runs with no rules_validated.json.",
        '"""',
        "",
    ]
    out.extend(external_imports)
    for module, body in sections:
        out += ["", "", f"# {'=' * 72}", f"# {module}", f"# {'=' * 72}", body]
    return "\n".join(out) + "\n"


def _bundle_check_season():
    """A short, deterministic, well-formed season for behavioural parity checks."""
    obs_list = []
    for step in [0, 1, 12, 23, 24, 100, 670, 700, 719]:
        obs_list.append({
            "step": step,
            "crops": [{"pos": (i % 10, (i * 3) % 10), "type": "WHEAT",
                       "misses": (step + i) % 2, "needs_water": (i % 2 == 0)}
                      for i in range(10)],
            "weeds": [(9, 9), (0, 5)] if step % 3 else [],
            "empty_tiles": [(5, 5), (6, 6)],
            "shed": {"WHEAT": 8 + (step % 5), "MELON": step % 4},
            "market_stocks": {"WHEAT": 100 + step, "MELON": 40},
            "money": 1000 + step * 3,
            "workers": [{"id": "farmer", "pos": (4, 4), "carried": step % 3}]
                       + [{"id": f"hand_{i}", "pos": (i, (i * 2) % 10),
                           "carried": (step + i) % 4} for i in range(10)],
        })
    return obs_list


def verify_single_file(source, root="."):
    """Compile the bundle and assert it behaves identically to the modular agent.

    Loads the generated source as an isolated module and plays a short
    well-formed season through both a bundled and a modular agent, comparing the
    emitted envelope turn-by-turn. Returns ``(ok, message)``.
    """
    try:
        code = compile(source, "<single_file_bundle>", "exec")
    except SyntaxError as exc:
        return False, f"bundle does not compile: {exc}"

    bundle_mod = importlib.util.module_from_spec(
        importlib.util.spec_from_loader("_kaggri_single_file_bundle", loader=None))
    # Give the synthetic module a real __file__ inside the project root so
    # rules_loader._resolve_path() finds rules_validated.json exactly as it will
    # for the written-to-disk bundle. Without it, __file__ is undefined and the
    # agent PASSes on a NameError -- a spurious divergence from the modular agent.
    bundle_mod.__file__ = os.path.abspath(os.path.join(root, DEFAULT_SINGLE_FILE))
    try:
        exec(code, bundle_mod.__dict__)
    except Exception as exc:  # import-time error would zero the season
        return False, f"bundle failed to import: {exc!r}"

    for symbol in ("KaggricultureAgent", "agent"):
        if not hasattr(bundle_mod, symbol):
            return False, f"bundle is missing the {symbol!r} entry point"

    # Behavioural parity: a fresh bundled agent must match a fresh modular one.
    import main as modular_main
    bundled = bundle_mod.KaggricultureAgent()
    modular = modular_main.KaggricultureAgent()
    for obs in _bundle_check_season():
        # Pass independent copies so neither agent can mutate the other's input.
        b_out = bundled(json_roundtrip(obs), {})
        m_out = modular(json_roundtrip(obs), {})
        if b_out != m_out:
            return False, f"bundle diverges from modular agent at step {obs['step']}"
    return True, "single-file bundle compiles, imports, and matches the modular agent"


def json_roundtrip(obj):
    """A deep, independent copy of a plain observation dict (lists, not tuples).

    Both agents receive structurally identical, independent inputs so a parity
    mismatch reflects a real behavioural difference, not shared-state aliasing.
    """
    return json.loads(json.dumps(obj))


def build_single_file(output=DEFAULT_SINGLE_FILE, root=".", check_only=False):
    """Generate, verify, and (unless check_only) write the single-file bundle."""
    absent = missing_files(root)
    # The rules file may be absent for a pure single-file agent (embedded
    # fallback), so only the runtime modules are strictly required here.
    module_absent = [p for p in absent if os.path.basename(p) in RUNTIME_MODULES]
    if module_absent:
        return False, ["missing required modules: "
                       + ", ".join(os.path.basename(p) for p in module_absent)]

    ok, messages = verify_rules(root)
    if not ok:
        return False, messages

    source = bundle_single_file_source(root)
    ok, detail = verify_single_file(source, root)
    messages.append(detail)
    if not ok:
        return False, messages

    if check_only:
        messages.append("check-only: single-file bundle verified, not written")
        return True, messages

    with open(output, "w", encoding="utf-8") as handle:
        handle.write(source)
    messages.append(f"wrote {output} ({len(source)} bytes, self-contained single file)")
    return True, messages


def main(argv=None):
    parser = argparse.ArgumentParser(description="Package the Kaggriculture agent for submission.")
    parser.add_argument("-o", "--output", default=None,
                        help=f"output path (default: {DEFAULT_OUTPUT}, "
                             f"or {DEFAULT_SINGLE_FILE} with --single-file)")
    parser.add_argument("--root", default=".", help="project root to package from")
    parser.add_argument("--check-only", action="store_true",
                        help="run verification only; do not write the artifact")
    parser.add_argument("--single-file", action="store_true",
                        help="emit a single self-contained main.py instead of the "
                             "multi-file tarball (derived artifact; source stays modular)")
    args = parser.parse_args(argv)

    if args.single_file:
        output = args.output or DEFAULT_SINGLE_FILE
        ok, messages = build_single_file(output, args.root, args.check_only)
    else:
        output = args.output or DEFAULT_OUTPUT
        ok, messages = build_submission(output, args.root, args.check_only)
    stream = sys.stdout if ok else sys.stderr
    for message in messages:
        print(("OK: " if ok else "FAIL: ") + message, file=stream)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
