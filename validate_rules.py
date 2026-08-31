"""validate_rules.py -- check ``rules_validated.json`` for correctness.

Release step 1 of the design doc's workflow: validate the rules and their
provenance against the official engine. Two layers run here:

1. **Self-consistency checks** (always run, no engine needed). These catch the
   mistakes that actually happen when hand-editing the rule table: a crop we
   can grow but not price, an animal whose product has no market, a scrambled
   priority band, a price floor that drifted off $1. They are cheap and worth
   running on every edit.
2. **Engine comparison** (best effort). If ``kaggle_environments`` is importable
   and exposes the farming environment, we compare our ``constants`` against the
   engine's configuration defaults and report any drift. The engine is usually
   absent in local dev, so by default a missing engine is reported and skipped
   rather than failed; pass ``--strict`` to make a missing/unusable engine an
   error (for a release gate that must run where the engine is installed).

Usage:
    python validate_rules.py                     # self-checks + engine if present
    python validate_rules.py rules_validated.json
    python validate_rules.py --strict            # require the engine to be present

Exit 0 when every check passes (engine skipped is not a failure unless
``--strict``); non-zero if any check fails.
"""

import argparse
import sys

from rules_loader import RulesLoader

# Candidate environment names to probe in kaggle_environments. The engine is
# not installed in this workspace, so discovery is best-effort and defensive.
ENGINE_ENV_CANDIDATES = ("kaggriculture", "farming", "agriculture")

# The band ordering the scheduler and design doc agree on (highest first).
EXPECTED_BAND_ORDER = (
    ("WATER", "priority_dying"),
    ("HARVEST_ANIMAL", "priority"),
    ("HARVEST_CROP", "priority"),
    ("FEED", "priority"),
    ("WATER", "priority_bonus"),
    ("WATER", "priority_ongoing"),
    ("PLANT", "priority"),
    ("DIG", "priority"),
    ("WATER", "priority_normal"),
    ("CARE", "priority"),
)


# --------------------------------------------------------------------------- #
# Self-consistency checks (engine-independent)
# --------------------------------------------------------------------------- #
def self_consistency_checks(rules):
    """Return a list of ``(name, ok, detail)`` for engine-independent checks."""
    results = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok), detail))

    constants = rules.get("constants", {})
    check("price_floor is $1", constants.get("price_floor") == 1,
          f"got {constants.get('price_floor')}")
    check("shed_size positive", constants.get("shed_size", 0) > 0,
          f"got {constants.get('shed_size')}")
    check("season_length is whole days",
          constants.get("season_length", 0) > 0 and constants.get("season_length", 0) % 24 == 0,
          f"got {constants.get('season_length')}")
    check("max_market_orders positive", constants.get("max_market_orders", 0) > 0,
          f"got {constants.get('max_market_orders')}")
    check("MIN_TERMINAL_TARGET <= CORRECTION_THRESHOLD",
          constants.get("MIN_TERMINAL_TARGET", 0) <= constants.get("CORRECTION_THRESHOLD", 0),
          f"{constants.get('MIN_TERMINAL_TARGET')} vs {constants.get('CORRECTION_THRESHOLD')}")

    # Every sellable crop must have a market entry, else we can grow what we
    # cannot price/sell.
    market = rules.get("market_params", {})
    for crop in rules.get("crop_params", {}):
        check(f"crop {crop} is priced", crop in market, "no market_params entry")

    # Fertilized yield never below unfertilized (the buy decision relies on it).
    for crop, params in rules.get("crop_params", {}).items():
        if params.get("type") == "one-time":
            fert = params.get("yield_fertilized", params.get("yield_no_fertilizer", 0))
            base = params.get("yield_no_fertilizer", 0)
            check(f"crop {crop} fertilized yield >= base", fert >= base,
                  f"{fert} < {base}")

    # Animals: product must be priced, and the home must be buildable.
    for animal, params in rules.get("animal_params", {}).items():
        product = params.get("produces")
        check(f"animal {animal} product priced", product in market,
              f"product {product!r} has no market_params")
        check(f"animal {animal} home buildable", params.get("home") in ("COOP", "PASTURE"),
              f"home {params.get('home')!r} not COOP/PASTURE")

    # Priority bands: all numeric and strictly descending in the documented order.
    dr = rules.get("daily_routines", {})
    band_values = []
    bands_ok = True
    for section, key in EXPECTED_BAND_ORDER:
        value = dr.get(section, {}).get(key)
        if not isinstance(value, (int, float)):
            check(f"band {section}.{key} present & numeric", False, f"got {value!r}")
            bands_ok = False
        else:
            band_values.append((f"{section}.{key}", value))
    if bands_ok:
        descending = all(band_values[i][1] > band_values[i + 1][1]
                         for i in range(len(band_values) - 1))
        check("priority bands strictly descending", descending,
              " ".join(f"{n}={v}" for n, v in band_values))

    return results


# --------------------------------------------------------------------------- #
# Engine comparison (best effort)
# --------------------------------------------------------------------------- #
def load_engine_constants():
    """Best-effort load of the engine's configuration defaults.

    Returns a flat dict of engine-defined constants, or ``None`` when the
    ``kaggle_environments`` package (or the farming environment within it) is
    not available. Never raises -- discovery failures degrade to ``None``.
    """
    try:
        import kaggle_environments  # noqa: F401  (import guarded on purpose)
    except Exception:
        return None

    for name in ENGINE_ENV_CANDIDATES:
        try:
            env = kaggle_environments.make(name)
        except Exception:
            continue
        # Environments expose defaults through specification/configuration; pull
        # whatever scalar config we can find without assuming an exact schema.
        for attr in ("configuration", "specification"):
            spec = getattr(env, attr, None)
            if isinstance(spec, dict) and spec:
                config = spec.get("configuration", spec)
                if isinstance(config, dict):
                    return {k: v for k, v in config.items() if isinstance(v, (int, float, str))}
    return None


def compare_with_engine(rules, engine_constants):
    """Return ``(name, ok, detail)`` rows for keys shared with the engine."""
    results = []
    constants = rules.get("constants", {})
    overlap = set(constants) & set(engine_constants)
    if not overlap:
        results.append(("engine overlap", True, "no shared constant keys to compare"))
        return results
    for key in sorted(overlap):
        ours, theirs = constants[key], engine_constants[key]
        results.append((f"engine constant {key}", ours == theirs, f"local={ours} engine={theirs}"))
    return results


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def validate(rules_path="rules_validated.json", strict=False):
    """Run all checks. Returns ``(ok, results, engine_status)``."""
    rules = RulesLoader(rules_path).load_rules()
    results = self_consistency_checks(rules)

    engine_constants = load_engine_constants()
    if engine_constants is None:
        engine_status = "unavailable"
        if strict:
            results.append(("engine present (strict)", False,
                            "kaggle_environments not importable / no farming env"))
        else:
            results.append(("engine comparison", True,
                            "kaggle_environments unavailable -- skipped (use --strict to require)"))
    else:
        engine_status = "compared"
        results.extend(compare_with_engine(rules, engine_constants))

    ok = all(row[1] for row in results)
    return ok, results, engine_status


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate rules_validated.json.")
    parser.add_argument("rules_path", nargs="?", default="rules_validated.json",
                        help="path to the rules JSON (default: rules_validated.json)")
    parser.add_argument("--strict", action="store_true",
                        help="fail if the kaggle_environments engine is unavailable")
    args = parser.parse_args(argv)

    try:
        ok, results, engine_status = validate(args.rules_path, args.strict)
    except (OSError, ValueError) as exc:
        print(f"FAIL: could not load rules: {exc}", file=sys.stderr)
        return 1

    for name, passed, detail in results:
        tag = "OK  " if passed else "FAIL"
        # detail is the failure explanation, so only show it when a check fails.
        suffix = "" if passed or not detail else f" -- {detail}"
        print(f"[{tag}] {name}{suffix}", file=sys.stdout if passed else sys.stderr)

    passed_n = sum(1 for r in results if r[1])
    print(f"\n{passed_n}/{len(results)} checks passed (engine: {engine_status})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
