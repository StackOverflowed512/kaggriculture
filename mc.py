"""mc.py -- Monte-Carlo match runner against baseline opponents.

Release step 4 of the design doc's workflow: the season simulator that plays the
agent against reference opponents and reports the terminal-cash spread, so a
release can be judged against the $MIN_TERMINAL_TARGET bar before submission.

Two opponents ship as local stand-ins so the harness is fully wired even before
the official baselines are dropped in:

  * ``starter`` -- a do-nothing PASS agent (the floor any real agent must beat).
  * ``copy``    -- a fresh copy of our own agent (self-play sanity: symmetric
    play should straddle the target rather than collapse).

The Kaggle engine (``kaggle_environments``) is not installed in this workspace.
By default the harness reports that the engine is unavailable and exits 0
(local dev).  Use ``--require-engine`` to make engine absence a hard failure
(release gate).  ``--require-target`` makes the mean-falls-below check a
hard failure as well.

Episode failures (crashes) are tracked as first-class results, not silently
discarded.  The summary reports ``failures`` alongside ``n`` so survivorship
bias is visible.  ``--require-engine`` also fails if the failure rate exceeds
a threshold (default 20%).

Usage:
    python mc.py                         # default matches vs both opponents
    python mc.py --matches 20 --opponent copy
    python mc.py --require-target        # exit non-zero if mean < target (gate)
    python mc.py --require-engine        # exit non-zero if engine unavailable

Exit 0 on a clean report; non-zero under ``--require-target`` with a mean below
target, under ``--require-engine`` when the engine is missing or failure rate
is too high, or on a bad invocation.
"""

import argparse
import math
import statistics
import sys

from rules_loader import RulesLoader

# Candidate environment names to probe (the engine names it inconsistently
# across competitions); the first that makes successfully wins.
ENGINE_ENV_CANDIDATES = ("kaggriculture", "farming", "agriculture")


# --------------------------------------------------------------------------- #
# Opponents
# --------------------------------------------------------------------------- #
def make_starter():
    """A do-nothing baseline: always PASS. The floor a real agent must clear."""
    def starter(obs, config):
        return {"farmer": [], "hands": [], "market": []}
    return starter


def make_copy():
    """A fresh copy of our own agent, for self-play sanity checks."""
    from main import KaggricultureAgent
    return KaggricultureAgent()


OPPONENTS = {"starter": make_starter, "copy": make_copy}


# --------------------------------------------------------------------------- #
# Engine plumbing
# --------------------------------------------------------------------------- #
def load_engine():
    """Return the ``kaggle_environments`` module, or ``None`` if unimportable."""
    try:
        import kaggle_environments
        return kaggle_environments
    except Exception:
        return None


def _make_env(engine):
    """Make the farming environment under whatever name the engine exposes."""
    for name in ENGINE_ENV_CANDIDATES:
        try:
            return engine.make(name, debug=False)
        except Exception:
            continue
    return None


def _terminal_reward(env):
    """Pull our agent's terminal cash from the finished environment."""
    try:
        last = env.steps[-1]
        seat0 = last[0]
        reward = seat0.get("reward") if isinstance(seat0, dict) else getattr(seat0, "reward", None)
        return float(reward) if reward is not None else None
    except Exception:
        return None


def run_matches(matches, opponent, engine=None, our_agent_factory=None):
    """Play ``matches`` episodes of our agent vs ``opponent``.

    Returns a dict with ``scores`` (list of completed terminal-cash values),
    ``failures`` (count of episodes that crashed), and ``n`` (total requested).
    ``None`` is returned only when the engine itself is unavailable.
    """
    engine = engine if engine is not None else load_engine()
    if engine is None:
        return None
    if opponent not in OPPONENTS:
        raise ValueError(f"unknown opponent {opponent!r}; choose from {sorted(OPPONENTS)}")

    from main import KaggricultureAgent
    our_agent_factory = our_agent_factory or KaggricultureAgent

    scores = []
    failures = 0
    for _ in range(matches):
        env = _make_env(engine)
        if env is None:
            return None  # engine present but no farming env
        try:
            env.run([our_agent_factory(), OPPONENTS[opponent]()])
        except Exception:
            failures += 1  # track, don't discard
            continue
        score = _terminal_reward(env)
        if score is not None:
            scores.append(score)
        else:
            failures += 1  # no reward extracted = effectively a failure
    return {"scores": scores, "failures": failures, "n": matches}


# --------------------------------------------------------------------------- #
# Scoring (pure -- unit-tested without the engine)
# --------------------------------------------------------------------------- #
def summarize(scores, target, failures=0, n=None):
    """Summarise a list of terminal-cash scores against ``target``.

    Pure and engine-free. Returns a dict with count, mean/min/max, spread,
    stdev, 95% confidence interval, how many fell below target, pass rate,
    and failure count.  An empty list yields a well-formed dict with None
    statistics.
    """
    n_scores = len(scores)
    n_total = n if n is not None else n_scores
    if n_scores == 0:
        return {"n": 0, "failures": failures, "mean": None, "min": None,
                "max": None, "spread": None, "stdev": None,
                "ci_low": None, "ci_high": None,
                "below_target": 0, "pass_rate": None, "target": target}
    below = sum(1 for s in scores if s < target)
    mean = statistics.fmean(scores)
    stdev = statistics.pstdev(scores) if n_scores > 1 else 0.0
    # 95% CI using normal approximation (fine for n >= 10)
    if n_scores >= 10 and stdev > 0:
        ci_margin = 1.96 * stdev / math.sqrt(n_scores)
        ci_low = mean - ci_margin
        ci_high = mean + ci_margin
    else:
        ci_low = ci_high = mean
    return {
        "n": n_scores,
        "failures": failures,
        "mean": mean,
        "min": min(scores),
        "max": max(scores),
        "spread": max(scores) - min(scores),
        "stdev": stdev,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "below_target": below,
        "pass_rate": (n_scores - below) / n_scores,
        "target": target,
    }


def format_report(opponent, summary):
    """Render a one-block human report for a summarised opponent sweep."""
    if summary["n"] == 0:
        return (f"vs {opponent}: no completed matches "
                f"(failures={summary['failures']})")
    ci_str = ""
    if summary.get("ci_low") is not None and summary.get("ci_high") is not None:
        ci_str = f" 95%CI=[${summary['ci_low']:,.0f}, ${summary['ci_high']:,.0f}]"
    fail_str = f" failures={summary['failures']}" if summary["failures"] else ""
    return (
        f"vs {opponent}: {summary['n']} matches{fail_str} | "
        f"mean=${summary['mean']:,.0f} min=${summary['min']:,.0f} "
        f"max=${summary['max']:,.0f} spread=${summary['spread']:,.0f} "
        f"stdev=${summary['stdev']:,.0f}{ci_str} | "
        f"{summary['n'] - summary['below_target']}/{summary['n']} >= "
        f"${summary['target']:,.0f} (pass {summary['pass_rate'] * 100:.0f}%)"
    )


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main(argv=None):
    parser = argparse.ArgumentParser(description="Monte-Carlo matches vs baseline opponents.")
    parser.add_argument("--matches", type=int, default=10, help="episodes per opponent")
    parser.add_argument("--opponent", choices=sorted(OPPONENTS) + ["all"], default="all",
                        help="opponent to play (default: all)")
    parser.add_argument("--target", type=float, default=None,
                        help="terminal-cash bar (default: rules MIN_TERMINAL_TARGET)")
    parser.add_argument("--rules", default="rules_validated.json", help="rules file")
    parser.add_argument("--require-target", action="store_true",
                        help="exit non-zero if any opponent's mean falls below target")
    parser.add_argument("--require-engine", action="store_true",
                        help="exit non-zero if the engine is unavailable or failure rate > 20%%")
    args = parser.parse_args(argv)

    rules = RulesLoader(args.rules).load_rules()
    target = args.target if args.target is not None else rules["constants"]["MIN_TERMINAL_TARGET"]

    engine = load_engine()
    if engine is None:
        print("kaggle_environments unavailable -- cannot play matches.",
              file=sys.stderr)
        if args.require_engine:
            print("FAIL: --require-engine set but engine is not installed",
                  file=sys.stderr)
            return 1
        if args.require_target:
            print("FAIL: --require-target set but no matches could be played",
                  file=sys.stderr)
            return 1
        return 0

    opponents = sorted(OPPONENTS) if args.opponent == "all" else [args.opponent]
    all_pass = True
    for opponent in opponents:
        result = run_matches(args.matches, opponent, engine=engine)
        if result is None:
            print(f"vs {opponent}: engine environment unavailable", file=sys.stderr)
            all_pass = False
            continue
        summary = summarize(result["scores"], target,
                            failures=result["failures"], n=result["n"])
        print(format_report(opponent, summary))
        # Failure rate check
        if args.require_engine and result["n"] > 0:
            failure_rate = result["failures"] / result["n"]
            if failure_rate > 0.2:
                print(f"  WARN: failure rate {failure_rate*100:.0f}% exceeds 20% threshold",
                      file=sys.stderr)
                all_pass = False
        if summary["n"] == 0 or (summary["mean"] is not None and summary["mean"] < target):
            all_pass = False

    if args.require_target and not all_pass:
        print(f"\nFAIL: at least one opponent's mean fell below ${target:,.0f}",
              file=sys.stderr)
        return 1
    if args.require_engine and not all_pass:
        print("\nFAIL: engine validation failed (missing engine or high failure rate)",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
