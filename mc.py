"""mc.py -- Monte-Carlo match runner against baseline opponents.

Release step 4 of the design doc's workflow: the season simulator that plays the
agent against reference opponents and reports the terminal-cash spread, so a
release can be judged against the $MIN_TERMINAL_TARGET bar before submission.

Two opponents ship as local stand-ins so the harness is fully wired even before
the official baselines are dropped in:

  * ``starter`` -- a do-nothing PASS agent (the floor any real agent must beat).
  * ``copy``    -- a fresh copy of our own agent (self-play sanity: symmetric
    play should straddle the target rather than collapse).

The Kaggle engine (``kaggle_environments``) is not installed in this workspace,
so :func:`run_matches` returns ``None`` when it cannot import/make the
environment, and :func:`main` reports that and skips rather than failing. The
scoring math lives in the pure :func:`summarize`, which is unit-tested directly
on synthetic score lists and needs no engine.

Usage:
    python mc.py                         # default matches vs both opponents
    python mc.py --matches 20 --opponent copy
    python mc.py --require-target        # exit non-zero if mean < target (gate)

Exit 0 on a clean report (or when the engine is unavailable and no gate is
requested); non-zero only under ``--require-target`` with a mean below target,
or on a bad invocation.
"""

import argparse
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
# Engine plumbing (best effort -- the engine is usually absent locally)
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
    # kaggle_environments stores per-agent state; the last step holds terminal
    # rewards. Our agent runs in seat 0. Guard every access -- schemas vary.
    try:
        last = env.steps[-1]
        seat0 = last[0]
        reward = seat0.get("reward") if isinstance(seat0, dict) else getattr(seat0, "reward", None)
        return float(reward) if reward is not None else None
    except Exception:
        return None


def run_matches(matches, opponent, engine=None, our_agent_factory=None):
    """Play ``matches`` episodes of our agent vs ``opponent``.

    Returns a list of our terminal-cash scores, or ``None`` when the engine is
    unavailable (or the environment cannot be made). Never raises on engine
    quirks -- a match that errors contributes no score.
    """
    engine = engine if engine is not None else load_engine()
    if engine is None:
        return None
    if opponent not in OPPONENTS:
        raise ValueError(f"unknown opponent {opponent!r}; choose from {sorted(OPPONENTS)}")

    from main import KaggricultureAgent
    our_agent_factory = our_agent_factory or KaggricultureAgent

    scores = []
    for _ in range(matches):
        env = _make_env(engine)
        if env is None:
            return None  # engine present but no farming env -- treat as unavailable
        try:
            env.run([our_agent_factory(), OPPONENTS[opponent]()])
        except Exception:
            continue  # a crashed episode yields no score, never aborts the sweep
        score = _terminal_reward(env)
        if score is not None:
            scores.append(score)
    return scores


# --------------------------------------------------------------------------- #
# Scoring (pure -- unit-tested without the engine)
# --------------------------------------------------------------------------- #
def summarize(scores, target):
    """Summarise a list of terminal-cash scores against ``target``.

    Pure and engine-free. Returns a dict with the count, mean/min/max, spread
    (max-min), population stdev, how many fell below target, and the pass rate.
    An empty list yields a well-formed dict with ``None`` statistics.
    """
    n = len(scores)
    if n == 0:
        return {"n": 0, "mean": None, "min": None, "max": None, "spread": None,
                "stdev": None, "below_target": 0, "pass_rate": None, "target": target}
    below = sum(1 for s in scores if s < target)
    return {
        "n": n,
        "mean": statistics.fmean(scores),
        "min": min(scores),
        "max": max(scores),
        "spread": max(scores) - min(scores),
        "stdev": statistics.pstdev(scores),
        "below_target": below,
        "pass_rate": (n - below) / n,
        "target": target,
    }


def format_report(opponent, summary):
    """Render a one-block human report for a summarised opponent sweep."""
    if summary["n"] == 0:
        return f"vs {opponent}: no completed matches (engine unavailable or all errored)"
    return (
        f"vs {opponent}: {summary['n']} matches | "
        f"mean=${summary['mean']:,.0f} min=${summary['min']:,.0f} "
        f"max=${summary['max']:,.0f} spread=${summary['spread']:,.0f} "
        f"stdev=${summary['stdev']:,.0f} | "
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
    args = parser.parse_args(argv)

    rules = RulesLoader(args.rules).load_rules()
    target = args.target if args.target is not None else rules["constants"]["MIN_TERMINAL_TARGET"]

    engine = load_engine()
    if engine is None:
        print("kaggle_environments unavailable -- cannot play matches (skipped). "
              "Install the engine to run Monte-Carlo seasons.", file=sys.stderr)
        # Not a failure by default: local dev has no engine. A gate can still
        # demand it by treating "no data" as below-target below.
        if args.require_target:
            print("FAIL: --require-target set but no matches could be played", file=sys.stderr)
            return 1
        return 0

    opponents = sorted(OPPONENTS) if args.opponent == "all" else [args.opponent]
    all_pass = True
    for opponent in opponents:
        scores = run_matches(args.matches, opponent, engine=engine)
        summary = summarize(scores or [], target)
        print(format_report(opponent, summary))
        if summary["n"] == 0 or (summary["mean"] is not None and summary["mean"] < target):
            all_pass = False

    if args.require_target and not all_pass:
        print(f"\nFAIL: at least one opponent's mean fell below ${target:,.0f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
