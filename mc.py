"""mc.py -- Monte-Carlo match runner + Bradley-Terry tournament analysis.

Release step 4 of the design doc's workflow: the season simulator that plays the
agent against reference opponents and reports the terminal-cash spread, so a
release can be judged against the $MIN_TERMINAL_TARGET bar before submission.

Six pure-Python opponents ship as local stand-ins so the harness is fully wired
-- and the Bradley-Terry pool is genuinely diverse -- even before the official
baselines are dropped in.  They are constructed without the engine (only *playing*
them needs it), so the opponent pool and the held-out evaluator are unit-tested
offline; only the real terminal-cash *numbers* wait on ``kaggle_environments``:

  * ``starter``   -- a do-nothing PASS agent (the floor any real agent must beat).
  * ``copy``      -- a fresh copy of our own agent (self-play sanity: symmetric
    play should straddle the target rather than collapse).
  * ``sell_only`` -- banks shed inventory every turn, never farms.
  * ``water_only``-- waters a crop underfoot when it needs it, else PASS.
  * ``greedy``    -- harvests/waters the tile underfoot and liquidates the shed.
  * ``varied``    -- a deterministic step-seeded mover (structurally distinct).

The Kaggle engine (``kaggle_environments``) is not installed in this workspace.
By default the harness reports that the engine is unavailable and exits 0
(local dev).  Use ``--require-engine`` to make engine absence a hard failure
(release gate).  ``--require-target`` makes the mean-falls-below check a
hard failure as well.

Episode failures (crashes) are tracked as first-class results, not silently
discarded.  The summary reports ``failures`` alongside ``n`` so survivorship
bias is visible.  ``--require-engine`` also fails if the failure rate exceeds
a threshold (default 20%).

``--tournament`` additionally fits a Bradley-Terry model over paired, both-seat
matches (see :mod:`bt_model`) when the engine is present.  It is analysis only
and never changes the release gate's exit code; it prints, per opponent, the
model win-probability *alongside* the empirical pairwise win-rate and its
Wilson confidence interval so the two (a latent-strength estimate vs. a raw
proportion) are never conflated.

Usage:
    python mc.py                         # default matches vs both opponents
    python mc.py --matches 20 --opponent copy
    python mc.py --require-target        # exit non-zero if mean < target (gate)
    python mc.py --require-engine        # exit non-zero if engine unavailable
    python mc.py --tournament            # add Bradley-Terry rankings (engine only)

Exit 0 on a clean report; non-zero under ``--require-target`` with a mean below
target, under ``--require-engine`` when the engine is missing or failure rate
is too high, or on a bad invocation.

Release-target caveat (#34): ``MIN_TERMINAL_TARGET`` ($35,000) is a *calibration
baseline*, measured against idle/starter opponents on a limited reference setup
-- NOT a universal competition-readiness gate.  It is deliberately overridable
(``--target``) and its enforcement is opt-in (``--require-target``) precisely so
it is not mistaken for a proof of readiness.  Recompute it when the opponent
pool, season count, or evaluation variance changes; a different matchup (e.g.
self-play, where two identical farms split one market) is a different
measurement this bar was never meant to clear.  See VALIDATION.md.
"""

import argparse
import math
import random
import statistics
import sys

from rules_loader import RulesLoader

# Candidate environment names to probe (the engine names it inconsistently
# across competitions); the first that makes successfully wins.
ENGINE_ENV_CANDIDATES = ("kaggriculture", "farming", "agriculture")


# --------------------------------------------------------------------------- #
# Opponents (all pure-Python; constructing them needs no engine)
# --------------------------------------------------------------------------- #
def _obs_get(obs, key, default=None):
    """Read ``key`` from a dict-or-attr observation, defaulting when absent."""
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _crops_by_pos(obs):
    """Index the observation's crops by ``(x, y)`` tile (best-effort, safe)."""
    out = {}
    for c in (_obs_get(obs, "crops", []) or []):
        pos = c.get("pos") if isinstance(c, dict) else None
        if pos is not None:
            out[tuple(pos)] = c
    return out


def _farmer_pos(obs):
    """The first worker's ``(x, y)`` position, or None if unreported."""
    workers = _obs_get(obs, "workers", []) or []
    if workers and isinstance(workers[0], dict) and workers[0].get("pos") is not None:
        return tuple(workers[0]["pos"])
    return None


def _shed_sell_orders(obs, cap=10):
    """Legal SELL orders for whatever the shed reports (capped)."""
    shed = _obs_get(obs, "shed", {}) or {}
    return [["SELL", item, int(qty)] for item, qty in shed.items()
            if qty and int(qty) > 0][:cap]


def make_starter():
    """A do-nothing baseline: always PASS. The floor a real agent must clear."""
    def starter(obs, config):
        return {"farmer": [], "hands": [], "market": []}
    return starter


def make_copy():
    """A fresh copy of our own agent, for self-play sanity checks."""
    from main import KaggricultureAgent
    return KaggricultureAgent()


def make_sell_only():
    """Banks shed inventory every turn but never farms -- a market-only floor."""
    def sell_only(obs, config):
        try:
            return {"farmer": [], "hands": [], "market": _shed_sell_orders(obs)}
        except Exception:
            return {"farmer": [], "hands": [], "market": []}
    return sell_only


def make_water_only():
    """Waters a crop the farmer stands on when it needs it, else PASS."""
    def water_only(obs, config):
        try:
            crops = _crops_by_pos(obs)
            pos = _farmer_pos(obs)
            crop = crops.get(pos) if pos is not None else None
            if crop and (crop.get("needs_water") or crop.get("misses", 0) >= 1):
                return {"farmer": ["WATER"], "hands": [], "market": []}
            return {"farmer": [], "hands": [], "market": []}
        except Exception:
            return {"farmer": [], "hands": [], "market": []}
    return water_only


def make_greedy():
    """Harvests/waters the tile underfoot and liquidates the shed each turn."""
    def greedy(obs, config):
        try:
            crops = _crops_by_pos(obs)
            pos = _farmer_pos(obs)
            crop = crops.get(pos) if pos is not None else None
            action = []
            if crop and crop.get("ready"):
                action = ["HARVEST"]
            elif crop and (crop.get("needs_water") or crop.get("misses", 0) >= 1):
                action = ["WATER"]
            return {"farmer": action, "hands": [], "market": _shed_sell_orders(obs)}
        except Exception:
            return {"farmer": [], "hands": [], "market": []}
    return greedy


def make_varied():
    """A deterministic step-seeded mover -- structurally distinct from the rest.

    Cycles the farmer through the four moves by turn index (no RNG, so play is
    reproducible), giving the tournament graph a policy that is neither passive
    nor economically motivated."""
    moves = ["NORTH", "SOUTH", "EAST", "WEST"]
    def varied(obs, config):
        try:
            step = int(_obs_get(obs, "step", 0) or 0)
            return {"farmer": [moves[step % 4]], "hands": [], "market": []}
        except Exception:
            return {"farmer": [], "hands": [], "market": []}
    return varied


OPPONENTS = {
    "starter": make_starter,
    "copy": make_copy,
    "sell_only": make_sell_only,
    "water_only": make_water_only,
    "greedy": make_greedy,
    "varied": make_varied,
}


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


def _terminal_reward(env, seat=0):
    """Pull a seat's terminal cash from the finished environment (seat 0 = us)."""
    try:
        last = env.steps[-1]
        s = last[seat]
        reward = s.get("reward") if isinstance(s, dict) else getattr(s, "reward", None)
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
# Bradley-Terry tournament (engine-gated) + held-out evaluator (pure)
# --------------------------------------------------------------------------- #
def bt_holdout_eval(agents, matches, train_frac=0.7, seed=None, reg_lambda=None):
    """Held-out evaluation of the Bradley-Terry model (pure -- no engine needed).

    Splits ``matches`` (a list of ``(winner, loser)`` agent-name tuples) into a
    train and a test partition, fits the model on the *train* partition only,
    and scores its predictions on the held-out *test* matches.  Reports:

      * ``accuracy`` -- fraction of test matches whose actual winner the model
        gave > 50% (a hard 0/1 score), and
      * ``log_loss`` -- mean ``-log P(actual winner)`` with the probability
        clamped to ``[1e-12, 1-1e-12]`` (a proper scoring rule that also
        punishes confident-but-wrong calls, not just the sign of the call).

    This is the engine-free half of the manager's evaluation-layer requirement
    (#29/#33): it exercises and unit-tests the evaluation *machinery* on given
    match records, so predicting-on-held-out-data is validated offline.  Only
    the live match data that would feed it still needs ``kaggle_environments`` --
    the code path is engine-free and unit-tested on synthetic matches.

    Returns a dict with ``n_train``, ``n_test``, ``accuracy`` and ``log_loss``.
    ``accuracy``/``log_loss`` are ``None`` when the test partition is empty (too
    few matches to hold any out), so callers never divide by zero or read a
    fabricated score.
    """
    from bt_model import BradleyTerryModel

    matches = list(matches)
    n = len(matches)
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    n_train = int(round(n * train_frac))
    # With >= 2 matches, guarantee at least one on each side so a valid split is
    # never silently collapsed to train-only (no test) or test-only (no fit).
    if n >= 2:
        n_train = max(1, min(n - 1, n_train))
    train = [matches[order[k]] for k in range(n_train)]
    test = [matches[order[k]] for k in range(n_train, n)]

    model = BradleyTerryModel(agents)
    for winner, loser in train:
        model.add_match(winner, loser)
    model.fit(reg_lambda=reg_lambda)

    if not test:
        return {"n_train": len(train), "n_test": 0,
                "accuracy": None, "log_loss": None}

    correct = 0
    total_ll = 0.0
    for winner, loser in test:
        p = model.win_probability(winner, loser)  # P(the actual winner wins)
        if p > 0.5:
            correct += 1
        p_clamped = min(1 - 1e-12, max(1e-12, p))
        total_ll += -math.log(p_clamped)
    return {
        "n_train": len(train),
        "n_test": len(test),
        "accuracy": correct / len(test),
        "log_loss": total_ll / len(test),
    }


def _play_pair(engine, factory_a, factory_b):
    """Play one paired episode; return ``(reward_a, reward_b)`` or ``(None, None)``
    if the engine could not make the env or the episode crashed."""
    env = _make_env(engine)
    if env is None:
        return None, None
    try:
        env.run([factory_a(), factory_b()])
    except Exception:
        return None, None
    return _terminal_reward(env, 0), _terminal_reward(env, 1)


def run_bt_tournament(engine, matches, opponents=None):
    """Fit a Bradley-Terry model over paired, both-seat matches vs the pool.

    Analysis only -- it never affects the release gate.  Each opponent is played
    ``matches`` times in *each* seat (both-seat) so first-mover advantage cannot
    bias the strengths.  Prints, per opponent, the model win-probability next to
    the empirical pairwise win-rate and its Wilson CI, keeping the latent-
    strength estimate and the raw proportion visibly distinct.  Returns the
    fitted :class:`bt_model.BradleyTerryModel`, or ``None`` if no game completed.
    """
    from bt_model import BradleyTerryModel
    from main import KaggricultureAgent

    opponents = list(opponents) if opponents is not None else list(OPPONENTS)
    our = "candidate"
    bt = BradleyTerryModel([our] + opponents)
    factories = {our: KaggricultureAgent}
    factories.update(OPPONENTS)

    completed = 0
    match_log = []  # (winner, loser) names, for the held-out evaluator
    for opp in opponents:
        for _ in range(matches):
            for a, b in ((our, opp), (opp, our)):  # both seats
                ra, rb = _play_pair(engine, factories[a], factories[b])
                if ra is None or rb is None or ra == rb:
                    continue
                winner, loser = (a, b) if ra > rb else (b, a)
                bt.add_match(winner, loser)
                match_log.append((winner, loser))
                completed += 1

    if completed == 0:
        print("Bradley-Terry tournament: no games completed.", file=sys.stderr)
        return None

    bt.fit()
    print(f"\nBradley-Terry tournament ({completed} completed paired games):")

    # Whether the played graph actually links every agent. When it does not, at
    # least one comparison rests on the regularizer's prior rather than on games
    # played, so we say so instead of printing a falsely authoritative ranking.
    if not bt.is_fully_connected():
        comps = bt.components()
        print(f"  WARNING: match graph is not fully connected ({len(comps)} "
              f"components: {comps}); cross-component strengths lean on the "
              f"prior, not on head-to-head data.")

    for opp in opponents:
        p_model = bt.win_probability(our, opp)
        p_emp = bt.empirical_win_rate(our, opp)
        low, high = bt.confidence_interval(our, opp)
        # empirical_win_rate is None when the pair never met -- render "n/a"
        # rather than a fabricated 50% (#25).
        emp_str = "   n/a" if p_emp is None else f"{p_emp * 100:5.1f}%"
        print(f"  vs {opp:9} | model P(win)={p_model * 100:5.1f}% | "
              f"empirical={emp_str} 95%CI=[{low * 100:4.1f}%, {high * 100:4.1f}%]")

    # Point strengths with a bootstrap uncertainty band (#28), so the ranking
    # is never read as more precise than the handful of games behind it.
    print("  strength rankings (with bootstrap 95% band):")
    for a, s in bt.get_summary():
        interval = bt.strength_interval(a, seed=0)
        band = "" if interval is None else f" [{interval[0]:.3f}, {interval[1]:.3f}]"
        print(f"    {a:9} = {s:.3f}{band}")

    # Held-out predictive check on the same match records (pure; no engine).
    holdout = bt_holdout_eval([our] + opponents, match_log, seed=0)
    if holdout["accuracy"] is None:
        print(f"  held-out eval: too few matches to split "
              f"(n_train={holdout['n_train']}, n_test=0)")
    else:
        print(f"  held-out eval (train={holdout['n_train']}, "
              f"test={holdout['n_test']}): accuracy={holdout['accuracy'] * 100:.0f}% "
              f"log-loss={holdout['log_loss']:.3f}")
    return bt


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
    parser.add_argument("--tournament", action="store_true",
                        help="fit a Bradley-Terry model over the pool (engine only; analysis only)")
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

    # Bradley-Terry analysis is opt-in and never gates the release.
    if args.tournament:
        run_bt_tournament(engine, args.matches, opponents=list(OPPONENTS))

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
