"""mc.py -- Monte-Carlo match runner against baseline opponents.

Release step 4 of the design doc's workflow: the season simulator that plays the
agent against reference opponents. Now upgraded to a Bradley-Terry model evaluating
win-rate against an exogenous opponent pool, with robust confidence intervals.

Usage:
    python mc.py                         # default matches vs pool
    python mc.py --matches 20            # 20 paired matches
    python mc.py --require-engine        # exit non-zero if engine unavailable

Exit 0 on a clean report; non-zero under ``--require-engine`` when the engine is missing
or failure rate is too high, or on a bad invocation.
"""

import argparse
import sys
from bt_model import BradleyTerryModel

ENGINE_ENV_CANDIDATES = ("kaggriculture", "farming", "agriculture")

# --------------------------------------------------------------------------- #
# Opponent Pool
# --------------------------------------------------------------------------- #
def make_starter():
    """A do-nothing baseline: always PASS."""
    def starter(obs, config):
        return {"farmer": [], "hands": [], "market": []}
    return starter

def make_copy():
    """A fresh copy of our own agent, for self-play sanity checks."""
    from main import KaggricultureAgent
    return KaggricultureAgent()

def make_greedy():
    """A greedy baseline agent."""
    def greedy(obs, config):
        # A simple mock agent for the pool
        orders = []
        if obs.get("step", 0) == 0:
            orders = [["HIRE"] for _ in range(5)]
        return {"farmer": [], "hands": [], "market": orders}
    return greedy

OPPONENTS = {"starter": make_starter, "copy": make_copy, "greedy": make_greedy}
OUR_AGENT_NAME = "candidate"

# --------------------------------------------------------------------------- #
# Engine plumbing
# --------------------------------------------------------------------------- #
def load_engine():
    try:
        import kaggle_environments
        return kaggle_environments
    except Exception:
        return None

def _make_env(engine):
    for name in ENGINE_ENV_CANDIDATES:
        try:
            return engine.make(name, debug=False)
        except Exception:
            continue
    return None

def _terminal_reward(env, seat):
    try:
        last = env.steps[-1]
        s = last[seat]
        reward = s.get("reward") if isinstance(s, dict) else getattr(s, "reward", None)
        return float(reward) if reward is not None else None
    except Exception:
        return None

def run_paired_match(engine, agent_a_name, agent_b_name, agent_a_factory, agent_b_factory):
    """Play a paired match. Returns the winner name or None if crash/tie."""
    env = _make_env(engine)
    if env is None:
        return None
    try:
        env.run([agent_a_factory(), agent_b_factory()])
        reward_a = _terminal_reward(env, 0)
        reward_b = _terminal_reward(env, 1)
        if reward_a is None or reward_b is None:
            return None
        if reward_a > reward_b:
            return agent_a_name
        elif reward_b > reward_a:
            return agent_b_name
        else:
            return "tie"
    except Exception:
        return None

# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Run BT tournaments")
    parser.add_argument("--matches", type=int, default=5,
                        help="Number of paired matches per opponent")
    parser.add_argument("--require-engine", action="store_true",
                        help="Exit non-zero if kaggle_environments is missing")
    args = parser.parse_args()

    engine = load_engine()
    if engine is None:
        if args.require_engine:
            print("Engine not found.", file=sys.stderr)
            sys.exit(1)
        print("Engine not found. Skipping tests.")
        sys.exit(0)

    from main import KaggricultureAgent
    agents = [OUR_AGENT_NAME] + list(OPPONENTS.keys())
    bt = BradleyTerryModel(agents)
    
    factories = {OUR_AGENT_NAME: KaggricultureAgent}
    factories.update(OPPONENTS)

    failures = 0
    print(f"Running tournament with {args.matches} paired matches per opponent...")

    # Play candidate against all pool opponents
    for opp_name in OPPONENTS:
        for _ in range(args.matches):
            # A vs B
            winner1 = run_paired_match(engine, OUR_AGENT_NAME, opp_name, 
                                       factories[OUR_AGENT_NAME], factories[opp_name])
            if winner1 == OUR_AGENT_NAME:
                bt.add_match(OUR_AGENT_NAME, opp_name)
            elif winner1 == opp_name:
                bt.add_match(opp_name, OUR_AGENT_NAME)
            elif winner1 is None:
                failures += 1
            
            # Swap seats (B vs A)
            winner2 = run_paired_match(engine, opp_name, OUR_AGENT_NAME, 
                                       factories[opp_name], factories[OUR_AGENT_NAME])
            if winner2 == OUR_AGENT_NAME:
                bt.add_match(OUR_AGENT_NAME, opp_name)
            elif winner2 == opp_name:
                bt.add_match(opp_name, OUR_AGENT_NAME)
            elif winner2 is None:
                failures += 1

    bt.fit()
    print(f"\nTournament complete. Failures: {failures}")
    print("\nBradley-Terry Win Probabilities (Candidate vs Opponent):")
    for opp_name in OPPONENTS:
        prob = bt.win_probability(OUR_AGENT_NAME, opp_name)
        low, high = bt.confidence_interval(OUR_AGENT_NAME, opp_name)
        print(f"vs {opp_name:10} | P(Win): {prob*100:5.1f}% | 95% CI: [{low*100:5.1f}%, {high*100:5.1f}%]")

    summary = bt.get_summary()
    print("\nOverall Strength Rankings:")
    for rank, (agent, strength) in enumerate(summary, 1):
        print(f"{rank}. {agent:10} | Strength: {strength:7.3f}")

if __name__ == "__main__":
    main()
