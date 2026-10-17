"""bt_model.py -- Bradley-Terry robustness machinery for evaluating strategy.

The manager explicitly required an exogenous opponent pool and an evaluation
layer that maps tournament results to true strength via Bradley-Terry models
rather than simply maximizing expected terminal cash.

This module provides the statistical machinery to compute:
    1. Maximum-Likelihood Estimation (MLE) of agent strengths from a win/loss matrix.
    2. Pairwise win probabilities based on those strengths.
    3. The raw empirical head-to-head win rate, and a Wilson score confidence
       interval around *that observed proportion* (not the model estimate).
"""

import math
import random

class BradleyTerryModel:
    # Default add-λ regularization: every agent plays this many *symmetric*
    # pseudo-wins and pseudo-losses against a phantom average-strength opponent.
    # It replaces the old ``1e-6`` zero-win hack (a degenerate point strength) with
    # a principled Bayesian shrinkage toward the mean, and -- because it connects
    # every real agent to the phantom -- guarantees the estimation is well-posed
    # even when the real match graph is disconnected (see :meth:`components`).
    DEFAULT_REG_LAMBDA = 1.0

    def __init__(self, agents):
        """
        Initialize the model with a list of agent names.
        """
        self.agents = agents
        self.n = len(agents)
        self.agent_to_idx = {agent: i for i, agent in enumerate(agents)}
        # W[i][j] = number of times i beat j
        self.W = [[0] * self.n for _ in range(self.n)]
        # Maximum Likelihood Estimates for strengths (theta)
        self.strengths = [1.0] * self.n
        # Number of matches played between i and j
        self.N = [[0] * self.n for _ in range(self.n)]
        # Flat log of every recorded outcome as (winner_idx, loser_idx). Kept so
        # :meth:`strength_interval` can bootstrap-resample the raw matches.
        self.matches = []

    def add_match(self, winner, loser):
        """Record a match outcome."""
        if winner not in self.agent_to_idx or loser not in self.agent_to_idx:
            raise ValueError("Unknown agent in match")
        i = self.agent_to_idx[winner]
        j = self.agent_to_idx[loser]
        self.W[i][j] += 1
        self.N[i][j] += 1
        self.N[j][i] += 1
        self.matches.append((i, j))

    def fit(self, max_iterations=1000, tol=1e-6, reg_lambda=None):
        """
        Fit the Bradley-Terry model using Minorize-Maximization (MM) /
        Zermelo-Bradley-Terry iterative algorithm, regularized with add-λ
        pseudo-observations against a phantom average opponent.

        ``reg_lambda`` (default :data:`DEFAULT_REG_LAMBDA`) is the strength of a
        symmetric prior: each agent is credited ``λ`` pseudo-wins and charged
        ``λ`` pseudo-losses against a phantom opponent pinned at the mean
        strength (1.0, since strengths are normalized to sum to ``n``).  This
        shrinks every estimate gently toward the mean, so a winless agent gets a
        small-but-finite strength instead of the old ``1e-6`` sentinel, and two
        agents that never met are still jointly identifiable through the phantom
        rather than leaving the graph disconnected (cf. :meth:`components`).
        Set ``reg_lambda=0`` to recover the unregularized MLE.
        """
        lam = self.DEFAULT_REG_LAMBDA if reg_lambda is None else reg_lambda
        s_phantom = 1.0  # phantom opponent sits at the normalized mean strength
        wins = [sum(self.W[i]) for i in range(self.n)]

        for _ in range(max_iterations):
            new_strengths = [0.0] * self.n
            max_diff = 0.0

            for i in range(self.n):
                denominator = 0.0
                for j in range(self.n):
                    if i != j and self.N[i][j] > 0:
                        denominator += self.N[i][j] / (self.strengths[i] + self.strengths[j])
                # Add-λ phantom: 2λ pseudo-games (λ won, λ lost) against an
                # opponent of strength ``s_phantom``.  This term is always > 0,
                # so the denominator can never collapse to zero and no winless
                # agent needs a hand-picked floor.
                denominator += (2 * lam) / (self.strengths[i] + s_phantom)
                new_strengths[i] = (wins[i] + lam) / denominator

            # Normalize so that sum(strengths) = n
            total_strength = sum(new_strengths)
            for i in range(self.n):
                new_strengths[i] = new_strengths[i] * self.n / total_strength
                max_diff = max(max_diff, abs(new_strengths[i] - self.strengths[i]))

            self.strengths = new_strengths
            if max_diff < tol:
                break

    def win_probability(self, agent_a, agent_b):
        """Model estimate: P(agent_a beats agent_b) from fitted latent strengths.

        This is the Bradley-Terry model's smoothed estimate, which pools
        information across the whole tournament graph -- distinct from the raw
        head-to-head proportion (see :meth:`empirical_win_rate`).
        """
        i = self.agent_to_idx[agent_a]
        j = self.agent_to_idx[agent_b]
        return self.strengths[i] / (self.strengths[i] + self.strengths[j])

    def empirical_win_rate(self, agent_a, agent_b):
        """Raw head-to-head proportion: (a's wins over b) / (games a vs b).

        Returns ``None`` when the pair never met -- there is no observed
        frequency to report, and inventing 0.5 would fabricate a data point that
        was never played (a spurious "even" record).  Callers must render the
        no-data case explicitly (e.g. "n/a") rather than treating it as a real
        50%.  This is the direct observed frequency -- the quantity
        :meth:`confidence_interval` brackets -- and it deliberately does NOT use
        the fitted strengths, so callers can show the model estimate and the
        empirical proportion side by side.
        """
        i = self.agent_to_idx[agent_a]
        j = self.agent_to_idx[agent_b]
        n_matches = self.N[i][j]
        if n_matches == 0:
            return None
        return self.W[i][j] / n_matches

    def get_summary(self):
        """Return a sorted list of (agent, strength)."""
        return sorted(
            zip(self.agents, self.strengths),
            key=lambda x: x[1],
            reverse=True
        )

    def confidence_interval(self, agent_a, agent_b, z=1.96):
        """Wilson score interval for the *empirical* pairwise win rate of A vs B.

        NOTE: this brackets the observed head-to-head proportion
        (:meth:`empirical_win_rate`), NOT the model's fitted
        :meth:`win_probability`.  For the small match counts a Monte-Carlo
        sweep produces, a Wilson score interval on the direct pairwise records
        is more honest than an asymptotic-normal interval on the MLE, which
        would assume a sample size we do not have.  The two quantities differ:
        the model estimate pools information across the whole tournament graph,
        while this interval uses only the A-vs-B games.
        """
        i = self.agent_to_idx[agent_a]
        j = self.agent_to_idx[agent_b]
        wins_a = self.W[i][j]
        n_matches = self.N[i][j]

        if n_matches == 0:
            return (0.0, 1.0)

        p_hat = wins_a / n_matches
        # Wilson score interval for binomial proportion
        denominator = 1 + z**2 / n_matches
        center = (p_hat + z**2 / (2 * n_matches)) / denominator
        spread = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_matches)) / n_matches) / denominator

        return (max(0.0, center - spread), min(1.0, center + spread))

    def components(self):
        """Connected components of the *real* head-to-head graph (union-find).

        Two agents share a component iff a chain of actually-played matchups
        links them.  The regularizer in :meth:`fit` makes the *estimation*
        well-posed even across components (every agent is tied to the phantom),
        but a cross-component strength comparison rests on the prior, not on
        data -- so this is the honest signal of what the tournament actually
        measured.  Returns a list of lists of agent names, one per component.
        """
        parent = list(range(self.n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.N[i][j] > 0:
                    union(i, j)

        groups = {}
        for i in range(self.n):
            groups.setdefault(find(i), []).append(self.agents[i])
        return list(groups.values())

    def is_fully_connected(self):
        """True iff every agent is linked to every other through played games.

        When this is False, at least one pair of agents can only be ranked
        relative to each other through the prior, and callers should flag the
        comparison as not empirically grounded.
        """
        return len(self.components()) <= 1

    def strength_interval(self, agent, n_boot=200, alpha=0.05, seed=None):
        """Bootstrap percentile interval for one agent's fitted latent strength.

        Resamples the recorded matches with replacement ``n_boot`` times,
        refitting the model on each resample, and returns the
        ``(alpha/2, 1-alpha/2)`` percentiles of that agent's strength across the
        resamples -- a data-driven uncertainty band on the point estimate the MLE
        alone does not provide (#28).  ``seed`` makes the resampling
        deterministic for tests.  Returns ``None`` when no matches have been
        recorded (there is nothing to resample).
        """
        idx = self.agent_to_idx[agent]
        m = len(self.matches)
        if m == 0:
            return None

        rng = random.Random(seed)
        samples = []
        for _ in range(n_boot):
            boot = BradleyTerryModel(self.agents)
            for _ in range(m):
                w, l = self.matches[rng.randrange(m)]
                boot.W[w][l] += 1
                boot.N[w][l] += 1
                boot.N[l][w] += 1
                boot.matches.append((w, l))
            boot.fit()
            samples.append(boot.strengths[idx])

        samples.sort()
        lo_i = max(0, min(n_boot - 1, int((alpha / 2) * n_boot)))
        hi_i = max(0, min(n_boot - 1, int(math.ceil((1 - alpha / 2) * n_boot)) - 1))
        return (samples[lo_i], samples[hi_i])
