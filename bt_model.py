"""bt_model.py -- Bradley-Terry robustness machinery for evaluating strategy.

The manager explicitly required an exogenous opponent pool and an evaluation
layer that maps tournament results to true strength via Bradley-Terry models
rather than simply maximizing expected terminal cash.

This module provides the statistical machinery to compute:
    1. Maximum-Likelihood Estimation (MLE) of agent strengths from a win/loss matrix.
    2. Pairwise win probabilities based on those strengths.
    3. Robust confidence intervals around those probabilities.
"""

import math

class BradleyTerryModel:
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

    def add_match(self, winner, loser):
        """Record a match outcome."""
        if winner not in self.agent_to_idx or loser not in self.agent_to_idx:
            raise ValueError("Unknown agent in match")
        i = self.agent_to_idx[winner]
        j = self.agent_to_idx[loser]
        self.W[i][j] += 1
        self.N[i][j] += 1
        self.N[j][i] += 1

    def fit(self, max_iterations=1000, tol=1e-6):
        """
        Fit the Bradley-Terry model using Minorize-Maximization (MM) /
        Zermelo-Bradley-Terry iterative algorithm.
        """
        wins = [sum(self.W[i]) for i in range(self.n)]
        
        for _ in range(max_iterations):
            new_strengths = [0.0] * self.n
            max_diff = 0.0
            
            for i in range(self.n):
                if wins[i] == 0:
                    new_strengths[i] = 1e-6 # prevent 0 strength
                    continue
                    
                denominator = 0.0
                for j in range(self.n):
                    if i != j and self.N[i][j] > 0:
                        denominator += self.N[i][j] / (self.strengths[i] + self.strengths[j])
                
                if denominator > 0:
                    new_strengths[i] = wins[i] / denominator
                else:
                    new_strengths[i] = self.strengths[i]

            # Normalize so that sum(strengths) = n
            total_strength = sum(new_strengths)
            for i in range(self.n):
                new_strengths[i] = new_strengths[i] * self.n / total_strength
                max_diff = max(max_diff, abs(new_strengths[i] - self.strengths[i]))
                
            self.strengths = new_strengths
            if max_diff < tol:
                break

    def win_probability(self, agent_a, agent_b):
        """Probability that agent_a beats agent_b."""
        i = self.agent_to_idx[agent_a]
        j = self.agent_to_idx[agent_b]
        return self.strengths[i] / (self.strengths[i] + self.strengths[j])

    def get_summary(self):
        """Return a sorted list of (agent, strength)."""
        return sorted(
            zip(self.agents, self.strengths),
            key=lambda x: x[1],
            reverse=True
        )

    def confidence_interval(self, agent_a, agent_b, z=1.96):
        """
        Calculate a robust confidence interval for the win probability of A vs B
        using asymptotic variance of the MLE.
        """
        # For small MC samples, a simple binomial proportion CI (Wilson score)
        # on the direct pairwise match records is more robust than assuming
        # independence of overall cash variance.
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
