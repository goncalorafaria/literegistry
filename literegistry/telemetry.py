import time
from typing import Dict, List


class LatencyMetricAggregator:

    def __init__(
        self,
        initial_optimistic_mean: float = 0.1,
        prior_observations: int = 5,
        decay_factor: float = 0.9,
    ):
        """
        Tracks URI response times with optimistic initialization

        Args:
            initial_optimistic_mean: Starting assumption for response times (seconds)
            prior_observations: Number of "virtual" observations for Bayesian averaging
            decay_factor: Weight for exponential moving average (0.9 = 90% weight on history)
        """
        self.prior_observations = prior_observations
        self.initial_mean = initial_optimistic_mean
        self.decay_factor = decay_factor

        # Stores tuple of (weighted_sum, total_weight)
        self.metrics = {}

    def report(self, uri: str, response_time: float):
        """Update metrics for a URI with new response time measurement"""

        if uri in self.metrics:

            current_sum, current_count, _ = self.metrics[uri]

            # Exponential moving average update
            new_sum = self.decay_factor * current_sum + response_time
            new_count = self.decay_factor * current_count + 1

            self.metrics[uri] = (new_sum, new_count, time.time())

        else:
            # Initialize with prior observations
            self.metrics[uri] = (
                response_time,
                self.prior_observations,
                time.time(),
            )

    def get_estimated_latency(self, uri: str) -> float:
        """Get current latency estimate for a URI"""

        if uri not in self.metrics:
            return self.initial_mean
        else:
            total, count, _ = self.metrics[uri]
            return total / count if count != 0 else self.initial_mean

    def get_all_metrics(self) -> Dict[str, float]:
        """Return copy of all current latency estimates"""
        return {uri: self.get_estimated_latency(uri) for uri in self.metrics.keys()}

    def prune_inactive(self, max_age_seconds: int = 60 * 15):
        """Remove entries that haven't been updated in X seconds"""
        cutoff = time.time() - max_age_seconds
        # Requires Python 3.7+ for dict comprehensions
        self.metrics = {
            uri: val for uri, val in self.metrics.items() if val[-1] > cutoff
        }
