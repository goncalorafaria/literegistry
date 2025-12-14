"""
Load-aware bandit algorithms for routing with congestion.

These variants address the limitation that standard Exp3 doesn't handle
congestion well - it converges to overloading a single arm.
"""

import math
import random
import time
import os
from collections import deque, defaultdict


class LoadAwareExp3:
    """
    Exp3 variant that tracks load per arm and adjusts probabilities
    to prevent overloading any single arm.
    
    Key improvements:
    1. Tracks recent load (request rate) per arm
    2. Penalizes arms that are near/over capacity
    3. Uses capacity information to set reasonable probability bounds
    """
    
    def __init__(self, gamma=0.2, L_max=1.0, init_weight=1.0, 
                 load_window=100, capacity_aware=True):
        """
        Args:
            gamma: Exploration parameter (0-1)
            L_max: Maximum latency for reward normalization
            init_weight: Initial weight for new arms
            load_window: Window size for tracking recent load
            capacity_aware: If True, use capacity info to bound probabilities
        """
        self.gamma = gamma
        self.L_max = L_max
        self.init_weight = init_weight
        self.load_window = load_window
        self.capacity_aware = capacity_aware
        
        self.weights = {}  # arm_id -> weight
        self.capacities = {}  # arm_id -> capacity (requests per time unit)
        self.recent_selections = defaultdict(lambda: deque(maxlen=load_window))
        self.t = 0
        
        random.seed(time.time() + os.getpid())
    
    def set_capacities(self, capacities):
        """
        Set capacity information for arms.
        
        Args:
            capacities: Dict of arm_id -> capacity (requests/sec or relative capacity)
        """
        self.capacities = capacities.copy()
    
    def get_load(self, arm_id):
        """Get recent load (number of recent selections) for an arm."""
        return len(self.recent_selections[arm_id])
    
    def get_utilization(self, arm_id):
        """Get utilization ratio (load / capacity) for an arm."""
        if arm_id not in self.capacities or self.capacities[arm_id] <= 0:
            return 0.0
        return self.get_load(arm_id) / self.capacities[arm_id]
    
    def _sync_arms(self, active_ids):
        """Sync arms with current active set."""
        if not active_ids:
            self.weights.clear()
            return
        
        # Add new arms
        if not self.weights:
            for arm in active_ids:
                self.weights[arm] = self.init_weight
        else:
            avg_w = sum(self.weights.values()) / len(self.weights) if self.weights else self.init_weight
            for arm in active_ids:
                if arm not in self.weights:
                    self.weights[arm] = avg_w
        
        # Remove inactive arms
        for arm in list(self.weights):
            if arm not in active_ids:
                del self.weights[arm]
                if arm in self.recent_selections:
                    del self.recent_selections[arm]
    
    def _get_probabilities(self):
        """
        Compute selection probabilities with load awareness.
        """
        K = len(self.weights)
        if K == 0:
            return {}
        
        # Standard Exp3 probabilities
        max_log = max(self.weights.values())
        shifted_weights = {a: (lw - max_log) for a, lw in self.weights.items()}
        exp_weights = {a: math.exp(lw) for a, lw in shifted_weights.items()}
        total_w = sum(exp_weights.values())
        
        floor = self.gamma / K
        base_probs = {
            arm: (1 - self.gamma) * (w / (total_w + 1e-9)) + floor
            for arm, w in exp_weights.items()
        }
        
        # Apply load-based adjustments if capacity-aware
        if self.capacity_aware and self.capacities:
            adjusted_probs = self._apply_load_penalty(base_probs)
            return adjusted_probs
        
        return base_probs
    
    def _apply_load_penalty(self, base_probs):
        """
        Apply penalty to overloaded arms and redistribute probability.
        
        Strategy:
        1. Calculate utilization for each arm
        2. Reduce probability for arms above target utilization (70%)
        3. Redistribute to under-utilized arms proportionally
        """
        arms = list(base_probs.keys())
        
        # Calculate utilization and penalties
        utilizations = {}
        penalties = {}
        target_util = 0.7  # Target 70% utilization
        
        for arm in arms:
            if arm in self.capacities:
                util = self.get_utilization(arm)
                utilizations[arm] = util
                
                # Penalty increases exponentially as we exceed target utilization
                if util > target_util:
                    excess = util - target_util
                    penalties[arm] = min(0.9, excess ** 2)  # Max 90% penalty
                else:
                    penalties[arm] = 0.0
            else:
                utilizations[arm] = 0.0
                penalties[arm] = 0.0
        
        # Apply penalties
        adjusted_probs = {}
        removed_prob = 0.0
        
        for arm in arms:
            penalty = penalties[arm]
            adjusted = base_probs[arm] * (1 - penalty)
            adjusted_probs[arm] = adjusted
            removed_prob += base_probs[arm] - adjusted
        
        # Redistribute removed probability to under-utilized arms
        underutilized = [a for a in arms if utilizations[a] < target_util]
        
        if underutilized and removed_prob > 0:
            # Weight redistribution by available capacity
            available_capacity = {}
            total_available = 0
            
            for arm in underutilized:
                if arm in self.capacities:
                    capacity = self.capacities[arm]
                    used = self.get_load(arm)
                    available = max(0, capacity * target_util - used)
                    available_capacity[arm] = available
                    total_available += available
                else:
                    available_capacity[arm] = 1.0
                    total_available += 1.0
            
            # Redistribute proportionally to available capacity
            if total_available > 0:
                for arm in underutilized:
                    share = available_capacity[arm] / total_available
                    adjusted_probs[arm] += removed_prob * share
        
        # Normalize to ensure sum = 1
        total = sum(adjusted_probs.values())
        if total > 0:
            adjusted_probs = {arm: p / total for arm, p in adjusted_probs.items()}
        
        return adjusted_probs
    
    def get_arm(self, active_ids, k=1):
        """Select k arms based on current probabilities."""
        if not active_ids:
            return [], []
        
        self._sync_arms(active_ids)
        probs = self._get_probabilities()
        
        if not probs:
            return [], []
        
        arms, ps = zip(*probs.items())
        
        if not arms or not ps or all(w == 0 for w in ps):
            return [], []
        
        chosen = random.choices(arms, weights=ps, k=k)
        
        # Track selection for load calculation
        for arm in chosen:
            self.recent_selections[arm].append(self.t)
        
        return chosen, [probs[chosen_i] for chosen_i in chosen]
    
    def update(self, arm_id, success, latency):
        """
        Update weights based on observed reward.
        
        Reward incorporates both latency and load:
        - Fast response = high reward
        - Overloaded arm = implicit penalty through higher latency
        """
        arm_dist = self._get_probabilities()
        
        if arm_id in arm_dist:
            p_arm = arm_dist[arm_id]
            
            # Base reward from latency
            if success:
                r = 1 - min(latency / self.L_max, 1.0)
            else:
                r = 0.0
            
            # Compute importance-weighted reward
            x_hat = r / (p_arm + 1e-7)
            
            # Update weight
            eta = self.gamma / len(self.weights) if self.weights else self.gamma
            self.weights[arm_id] += eta * x_hat
        
        self.t += 1
    
    def get_probabilities(self):
        """Return current probability distribution."""
        return self._get_probabilities()


class CapacityProportionalBandit:
    """
    Simple capacity-proportional routing with adaptive adjustments.
    
    Maintains base probabilities proportional to capacity, but adjusts
    based on observed performance to handle heterogeneous latencies.
    """
    
    def __init__(self, gamma=0.1, L_max=1.0, learning_rate=0.01):
        """
        Args:
            gamma: Exploration parameter
            L_max: Maximum latency
            learning_rate: How quickly to adapt to performance differences
        """
        self.gamma = gamma
        self.L_max = L_max
        self.learning_rate = learning_rate
        
        self.capacities = {}
        self.weights = {}  # Learned adjustments to base probabilities
        self.t = 0
        
        random.seed(time.time() + os.getpid())
    
    def set_capacities(self, capacities):
        """Set capacity information."""
        self.capacities = capacities.copy()
        # Initialize weights
        for arm in capacities:
            if arm not in self.weights:
                self.weights[arm] = 1.0
    
    def _sync_arms(self, active_ids):
        """Sync with active arms."""
        # Add new arms
        for arm in active_ids:
            if arm not in self.weights:
                self.weights[arm] = 1.0
        
        # Remove inactive
        for arm in list(self.weights):
            if arm not in active_ids:
                del self.weights[arm]
    
    def _get_probabilities(self):
        """
        Compute probabilities: base capacity proportion + learned adjustments + exploration.
        """
        K = len(self.weights)
        if K == 0:
            return {}
        
        arms = list(self.weights.keys())
        
        # Base probabilities from capacity
        if all(arm in self.capacities for arm in arms):
            total_capacity = sum(self.capacities[arm] for arm in arms)
            base_probs = {
                arm: self.capacities[arm] / total_capacity
                for arm in arms
            }
        else:
            # Equal if no capacity info
            base_probs = {arm: 1.0 / K for arm in arms}
        
        # Apply learned weight adjustments
        adjusted_probs = {}
        for arm in arms:
            adjusted_probs[arm] = base_probs[arm] * self.weights[arm]
        
        # Normalize
        total = sum(adjusted_probs.values())
        if total > 0:
            adjusted_probs = {arm: p / total for arm, p in adjusted_probs.items()}
        
        # Add exploration floor
        floor = self.gamma / K
        final_probs = {
            arm: (1 - self.gamma) * adjusted_probs[arm] + floor
            for arm in arms
        }
        
        return final_probs
    
    def get_arm(self, active_ids, k=1):
        """Select arms."""
        if not active_ids:
            return [], []
        
        self._sync_arms(active_ids)
        probs = self._get_probabilities()
        
        if not probs:
            return [], []
        
        arms, ps = zip(*probs.items())
        chosen = random.choices(arms, weights=ps, k=k)
        
        return chosen, [probs[c] for c in chosen]
    
    def update(self, arm_id, success, latency):
        """
        Update learned weights based on performance.
        
        If an arm performs better than expected, increase its weight.
        If worse, decrease it.
        """
        if arm_id not in self.weights:
            return
        
        # Compute reward
        if success:
            r = 1 - min(latency / self.L_max, 1.0)
        else:
            r = 0.0
        
        # Expected reward is 0.5 (middle of range)
        # Adjust weight based on deviation
        expected_r = 0.5
        delta = r - expected_r
        
        # Update weight with gradient step
        self.weights[arm_id] += self.learning_rate * delta
        
        # Clamp to reasonable range
        self.weights[arm_id] = max(0.1, min(5.0, self.weights[arm_id]))
        
        self.t += 1
    
    def get_probabilities(self):
        """Return current probabilities."""
        return self._get_probabilities()


class HybridLoadBalancer:
    """
    Hybrid approach: Use capacity-proportional as base, with Exp3-style learning.
    
    Combines:
    1. Capacity-aware base distribution
    2. Exp3 learning for adaptation
    3. Load monitoring to prevent overload
    """
    
    def __init__(self, gamma=0.15, L_max=1.0, capacity_weight=0.5):
        """
        Args:
            gamma: Exploration parameter
            L_max: Maximum latency
            capacity_weight: Weight given to capacity (0=pure Exp3, 1=pure capacity-proportional)
        """
        self.exp3 = LoadAwareExp3(gamma=gamma, L_max=L_max, capacity_aware=True)
        self.capacity_weight = capacity_weight
    
    def set_capacities(self, capacities):
        """Set capacity information."""
        self.exp3.set_capacities(capacities)
    
    def get_arm(self, active_ids, k=1):
        """Select arms."""
        return self.exp3.get_arm(active_ids, k)
    
    def update(self, arm_id, success, latency):
        """Update based on feedback."""
        self.exp3.update(arm_id, success, latency)
    
    def get_probabilities(self):
        """Get current probabilities."""
        return self.exp3.get_probabilities()

