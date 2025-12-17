import math
import random
import time, os
import logging 


class Exp3Dynamic:
    """
    Exp3 with dynamic arms. On each select, you pass the current list of active IDs;
    the router will add new arms (initializing their weight to the average of existing
    weights) and remove any that are no longer active.
    
    Parameters:
        gamma (float): Exploration parameter in [0, 1]. Controls exploration vs exploitation.
            Lower = more exploitation (faster convergence but may miss good arms).
            Higher = more exploration (better for non-stationary or changing environments).
            
            Practical guidelines:
            - Few arms (2-5): 0.1-0.2
            - Medium arms (5-10): 0.15-0.25  
            - Many arms (>10): 0.2-0.3
            - Non-stationary/dynamic arms: 0.2-0.4
            - Static environment, long runs: 0.05-0.15
            - Default 0.2 works well for most cases
            
        L_max (float): Maximum expected latency for reward normalization.
            Used to normalize rewards: reward = 1 - min(latency/L_max, 1.0) for successes.
            
            How to choose:
            - Should be >= worst-case latency you expect to see
            - Good default: 95th percentile of historical latencies
            - Rule of thumb: 2-3x your typical latency
            - Too small (< typical latency): fast arms get same reward (less differentiation)
            - Too large (>> typical latency): all arms get similar rewards (slower learning)
            - Example: if latencies are 0.1-0.5s, use L_max=1.0-1.5s
            
        init_weight (float): Initial weight for new arms (converted to log(init_weight) internally).
            - Default 1.0 is standard (becomes log(1.0) = 0.0 in log space)
            - All arms start equal, so this parameter has minimal practical effect
            - Keep at 1.0 unless you have specific reasons to change it
            
    Example:
        # For a service with typical latency 0.2-0.8s, expecting 5-10 endpoints
        bandit = Exp3Dynamic(gamma=0.2, L_max=1.5, init_weight=1.0)
        
        # For highly dynamic environment with many changing endpoints
        bandit = Exp3Dynamic(gamma=0.3, L_max=2.0)
        
        # For stable environment with few arms, prioritize exploitation
        bandit = Exp3Dynamic(gamma=0.1, L_max=1.0)
    """

    def __init__(self, gamma=0.2, L_max=1.0, init_weight=1.0):
        self.gamma = gamma
        self.L_max = L_max
        self.init_weight = init_weight
        self.weights = {}  # arm_id -> log_weight (stored in log space for numerical stability)
        self.t = 0  # rounds elapsed
        random.seed(time.time() + os.getpid())

    def _eta(self):
        K = len(self.weights)
        return (self.gamma / K) if K > 0 else self.gamma

    def _sync_arms(self, active_ids):
        # Handle empty active_ids
        if not active_ids:
            self.weights.clear()
            return
            
        # add any new arms
        # Note: We store log weights, so initialize new arms to log(init_weight)
        # For init_weight=1.0, this means log(1.0) = 0.0
        init_log_w = math.log(self.init_weight) if self.init_weight > 0 else -1
        
        if not self.weights:
            for arm in active_ids:
                self.weights[arm] = init_log_w
        else:
            # Average of existing log weights to initialize new arms
            avg_log_w = sum(self.weights.values()) / len(self.weights)
            for arm in active_ids:
                if arm not in self.weights:
                    self.weights[arm] = avg_log_w

        # remove any that have gone offline
        for arm in list(self.weights):
            if arm not in active_ids:
                del self.weights[arm]

    def _get_probabilities(self):
        K = len(self.weights)
        if K == 0:
            return {}
        else:

            max_log = max(self.weights.values())
            # unnormalized "weights" shifted by max to avoid overflow
            # Use temporary variable to avoid destroying self.weights!
            shifted_weights = {a: (lw - max_log) for a, lw in self.weights.items()}
            exp_weights = {a: math.exp(lw) for a, lw in shifted_weights.items()}
            total_w = sum(exp_weights.values())

            floor = self.gamma / K
            return {
                arm: (1 - self.gamma) * (w / (total_w + 1e-9)) + floor
                for arm, w in exp_weights.items()
            }

    def get_arm(self, active_ids, k=1):
        """
        Given the list of currently active arm IDs:
        - sync the weight dict (add/remove)
        - compute Exp3 probabilities
        - sample one arm
        Returns (chosen_id, p_chosen)
        """
        # Handle empty active_ids
        if not active_ids:
            return [], []
            
        self._sync_arms(active_ids)
        probs = self._get_probabilities()

        
        # Handle case when no probabilities are available
        if not probs:
            return [], []
            
        arms, ps = zip(*probs.items())

        # print("arms and probs", arms, ps)
        # Ensure we have valid arms and weights before calling random.choices
        if not arms or not ps or all(w == 0 for w in ps):
            return [], []
            
        chosen = random.choices(arms, weights=ps, k=k)
        return chosen, [probs[chosen_i] for chosen_i in chosen]

    def update(self, arm_id, success, latency, prob=1.0):
        """
        After routing to arm_id (with probability p_arm) and observing:
        - success: bool
          - latency: float (in same units as L_max)
        compute normalized reward in [0,1] and update weights.
        """
        arm_dist = self._get_probabilities()

        if arm_id in arm_dist:
          
            # normalized reward: fast success → near 1, slow or failure → near 0
            if success:
                r = - min(latency / self.L_max, 1.0)
            else:
                r = -1

            x_hat = r / (prob+ 1e-7)
            self.weights[arm_id] += self._eta() * x_hat

        self.t += 1

    def get_probabilities(self):
        """Return the current probability distribution (after syncing)."""
        return self._get_probabilities()

 
    
class UniformBandit(Exp3Dynamic):
    def __init__(self):
        # Use init_weight=1.0 so log weights initialize to log(1.0)=0.0
        # This ensures uniform distribution (all arms have equal log weights)
        super().__init__(gamma=0.2, L_max=1.0, init_weight=1.0)
        
    def update(self, arm_id, success, latency, prob=1.0):
        arm_dist = self._get_probabilities()

        if arm_id in arm_dist:
            
            self.weights[arm_id] = 0

        self.t += 1
        
        
        
if __name__ == "__main__":
    bandit = UniformBandit(["arm1", "arm2", "arm3"])
    active_arms = ["arm1", "arm2", "arm3"]

    for _ in range(10):
        chosen_arm, prob = bandit.get_arm(active_arms)
        print(f"Chosen arm: {chosen_arm}, Probability: {prob}")
    
    
