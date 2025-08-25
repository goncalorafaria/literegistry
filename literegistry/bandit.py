import math
import random
import time, os


class Exp3Dynamic:
    """
    Exp3 with dynamic arms. On each select, you pass the current list of active IDs;
    the router will add new arms (initializing their weight to the average of existing
    weights) and remove any that are no longer active.
    """

    def __init__(self, gamma=0.2, L_max=1.0, init_weight=1.0):
        self.gamma = gamma
        self.L_max = L_max
        self.init_weight = init_weight
        self.weights = {}  # arm_id -> weight
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
        if not self.weights:
            for arm in active_ids:
                self.weights[arm] = self.init_weight
        else:
            if len(self.weights) > 0:
                avg_w = sum(self.weights.values()) / len(self.weights)
            else:
                avg_w = 0

            for arm in active_ids:
                if arm not in self.weights:
                    self.weights[arm] = avg_w

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
            self.weights = {a: (lw - max_log) for a, lw in self.weights.items()}
            exp_weights = {a: math.exp(lw) for a, lw in self.weights.items()}
            total_w = sum(exp_weights.values())

            floor = self.gamma / K
            return {
                arm: (w / (total_w + 1e-4))  # +(1 - self.gamma) * +  floor
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

    def update(self, arm_id, success, latency):
        """
        After routing to arm_id (with probability p_arm) and observing:
        - success: bool
          - latency: float (in same units as L_max)
        compute normalized reward in [0,1] and update weights.
        """
        arm_dist = self._get_probabilities()

        if arm_id in arm_dist:
            p_arm = arm_dist[arm_id]
            # normalized reward: fast success → near 1, slow or failure → near 0
            if success:
                r = 1 - min(latency / self.L_max, 1.0)
            else:
                r = 0.0

            x_hat = r / (p_arm + 1e-7)
            self.weights[arm_id] += self._eta() * x_hat

            self.t += 1

    def get_probabilities(self):
        """Return the current probability distribution (after syncing)."""
        return self._get_probabilities()

 
    
class UniformBandit(Exp3Dynamic):
    def __init__(self):
        super().__init__(gamma=0.2, L_max=1.0, init_weight=0.0)
        
    def update(self, arm_id, success, latency):
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
    
    
