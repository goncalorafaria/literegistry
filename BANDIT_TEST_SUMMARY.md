# Bandit Algorithm Testing Summary

## Overview
Comprehensive testing of the UniformBandit and Exp3Dynamic bandit implementations in `literegistry/bandit.py`.

## Tests Created

### 1. `test_uniform_bandit.py`
Tests the UniformBandit distribution uniformity.

**Results**: ✅ **PASSED**
- Distribution is properly uniform across all arms
- Chi-square statistics: 1.05-9.38 (low values indicate good uniformity)
- No sequential patterns or autocorrelation detected
- Handles dynamic arm addition/removal correctly

**Example Results (5 arms, 10,000 samples):**
```
arm0: 20.13%  (expected: 20.00%)
arm1: 19.22%  (expected: 20.00%)
arm2: 20.95%  (expected: 20.00%)
arm3: 20.24%  (expected: 20.00%)
arm4: 19.46%  (expected: 20.00%)
```

### 2. `test_uniform_patterns.py`
Advanced pattern analysis for UniformBandit.

**Results**: ✅ **PASSED**
- Transition probabilities are uniform (all within ±2% of expected)
- Autocorrelation near zero at all lags (-0.016 to +0.016)
- Run length distribution matches theoretical expectations
- No hidden biases or patterns detected

### 3. `test_exp3_bandit.py`
Tests Exp3Dynamic for latency minimization (without congestion).

**Results**: ✅ **PASSED** (after bug fixes)

**Bugs Fixed:**
1. **Critical Bug #1**: Line 58 was permanently modifying `self.weights` during probability calculation
   - **Before**: `self.weights = {a: (lw - max_log) ...}`
   - **After**: `shifted_weights = {a: (lw - max_log) ...}`
   
2. **Critical Bug #2**: Line 65 had the exploration term commented out
   - **Before**: `arm: (w / (total_w + 1e-4))  # +(1 - self.gamma) * + floor`
   - **After**: `arm: (1 - self.gamma) * (w / (total_w + 1e-9)) + floor`

**Post-Fix Results:**
- **Scenario 1** (Clear winner): 92% selection of optimal arm ✓
- **Scenario 2** (Close competition): 88% selection of optimal arm ✓
- **Scenario 3** (Many arms): 83% selection of optimal arm ✓

The algorithm correctly balances exploitation (favoring fast arms) with exploration (gamma parameter).

### 4. `test_exp3_load_balancing.py`
Tests Exp3Dynamic with realistic congestion/capacity constraints.

**Results**: ⚠️ **PARTIAL** - Algorithm works but suboptimal for load balancing

**Key Finding**: Standard Exp3 is not well-suited for load balancing with congestion because:

1. **Confounded Feedback**: The bandit only learns from its own experience. When it overloads an arm, it sees high latency but doesn't understand that reducing load would improve latency.

2. **No Load Awareness**: The algorithm treats arms independently without understanding that latency is a function of current load.

3. **Exploration-Exploitation Mismatch**: For load balancing, you want sustained diversity, not convergence to a single "best" arm.

**Performance with Congestion:**
```
Scenario 1 (Balanced capacity):
  - Exp3 recent latency: 1.032
  - Theoretical optimal: 0.396
  - Difference: +160% worse

Scenario 2 (Heterogeneous capacity):
  - Exp3 recent latency: 1.055
  - Theoretical optimal: 0.433
  - Difference: +143% worse

Scenario 3 (Many arms):
  - Exp3 recent latency: 1.226
  - Theoretical optimal: 0.354
  - Difference: +246% worse
```

The algorithm still heavily favors one arm (80-90% probability) even when that arm is overloaded and showing high latency.

### 5. `examples/compare_load_balancers.py`
Simulates gateway routing with finite-capacity model servers, queueing, failures, and long-tail latency.

**Results**: ✅ **USEFUL FOR DEFAULTS** - conservative Exp3 and least-inflight are robust; greedy latency routing is risky.

**Scenario A: Homogeneous replicas with long tails**
- Four identical servers
- Mean latency: 250 ms
- Capacity: 32 concurrent requests per server
- Long-tail requests: 4% probability of multi-second latency
- Failure rate: 1%

In this setup, there is no true fast/slow server distinction. The only difficulty is that rare slow requests temporarily clog individual servers. Stable spread was best: random, round-robin, least-inflight, and Exp3 were close. Greedy EWMA-style routers could overreact to tail noise and concentrate traffic poorly.

**Scenario B: Two fast replicas, two 60%-speed replicas**
- `fast_0`, `fast_1`: 250 ms mean latency, capacity 32
- `slow_0`, `slow_1`: 417 ms mean latency, capacity 19
- All replicas: 4% long-tail requests and 1% failures
- Arrival rate: 120 requests/sec
- Requests: 5000
- Seed: 11
- Exp3: `gamma=0.2`, `L_max=60`, `penalty_latency=60`

Representative results:

```
least_inflight   avg=0.955s   p95=2.93s    failures=1.1%
exp3 gamma=0.2   avg=0.969s   p95=3.17s    failures=1.1%
exp3 gamma=0.4   avg=0.962s   p95=3.17s    failures=1.1%
round_robin      avg=0.990s   p95=3.40s    failures=1.1%
epsilon_ewma     avg=1.319s   p95=4.60s    failures=1.1%
ewma_latency     avg=5.015s   p95=17.7s    failures=1.1%
random           avg=57.4s    p95=720s     failures=1.1%
oracle_server_0  avg=2126s    p95=4168s    failures=1.1%
```

**Key finding**: load awareness helps, but pure latency chasing is unsafe under long-tail latency. `least_inflight` works well because slower servers hold requests longer and therefore naturally receive less new traffic. Exp3 with `gamma=0.2` stayed close to this baseline without the extra load-aware hyperparameters.

## Summary of Bugs Fixed

### Bug 1: Weight Corruption
**Location**: `bandit.py:58` in `_get_probabilities()`

**Problem**: The method was permanently modifying `self.weights` every time probabilities were calculated. This destroyed the accumulated learning signal.

**Impact**: Severe - weights could not accumulate properly, breaking the core learning mechanism.

**Fix**:
```python
# Before (WRONG):
self.weights = {a: (lw - max_log) for a, lw in self.weights.items()}

# After (CORRECT):
shifted_weights = {a: (lw - max_log) for a, lw in self.weights.items()}
```

### Bug 2: Missing Exploration
**Location**: `bandit.py:65` in `_get_probabilities()`

**Problem**: The exploration floor (gamma term) was commented out, causing the algorithm to converge to 100% selection of the highest-weighted arm with no continued exploration.

**Impact**: Critical - once an arm gained a lead, it would monopolize all selections, preventing discovery of better options or adaptation to changing conditions.

**Fix**:
```python
# Before (WRONG):
arm: (w / (total_w + 1e-4))  # +(1 - self.gamma) * + floor

# After (CORRECT):
arm: (1 - self.gamma) * (w / (total_w + 1e-9)) + floor
```

This ensures each arm receives at least `gamma/K` probability for exploration, which is essential for the Exp3 algorithm.

## Recommendations

### For Uniform Load Balancing
✅ **Use UniformBandit** - works perfectly for equal distribution across arms.

### For Latency Minimization (No Congestion)
✅ **Use Exp3Dynamic** - effectively learns to prefer faster arms when latency is independent of load.

**Recommended parameters:**
- `gamma=0.1` to `0.2` (balance exploration vs exploitation)
- `L_max` = realistic maximum latency for your system
- Lower gamma = faster convergence but less exploration
- Higher gamma = more exploration, slower convergence

### For Load Balancing with Congestion
⚠️ **Exp3Dynamic has limitations**

The gateway simulation suggests `least_inflight` and conservative Exp3 are safer defaults than greedy EWMA latency routing. Consider alternative approaches:
1. **Capacity-aware weighted round-robin**: Distribute load proportional to capacity
2. **Least-connections**: Route to arm with fewest active requests
3. **Power of two choices**: Sample 2 random arms, pick less loaded one
4. **Gradient-based bandits**: Use algorithms that estimate latency as a function of load
5. **UCB with load awareness**: Incorporate load metrics into upper confidence bounds

### For Distributed/Multi-Agent Settings
If multiple gateway instances are using Exp3 independently, consider:
1. **Central coordination**: Share load statistics across instances
2. **Randomized selection**: Add more exploration to reduce herding
3. **Sticky sessions**: Route related requests to same arm to reduce exploration cost
4. **Capacity reservations**: Pre-allocate quota to different gateway instances

## Visualization Files Generated

All tests generate detailed visualizations:

### UniformBandit
- `uniform_bandit_test.png` - Distribution histograms
- `uniform_bandit_transitions.png` - Transition matrix heatmap

### Exp3Dynamic (Simple)
- `exp3_scenario1_clear_winner.png`
- `exp3_scenario2_close_competition.png`
- `exp3_scenario3_many_arms.png`
- `exp3_gamma_comparison.png`

### Exp3Dynamic (Load Balancing)
- `exp3_balanced_arms.png`
- `exp3_heterogeneous_capacity.png`
- `exp3_many_arms_congestion.png`

Each visualization includes:
- Probability evolution over time
- Cumulative selection counts
- Latency performance
- Load/utilization metrics (for congestion tests)
- Distribution analysis

## Usage Examples

### UniformBandit
```python
from literegistry.bandit import UniformBandit

bandit = UniformBandit()
active_arms = ["model_a", "model_b", "model_c"]

# Select arm
chosen, prob = bandit.get_arm(active_arms, k=1)

# Update (keeps weights at 0 for uniform distribution)
bandit.update(chosen[0], success=True, latency=0.5)
```

### Exp3Dynamic
```python
from literegistry.bandit import Exp3Dynamic

# Initialize with exploration parameter and max latency
bandit = Exp3Dynamic(gamma=0.15, L_max=2.0, init_weight=1.0)
active_arms = ["fast_model", "accurate_model", "cheap_model"]

# Select arm
chosen, prob = bandit.get_arm(active_arms, k=1)

# Update with performance feedback
success = True  # Request succeeded
latency = 0.234  # Observed latency in seconds
bandit.update(chosen[0], success, latency)

# Get current probabilities
probs = bandit.get_probabilities()
print(probs)  # {'fast_model': 0.75, 'accurate_model': 0.15, 'cheap_model': 0.10}
```

## Running the Tests

```bash
# Test UniformBandit
python3 test_uniform_bandit.py
python3 test_uniform_patterns.py

# Test Exp3Dynamic
python3 test_exp3_bandit.py
python3 test_exp3_load_balancing.py

# Run gateway queueing simulation
python3 examples/compare_load_balancers.py --scenario two-groups --requests 5000 --arrival-rate 120 --seed 11 --gamma 0.2 --l-max 60 --penalty-latency 60
```

## Conclusion

The bandit implementations are now working correctly after fixing two critical bugs:
1. ✅ **UniformBandit**: Provides truly uniform distribution
2. ✅ **Exp3Dynamic**: Effectively learns to minimize latency in simple scenarios
3. ⚠️ **Load Balancing**: Exp3 has fundamental limitations for congestion-aware load balancing

For production use with capacity constraints and congestion, consider implementing additional load-aware algorithms or using Exp3 with additional heuristics (e.g., capacity-based priors, load thresholds, or hybrid approaches).
