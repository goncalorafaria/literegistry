# Bandit Algorithms for LiteRegistry

Complete documentation for bandit-based routing algorithms in LiteRegistry.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [What Was Fixed](#what-was-fixed)
3. [Algorithms Available](#algorithms-available)
4. [Parameter Tuning](#parameter-tuning)
5. [Testing & Validation](#testing--validation)
6. [Files Reference](#files-reference)

---

## Quick Start

### Basic Usage - Minimize Latency

```python
from literegistry.bandit import Exp3Dynamic

# Initialize bandit
bandit = Exp3Dynamic(
    gamma=0.15,      # Exploration parameter (0.10-0.20 for most cases)
    L_max=2.0,       # Maximum expected latency
    init_weight=1.0
)

# Your arms (models, servers, endpoints, etc.)
arms = ['model_a', 'model_b', 'model_c']

# In request handler:
def handle_request(request):
    # 1. Select arm
    chosen, prob = bandit.get_arm(arms, k=1)
    arm = chosen[0]
    
    # 2. Route request
    start = time.time()
    success, response = route_to(arm, request)
    latency = time.time() - start
    
    # 3. Update bandit
    bandit.update(arm, success, latency)
    
    return response
```

### Load Balancing with Capacity Constraints

```python
from literegistry.bandit_loadaware import LoadAwareExp3

# Initialize with capacity awareness
bandit = LoadAwareExp3(
    gamma=0.20,           # Higher exploration for load balancing
    L_max=2.0,
    load_window=100,
    capacity_aware=True
)

# Set arm capacities
bandit.set_capacities({
    'server_1': 100,  # requests/sec
    'server_2': 200,
    'server_3': 150,
})

# Use same pattern as above
chosen, prob = bandit.get_arm(arms, k=1)
# ... route and update ...
```

### Find Optimal Parameters

```python
from literegistry.bandit_tuning import analyze_arm_characteristics

# Collect latency samples from each arm
arm_data = {
    'arm1': {'latencies': [...], 'capacity': 100},
    'arm2': {'latencies': [...], 'capacity': 200},
}

# Get recommendations
recommendations = analyze_arm_characteristics(arm_data)

# Use recommended parameters
bandit = Exp3Dynamic(
    gamma=recommendations['recommended_gamma'],
    L_max=recommendations['recommended_L_max']
)
```

---

## What Was Fixed

### 🐛 Bug #1: Weight Corruption (CRITICAL)

**Location**: `bandit.py:58` in `_get_probabilities()`

**Problem**: The method permanently modified `self.weights` during probability calculation, destroying accumulated learning.

```python
# BEFORE (WRONG):
self.weights = {a: (lw - max_log) for a, lw in self.weights.items()}

# AFTER (FIXED):
shifted_weights = {a: (lw - max_log) for a, lw in self.weights.items()}
```

**Impact**: Weights couldn't accumulate properly, breaking the learning mechanism entirely.

### 🐛 Bug #2: Missing Exploration (CRITICAL)

**Location**: `bandit.py:65` in `_get_probabilities()`

**Problem**: Exploration floor (gamma term) was commented out, causing 100% selection of highest-weighted arm.

```python
# BEFORE (WRONG):
arm: (w / (total_w + 1e-4))  # +(1 - self.gamma) * + floor

# AFTER (FIXED):
arm: (1 - self.gamma) * (w / (total_w + 1e-9)) + floor
```

**Impact**: Once an arm gained a lead, it monopolized all selections. No continued exploration or adaptation.

### ✅ Verification

After fixes:
- ✅ UniformBandit: Truly uniform distribution (chi-square: 1-9)
- ✅ Exp3Dynamic: Correctly converges to optimal arm (82-92% selection)
- ✅ Load balancing: Now explores properly (not 100% one arm)

---

## Algorithms Available

### 1. UniformBandit

**Use When**: Equal distribution needed, A/B testing, fairness requirements

```python
from literegistry.bandit import UniformBandit

bandit = UniformBandit()
```

**Pros**: Simple, truly uniform, no tuning needed  
**Cons**: Doesn't learn or adapt

---

### 2. Exp3Dynamic (Fixed!)

**Use When**: Minimize latency, arms independent (no congestion)

```python
from literegistry.bandit import Exp3Dynamic

bandit = Exp3Dynamic(gamma=0.15, L_max=2.0)
```

**Parameters**:
- `gamma`: Exploration (0.10-0.20 recommended)
- `L_max`: Max latency for reward normalization (set to P99 * 1.1)
- `init_weight`: Initial weight for new arms (default: 1.0)

**Pros**: Learns to prefer faster arms, theoretically grounded  
**Cons**: Can overload if congestion exists

**Best for**: Routing between different model sizes, clouds, providers

---

### 3. LoadAwareExp3 (NEW!)

**Use When**: Congestion matters, have capacity information

```python
from literegistry.bandit_loadaware import LoadAwareExp3

bandit = LoadAwareExp3(
    gamma=0.20,
    L_max=2.0,
    load_window=100,
    capacity_aware=True
)

bandit.set_capacities({'arm1': 100, 'arm2': 200})
```

**Parameters**:
- `gamma`: 0.15-0.25 (higher for load balancing)
- `L_max`: Max latency (P99 * 1.1)
- `load_window`: Requests tracked for load (default: 100)
- `capacity_aware`: Enable capacity-based penalties (default: True)

**Key Features**:
- Tracks recent load per arm
- Penalizes overloaded arms
- Redistributes to under-utilized arms
- Capacity-proportional adjustment

**Best for**: Load balancing servers, managing capacity constraints

---

### 4. CapacityProportionalBandit (NEW!)

**Use When**: Capacity is primary concern, simpler than LoadAwareExp3

```python
from literegistry.bandit_loadaware import CapacityProportionalBandit

bandit = CapacityProportionalBandit(gamma=0.10, L_max=2.0)
bandit.set_capacities({'arm1': 100, 'arm2': 200})
```

**Best for**: Simple capacity-based routing with minor performance adjustments

---

## Parameter Tuning

### Quick Reference

See **[PARAMETER_CHEATSHEET.md](PARAMETER_CHEATSHEET.md)** for copy-paste code and quick decisions.

### Full Guide

Run the interactive tuning guide:

```bash
python3 literegistry/bandit_tuning.py
```

Or see [example_parameter_tuning.py](example_parameter_tuning.py) for complete workflow.

### Key Parameters Explained

#### Gamma (γ) - Exploration Parameter

| Value | Selection of Best | Use Case |
|-------|------------------|----------|
| 0.05-0.10 | 90%+ | Clear differences, stable |
| **0.10-0.20** | **80-90%** | **Most use cases (DEFAULT)** |
| 0.20-0.30 | 70-80% | Similar arms, load balancing |
| 0.30+ | 60-70% | High diversity needed |

**Rule**: Each arm gets minimum `γ/K` probability

#### L_max - Maximum Latency

**How to set**:
```python
import numpy as np

# Collect samples
latencies = collect_all_arm_latencies()

# Use P99 + buffer
L_max = np.percentile(latencies, 99) * 1.1
```

**Too low**: Rewards compressed, poor differentiation  
**Too high**: All latencies look equally good  
**Just right**: P99 of slowest arm + 10-20%

### Parameter Analysis Tool

```python
from literegistry.bandit_tuning import analyze_arm_characteristics

arm_data = {
    'arm1': {'latencies': [...], 'capacity': 100},
    'arm2': {'latencies': [...], 'capacity': 200},
}

# Automatically recommends gamma, L_max, and algorithm
recommendations = analyze_arm_characteristics(arm_data)
```

---

## Testing & Validation

### Test Files

All tests generate detailed visualizations and statistics:

#### 1. **test_uniform_bandit.py**
Tests UniformBandit distribution quality.

```bash
python3 test_uniform_bandit.py
```

**Generates**: `uniform_bandit_test.png`

**Validates**: Uniformity, chi-square test, distribution quality

---

#### 2. **test_uniform_patterns.py**
Advanced pattern analysis (autocorrelation, transitions, runs).

```bash
python3 test_uniform_patterns.py
```

**Generates**: `uniform_bandit_transitions.png`

**Validates**: Sequential patterns, autocorrelation, transition probabilities

---

#### 3. **test_exp3_bandit.py**
Tests Exp3Dynamic for latency minimization (no congestion).

```bash
python3 test_exp3_bandit.py
```

**Generates**:
- `exp3_scenario1_clear_winner.png`
- `exp3_scenario2_close_competition.png`
- `exp3_scenario3_many_arms.png`
- `exp3_gamma_comparison.png`

**Validates**: Convergence to optimal, gamma effects, learning behavior

---

#### 4. **test_exp3_load_balancing.py**
Tests with realistic congestion (latency increases under load).

```bash
python3 test_exp3_load_balancing.py
```

**Generates**:
- `exp3_balanced_arms.png`
- `exp3_heterogeneous_capacity.png`
- `exp3_many_arms_congestion.png`

**Validates**: Load balancing, capacity awareness, utilization

---

### Test Results Summary

See **[BANDIT_TEST_SUMMARY.md](BANDIT_TEST_SUMMARY.md)** for complete analysis.

**Key Results**:
- ✅ UniformBandit: Perfect uniformity
- ✅ Exp3Dynamic: 82-92% convergence to optimal (without congestion)
- ⚠️ Exp3Dynamic: Struggles with load balancing (140-240% worse than optimal)
- ✅ LoadAwareExp3: Designed to address congestion (use this for load balancing)

### Gateway Load-Balancing Simulation

For the serving gateway, also run:

```bash
python3 examples/compare_load_balancers.py --scenario two-groups --requests 5000 --arrival-rate 120 --seed 11 --gamma 0.2 --l-max 60 --penalty-latency 60
```

This simulation models four replicas with queueing, finite capacity, failures, and long-tail request latency. The two-group scenario uses:

- `fast_0`, `fast_1`: 250 ms mean latency, capacity 32
- `slow_0`, `slow_1`: 417 ms mean latency, capacity 19, i.e. about 60% of fast-server speed
- All replicas: 4% long-tail requests, 1% failures

Representative result at 120 requests/sec:

| Router | Avg Latency | P95 Latency | Failure Rate | Notes |
|--------|-------------|-------------|--------------|-------|
| `least_inflight` | 0.955s | 2.93s | 1.1% | Best robust baseline; slow replicas self-throttle via higher in-flight count |
| `exp3`, gamma=0.2 | 0.969s | 3.17s | 1.1% | Close to least-inflight; stable under heterogeneity and tails |
| `exp3`, gamma=0.4 | 0.962s | 3.17s | 1.1% | Similar, slightly more exploratory |
| `round_robin` | 0.990s | 3.40s | 1.1% | Good when deterministic spread prevents bursty overload |
| `epsilon_ewma` | 1.319s | 4.60s | 1.1% | Better than greedy EWMA, still too reactive |
| `ewma_latency` | 5.015s | 17.7s | 1.1% | Greedy latency chasing can clog replicas after tail events |
| `random` | 57.4s | 720s | 1.1% | Bursty assignment can overload a slow replica badly |
| `oracle_server_0` | 2126s | 4168s | 1.1% | Always choosing one fast server overloads it |

The homogeneous scenario, where all four servers have the same capacity and the only nuance is long-tail requests, showed that stable spread is usually better than aggressive latency chasing. Random, round-robin, least-inflight, and Exp3 were close, while pure EWMA-style routers could get trapped by tail noise and overload one replica.

Practical takeaway: for the current gateway, the conservative production defaults are `gamma=0.2`, `L_max=penalty_latency`, and `penalty_latency=60.0`. Do not renormalize Exp3 weights on every probability read; keep the numerically stable log-weight representation and only renormalize to prevent unbounded log-weight drift.

---

## Files Reference

### Core Implementation
- `literegistry/bandit.py` - Base implementations (UniformBandit, Exp3Dynamic) **[FIXED]**
- `literegistry/bandit_loadaware.py` - Load-aware algorithms (LoadAwareExp3, CapacityProportionalBandit) **[NEW]**
- `literegistry/bandit_tuning.py` - Parameter tuning utilities **[NEW]**

### Documentation
- `README_BANDITS.md` - This file (complete guide)
- `PARAMETER_CHEATSHEET.md` - Quick reference for parameters and algorithms
- `BANDIT_TEST_SUMMARY.md` - Detailed test results and analysis

### Examples
- `example_parameter_tuning.py` - Complete workflow examples **[NEW]**
- `examples/compare_load_balancers.py` - Queueing simulation for random, round-robin, least-inflight, EWMA, epsilon-EWMA, and Exp3 routers

### Tests
- `test_uniform_bandit.py` - UniformBandit distribution tests
- `test_uniform_patterns.py` - Advanced uniformity analysis
- `test_exp3_bandit.py` - Exp3 learning tests (no congestion)
- `test_exp3_load_balancing.py` - Exp3 with congestion simulation

---

## Recommended Workflow

### 1. Initial Setup (First Time)

```bash
# Run the tuning guide
python3 literegistry/bandit_tuning.py

# Read the cheat sheet
cat PARAMETER_CHEATSHEET.md
```

### 2. Collect Data

Run uniform routing for 1000-5000 requests to collect baseline latencies:

```python
from literegistry.bandit import UniformBandit

bandit = UniformBandit()
# ... collect latency data for each arm ...
```

### 3. Analyze & Configure

```python
from literegistry.bandit_tuning import analyze_arm_characteristics

arm_data = {
    'arm1': {'latencies': collected_latencies_1, 'capacity': 100},
    'arm2': {'latencies': collected_latencies_2, 'capacity': 200},
}

recommendations = analyze_arm_characteristics(arm_data)

# Use recommended algorithm and parameters
```

### 4. Deploy with Monitoring

```python
# Log these metrics
metrics = {
    'probabilities': bandit.get_probabilities(),
    'recent_avg_latency': rolling_avg(latencies, window=1000),
    'selection_counts': Counter(recent_selections),
}

# Alert if:
if max(metrics['probabilities'].values()) > 0.95:
    alert("Too concentrated - increase gamma")
```

### 5. Adjust Based on Production

Monitor for ~1000-5000 requests, then adjust:

```python
# If too concentrated (>95% one arm)
bandit.gamma *= 1.5

# If too spread out (near uniform)
bandit.gamma *= 0.7

# If latency increasing
# Switch to LoadAwareExp3 with capacity info
```

---

## Monitoring Checklist

Track these in production:

- [ ] **Selection distribution** - `bandit.get_probabilities()`
- [ ] **Average latency** - Rolling window of 1000
- [ ] **P95/P99 latencies** - Watch tail latency
- [ ] **Per-arm utilization** - If using LoadAwareExp3
- [ ] **Regret** - Cumulative vs optimal

**Warning Signs**:
- One arm >95% → Increase gamma
- All arms equal → Decrease gamma or check L_max
- Latency increasing → Check for overload, use LoadAwareExp3

---

## Support & Troubleshooting

### Common Issues

**Q: Bandit not learning / all arms get equal probability**
- Check L_max (might be too high)
- Check gamma (might be too high)
- Verify arms actually have different latencies

**Q: One arm getting 100% of traffic**
- Fixed in current version! Update to latest bandit.py
- If still happening, increase gamma

**Q: Latency worse with bandit than uniform**
- You likely have congestion - use LoadAwareExp3
- Set capacities with `bandit.set_capacities({...})`

**Q: How to handle dynamic arms (coming/going)?**
- Exp3Dynamic handles this automatically
- New arms initialize to average weight
- Removed arms are cleaned up

### Getting Help

1. Check the cheat sheet: [PARAMETER_CHEATSHEET.md](PARAMETER_CHEATSHEET.md)
2. Run the tuning guide: `python3 literegistry/bandit_tuning.py`
3. See examples: [example_parameter_tuning.py](example_parameter_tuning.py)
4. Review test results: [BANDIT_TEST_SUMMARY.md](BANDIT_TEST_SUMMARY.md)

---

## Summary

✅ **Fixed critical bugs** in Exp3Dynamic  
✅ **Created load-aware algorithms** for congestion handling  
✅ **Built parameter tuning tools** for optimal configuration  
✅ **Comprehensive testing** with visualizations  
✅ **Complete documentation** with examples  

**Start here**: [PARAMETER_CHEATSHEET.md](PARAMETER_CHEATSHEET.md) for quick reference  
**Go deep**: Run `python3 literegistry/bandit_tuning.py` for full guide
