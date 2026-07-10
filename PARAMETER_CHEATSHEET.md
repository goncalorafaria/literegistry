# Bandit Parameter Cheat Sheet

## Quick Decision Tree

```
START HERE
    ↓
Do you have capacity/congestion constraints?
    ├─ NO → Use Exp3Dynamic
    │        • gamma = 0.10-0.20
    │        • L_max = P99 latency * 1.1
    │
    └─ YES → Do you know arm capacities?
             ├─ YES → Use LoadAwareExp3
             │         • gamma = 0.15-0.25
             │         • L_max = P99 latency * 1.1
             │         • load_window = 100
             │         • Set capacities!
             │
             └─ NO → Use Exp3Dynamic (but expect suboptimal load balance)
                      OR measure capacities first
```

## Parameter Quick Reference

### Gamma (γ) - Exploration Parameter

| Value | Behavior | Use When |
|-------|----------|----------|
| 0.05-0.10 | 90%+ on best arm | Arms clearly different, stable |
| **0.10-0.20** | **80-90% on best arm** | **Most use cases (DEFAULT)** |
| 0.20-0.30 | 70-80% on best arm | Similar arms, load balancing |
| 0.30+ | 60-70% on best arm | Need diversity, very similar arms |

**Formula**: Each arm gets minimum `γ/K` probability

**Example**: γ=0.15, K=5 arms → each arm gets ≥3%

### L_max - Maximum Latency

**How to set**:
1. Collect ~100-1000 latency samples from all arms
2. Calculate P99 percentile
3. Add 10-20% buffer

```python
import numpy as np
latencies = [...]  # all your latency samples
L_max = np.percentile(latencies, 99) * 1.1
```

**Common values**: 1.0-5.0 seconds (depends on your application)

### Load Window (LoadAwareExp3 only)

| Value | Behavior |
|-------|----------|
| 50 | Fast reaction, more noisy |
| **100** | **Balanced (DEFAULT)** |
| 200 | Smooth, slower reaction |

## Algorithm Selection Matrix

| Scenario | Algorithm | Key Parameters |
|----------|-----------|----------------|
| Minimize latency, no congestion | `Exp3Dynamic` | γ=0.10-0.20, L_max=P99 |
| Load balancing with capacity info | `LoadAwareExp3` | γ=0.15-0.25, set capacities |
| Simple capacity-based routing | `CapacityProportionalBandit` | γ=0.10, set capacities |
| A/B testing, equal distribution | `UniformBandit` | (no params) |

## Copy-Paste Code Templates

### Template 1: Simple Latency Minimization

```python
from literegistry.bandit import Exp3Dynamic

# Initialize
bandit = Exp3Dynamic(
    gamma=0.15,      # Moderate exploration
    L_max=2.0,       # Set to your P99 latency
    init_weight=1.0
)

arms = ['model_a', 'model_b', 'model_c']

# Select and update
chosen, prob = bandit.get_arm(arms, k=1)
arm = chosen[0]

# ... route request and measure latency ...

bandit.update(arm, success=True, latency=0.234)
```

### Template 2: Load Balancing with Capacity

```python
from literegistry.bandit_loadaware import LoadAwareExp3

# Initialize
bandit = LoadAwareExp3(
    gamma=0.20,           # Higher for load balancing
    L_max=2.0,
    load_window=100,
    capacity_aware=True
)

# Set capacities (relative or absolute)
bandit.set_capacities({
    'server_1': 100,  # requests/sec
    'server_2': 200,
    'server_3': 150,
})

arms = ['server_1', 'server_2', 'server_3']

# Select and update
chosen, prob = bandit.get_arm(arms, k=1)
arm = chosen[0]

# ... route request ...

bandit.update(arm, success=True, latency=0.345)

# Check current load
utilization = {a: bandit.get_utilization(a) for a in arms}
print(f"Utilization: {utilization}")
```

### Template 3: Find Optimal Parameters

```python
from literegistry.bandit_tuning import analyze_arm_characteristics

# Collect data first (run uniform routing for 1000-5000 requests)
arm_data = {
    'arm1': {
        'latencies': [0.15, 0.16, 0.14, ...],  # measured latencies
        'capacity': 100  # optional: requests/sec
    },
    'arm2': {
        'latencies': [0.45, 0.43, 0.47, ...],
        'capacity': 50
    },
}

# Get recommendations
recommendations = analyze_arm_characteristics(arm_data)

# Use recommended values
bandit = Exp3Dynamic(
    gamma=recommendations['recommended_gamma'],
    L_max=recommendations['recommended_L_max']
)
```

## Monitoring Checklist

Monitor these metrics in production:

- [ ] **Average latency** (rolling window of 1000) - should decrease and stabilize
- [ ] **Selection distribution** - check `bandit.get_probabilities()`
- [ ] **Per-arm utilization** (if load-aware) - should be balanced
- [ ] **P95/P99 latencies** - watch for tail latency issues

### Warning Signs

| Issue | Likely Cause | Solution |
|-------|--------------|----------|
| One arm >95% | Gamma too low | Increase gamma by 1.5x |
| All arms equal % | Gamma too high or L_max wrong | Decrease gamma or check L_max |
| Latency increasing | Overload / bad L_max | Use LoadAwareExp3 or adjust L_max |
| No convergence | Arms too similar | Increase samples or use UniformBandit |

## Common Mistakes

❌ **Setting L_max too high**
- All latencies look equally good
- No differentiation between arms
- ✅ Set to P99, not max

❌ **Gamma too low**
- Converges to first "lucky" arm
- Doesn't adapt to changes
- ✅ Start with 0.15, adjust based on distribution

❌ **Using Exp3 with congestion**
- Overloads one arm
- Performance degrades
- ✅ Use LoadAwareExp3 with capacity info

❌ **Not monitoring**
- Issues go unnoticed
- Suboptimal routing persists
- ✅ Log probabilities and latencies

## Tuning Process

1. **Start**: Use recommended defaults
   - Exp3Dynamic: γ=0.15, L_max=P99*1.1
   - LoadAwareExp3: γ=0.20, L_max=P99*1.1, window=100

2. **Monitor**: Watch for 1000-5000 requests

3. **Adjust** if needed:
   ```
   Too concentrated (>95% one arm):
     gamma_new = gamma * 1.5
   
   Too spread out (near uniform):
     gamma_new = gamma * 0.7
   
   High latency with load:
     Switch to LoadAwareExp3
   ```

4. **Repeat**: until stable

## For More Details

- Full guide: `python3 literegistry/bandit_tuning.py`
- Examples: `python3 example_parameter_tuning.py`
- Test results: `BANDIT_TEST_SUMMARY.md`

## Emergency Quick Fixes

**Latency too high?**
```python
# Switch to uniform to stop bad routing
from literegistry.bandit import UniformBandit
bandit = UniformBandit()
```

**Need more exploration?**
```python
# Increase gamma
bandit.gamma = 0.25  # from 0.15
```

**One arm overloaded?**
```python
# Switch to load-aware
from literegistry.bandit_loadaware import LoadAwareExp3
bandit = LoadAwareExp3(gamma=0.20, L_max=2.0)
bandit.set_capacities({'arm1': 100, 'arm2': 200})
```

