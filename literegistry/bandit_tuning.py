"""
Parameter tuning utilities for bandit algorithms.

Helps determine optimal parameters (gamma, L_max, etc.) for your specific use case.
"""

import numpy as np
from typing import Dict, List, Tuple, Callable
import itertools


class ParameterTuner:
    """
    Grid search and adaptive tuning for bandit parameters.
    
    Usage:
        tuner = ParameterTuner(simulator_func)
        best_params = tuner.tune_exp3(
            gamma_range=[0.05, 0.1, 0.15, 0.2],
            L_max_range=[1.0, 2.0, 5.0]
        )
    """
    
    def __init__(self, simulator_func: Callable, num_iterations: int = 5000):
        """
        Args:
            simulator_func: Function that simulates your workload
                Should accept (bandit, num_iterations) and return metrics dict
            num_iterations: Number of iterations per test
        """
        self.simulator_func = simulator_func
        self.num_iterations = num_iterations
        self.results = []
    
    def tune_exp3(self, 
                  gamma_range: List[float] = None,
                  L_max_range: List[float] = None,
                  init_weight_range: List[float] = None) -> Dict:
        """
        Grid search over Exp3 parameters.
        
        Returns:
            Best parameters and their metrics
        """
        if gamma_range is None:
            gamma_range = [0.05, 0.1, 0.15, 0.2, 0.3]
        if L_max_range is None:
            L_max_range = [1.0, 2.0, 5.0]
        if init_weight_range is None:
            init_weight_range = [1.0]
        
        print("="*70)
        print("PARAMETER TUNING: Exp3Dynamic")
        print("="*70)
        print(f"Testing {len(gamma_range)} x {len(L_max_range)} x {len(init_weight_range)} = "
              f"{len(gamma_range) * len(L_max_range) * len(init_weight_range)} combinations\n")
        
        best_score = float('inf')
        best_params = None
        best_metrics = None
        
        for gamma, L_max, init_weight in itertools.product(
            gamma_range, L_max_range, init_weight_range
        ):
            params = {
                'gamma': gamma,
                'L_max': L_max,
                'init_weight': init_weight
            }
            
            # Run simulation
            metrics = self.simulator_func(params, self.num_iterations)
            
            # Score is average recent latency (lower is better)
            score = metrics.get('recent_avg_latency', float('inf'))
            
            result = {
                'params': params,
                'metrics': metrics,
                'score': score
            }
            self.results.append(result)
            
            print(f"γ={gamma:.2f}, L_max={L_max:.1f}, w0={init_weight:.1f} → "
                  f"latency={score:.4f}, regret={metrics.get('total_regret', 0):.1f}")
            
            if score < best_score:
                best_score = score
                best_params = params
                best_metrics = metrics
        
        print("\n" + "="*70)
        print("BEST PARAMETERS:")
        print("="*70)
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print(f"\nPerformance:")
        print(f"  Average latency: {best_score:.4f}")
        print(f"  Total regret: {best_metrics.get('total_regret', 0):.2f}")
        
        return {
            'best_params': best_params,
            'best_metrics': best_metrics,
            'all_results': self.results
        }
    
    def tune_load_aware_exp3(self,
                             gamma_range: List[float] = None,
                             L_max_range: List[float] = None,
                             load_window_range: List[int] = None) -> Dict:
        """
        Grid search over LoadAwareExp3 parameters.
        """
        if gamma_range is None:
            gamma_range = [0.1, 0.15, 0.2, 0.25]
        if L_max_range is None:
            L_max_range = [1.0, 2.0]
        if load_window_range is None:
            load_window_range = [50, 100, 200]
        
        print("="*70)
        print("PARAMETER TUNING: LoadAwareExp3")
        print("="*70)
        print(f"Testing {len(gamma_range)} x {len(L_max_range)} x {len(load_window_range)} combinations\n")
        
        best_score = float('inf')
        best_params = None
        best_metrics = None
        
        for gamma, L_max, load_window in itertools.product(
            gamma_range, L_max_range, load_window_range
        ):
            params = {
                'gamma': gamma,
                'L_max': L_max,
                'load_window': load_window,
                'capacity_aware': True
            }
            
            metrics = self.simulator_func(params, self.num_iterations)
            score = metrics.get('recent_avg_latency', float('inf'))
            
            result = {
                'params': params,
                'metrics': metrics,
                'score': score
            }
            self.results.append(result)
            
            print(f"γ={gamma:.2f}, L_max={L_max:.1f}, window={load_window} → "
                  f"latency={score:.4f}")
            
            if score < best_score:
                best_score = score
                best_params = params
                best_metrics = metrics
        
        print("\n" + "="*70)
        print("BEST PARAMETERS:")
        print("="*70)
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print(f"\nPerformance: {best_score:.4f}")
        
        return {
            'best_params': best_params,
            'best_metrics': best_metrics,
            'all_results': self.results
        }


def estimate_L_max(latency_samples: List[float], percentile: float = 99.0) -> float:
    """
    Estimate L_max from historical latency data.
    
    Args:
        latency_samples: Historical latency observations
        percentile: Percentile to use (default 99th percentile)
    
    Returns:
        Recommended L_max value
    """
    if not latency_samples:
        return 1.0
    
    p = np.percentile(latency_samples, percentile)
    
    # Add 10% buffer
    L_max = p * 1.1
    
    print(f"Latency Analysis:")
    print(f"  Min: {np.min(latency_samples):.4f}")
    print(f"  Mean: {np.mean(latency_samples):.4f}")
    print(f"  Median: {np.median(latency_samples):.4f}")
    print(f"  P95: {np.percentile(latency_samples, 95):.4f}")
    print(f"  P99: {np.percentile(latency_samples, 99):.4f}")
    print(f"  Max: {np.max(latency_samples):.4f}")
    print(f"\nRecommended L_max: {L_max:.4f}")
    
    return L_max


def recommend_gamma(num_arms: int, 
                    exploration_need: str = 'medium',
                    time_horizon: int = 10000) -> float:
    """
    Recommend gamma based on problem characteristics.
    
    Args:
        num_arms: Number of arms
        exploration_need: 'low', 'medium', or 'high'
        time_horizon: Expected number of iterations
    
    Returns:
        Recommended gamma value
    """
    # Theoretical Exp3 bound suggests gamma ~ sqrt(K*log(K)/T)
    K = num_arms
    T = time_horizon
    
    theoretical = np.sqrt(K * np.log(K) / T)
    
    # Adjust based on exploration need
    if exploration_need == 'low':
        gamma = theoretical * 0.5
    elif exploration_need == 'medium':
        gamma = theoretical * 1.0
    elif exploration_need == 'high':
        gamma = theoretical * 2.0
    else:
        gamma = theoretical
    
    # Clamp to reasonable range
    gamma = max(0.01, min(0.5, gamma))
    
    print(f"Gamma Recommendation:")
    print(f"  Number of arms: {K}")
    print(f"  Time horizon: {T}")
    print(f"  Exploration need: {exploration_need}")
    print(f"  Theoretical optimal: {theoretical:.4f}")
    print(f"  Recommended: {gamma:.4f}")
    print(f"\nInterpretation:")
    print(f"  - Each arm gets ~{gamma/K*100:.1f}% minimum probability")
    print(f"  - Convergence speed: {'fast' if gamma < 0.1 else 'medium' if gamma < 0.2 else 'slow'}")
    
    return gamma


def analyze_arm_characteristics(arm_data: Dict[str, Dict]) -> Dict:
    """
    Analyze characteristics of your arms to guide parameter selection.
    
    Args:
        arm_data: Dict of {arm_name: {'latencies': [...], 'capacity': ...}}
    
    Returns:
        Analysis results and recommendations
    """
    print("="*70)
    print("ARM CHARACTERISTICS ANALYSIS")
    print("="*70)
    
    arms = list(arm_data.keys())
    num_arms = len(arms)
    
    latency_stats = {}
    capacities = {}
    
    for arm_name, data in arm_data.items():
        latencies = data.get('latencies', [])
        capacity = data.get('capacity', None)
        
        if latencies:
            latency_stats[arm_name] = {
                'mean': np.mean(latencies),
                'std': np.std(latencies),
                'min': np.min(latencies),
                'max': np.max(latencies),
            }
        
        if capacity:
            capacities[arm_name] = capacity
    
    # Print analysis
    print(f"\nNumber of arms: {num_arms}")
    print(f"\nLatency characteristics:")
    
    for arm_name, stats in latency_stats.items():
        print(f"\n  {arm_name}:")
        print(f"    Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        if arm_name in capacities:
            print(f"    Capacity: {capacities[arm_name]}")
    
    # Calculate differences
    means = [stats['mean'] for stats in latency_stats.values()]
    
    if len(means) > 1:
        best_mean = min(means)
        worst_mean = max(means)
        relative_diff = (worst_mean - best_mean) / best_mean
        
        print(f"\nLatency spread:")
        print(f"  Best arm mean: {best_mean:.4f}")
        print(f"  Worst arm mean: {worst_mean:.4f}")
        print(f"  Relative difference: {relative_diff*100:.1f}%")
        
        # Recommendations
        print(f"\n" + "="*70)
        print("RECOMMENDATIONS")
        print("="*70)
        
        if relative_diff < 0.1:
            print("\n✓ Arms have SIMILAR latencies (<10% difference)")
            print("  → High exploration needed to differentiate")
            print("  → Consider gamma=0.2-0.3")
            print("  → Consider UniformBandit or LoadAwareExp3")
            exploration = 'high'
        elif relative_diff < 0.5:
            print("\n✓ Arms have MODERATE latency differences (10-50%)")
            print("  → Balanced exploration/exploitation")
            print("  → Consider gamma=0.1-0.2")
            print("  → Exp3Dynamic should work well")
            exploration = 'medium'
        else:
            print("\n✓ Arms have LARGE latency differences (>50%)")
            print("  → Fast convergence possible")
            print("  → Consider gamma=0.05-0.15")
            print("  → Exp3Dynamic with lower gamma")
            exploration = 'low'
        
        # Capacity analysis
        if capacities:
            print(f"\n✓ Capacity information available")
            print(f"  → Use LoadAwareExp3 or CapacityProportionalBandit")
            print(f"  → Set capacities with bandit.set_capacities({capacities})")
            
            total_cap = sum(capacities.values())
            for arm, cap in capacities.items():
                pct = (cap / total_cap) * 100
                print(f"    {arm}: {pct:.1f}% of total capacity")
        
        # L_max recommendation
        all_latencies = []
        for data in arm_data.values():
            all_latencies.extend(data.get('latencies', []))
        
        if all_latencies:
            recommended_L_max = estimate_L_max(all_latencies)
            print(f"\n✓ Recommended L_max: {recommended_L_max:.4f}")
        
        # Gamma recommendation
        recommended_gamma = recommend_gamma(num_arms, exploration)
        
        return {
            'num_arms': num_arms,
            'latency_spread': relative_diff,
            'exploration_need': exploration,
            'recommended_gamma': recommended_gamma,
            'recommended_L_max': recommended_L_max if all_latencies else None,
            'has_capacity_info': bool(capacities),
            'capacities': capacities
        }


def quick_tuning_guide():
    """
    Print a quick reference guide for parameter tuning.
    """
    guide = """
╔══════════════════════════════════════════════════════════════════════╗
║                  BANDIT PARAMETER TUNING GUIDE                       ║
╚══════════════════════════════════════════════════════════════════════╝

1. GAMMA (γ) - Exploration Parameter
   ════════════════════════════════════════════════════════════════════
   
   Controls exploration vs exploitation tradeoff:
   
   γ = 0.05-0.10  ⟹  LOW EXPLORATION
      • Fast convergence to best arm (90%+ selection)
      • Use when: arms clearly different, stable environment
      • Risk: May miss changes, get stuck on suboptimal
   
   γ = 0.10-0.20  ⟹  MODERATE EXPLORATION  [RECOMMENDED DEFAULT]
      • Balanced exploration (80-90% best arm)
      • Use when: moderate differences, some uncertainty
      • Good starting point for most use cases
   
   γ = 0.20-0.30  ⟹  HIGH EXPLORATION
      • More diversity (70-80% best arm)
      • Use when: arms similar, dynamic environment
      • Better for load balancing scenarios
   
   γ > 0.30       ⟹  VERY HIGH EXPLORATION
      • Mostly uniform (60-70% best arm)
      • Use when: need sustained diversity
      • Approaching uniform random selection

   Formula: Each arm gets minimum γ/K probability
   Example: γ=0.15, K=5 arms ⟹ each gets ≥3% probability


2. L_MAX - Maximum Latency
   ════════════════════════════════════════════════════════════════════
   
   Normalizes rewards: reward = 1 - (latency / L_max)
   
   • Set to ~P99 latency of your slowest arm
   • If too small: rewards compressed, less differentiation
   • If too large: small latencies all look equally good
   
   How to set:
   ✓ Collect latency samples from all arms
   ✓ Take 99th percentile
   ✓ Add 10-20% buffer
   
   Example:
     P99 latency = 1.8 seconds
     L_max = 1.8 * 1.1 = 2.0 seconds


3. INIT_WEIGHT - Initial Weight
   ════════════════════════════════════════════════════════════════════
   
   Starting weight for new arms (log space)
   
   • Default: 1.0 (works for most cases)
   • Higher: new arms start with higher selection probability
   • Lower: new arms start with lower probability
   
   Usually don't need to tune unless:
   • Adding arms dynamically during operation
   • Want to control initial exploration of new arms


4. LOAD_WINDOW - Load Tracking Window (LoadAwareExp3 only)
   ════════════════════════════════════════════════════════════════════
   
   Number of recent requests tracked for load calculation
   
   window = 50    ⟹  Fast response to load changes
   window = 100   ⟹  Balanced [RECOMMENDED]
   window = 200   ⟹  Smoothed, slower response
   
   Trade-off:
   • Smaller: More reactive, more noisy
   • Larger: More stable, slower adaptation


5. CAPACITY_AWARE - Use Capacity Information
   ════════════════════════════════════════════════════════════════════
   
   If you know arm capacities:
   ✓ Set: bandit.set_capacities({'arm1': 100, 'arm2': 200})
   ✓ Use LoadAwareExp3 or CapacityProportionalBandit
   
   Capacity units:
   • Relative (arm1=1, arm2=2) ⟹ arm2 has 2x capacity
   • Absolute (requests/sec) if available
   • Estimated from benchmarks


╔══════════════════════════════════════════════════════════════════════╗
║                        ALGORITHM SELECTION                           ║
╚══════════════════════════════════════════════════════════════════════╝

UniformBandit
─────────────────────────────────────────────────────────────────────
✓ When: Equal treatment of all arms desired
✓ Use case: A/B testing, initial exploration, fairness requirements
✗ Doesn't learn or adapt


Exp3Dynamic
─────────────────────────────────────────────────────────────────────
✓ When: Minimize latency, arms independent (no congestion)
✓ Use case: Routing between different model sizes, different clouds
✓ Parameters: gamma=0.10-0.20, L_max=P99 latency
✗ Can overload single arm if congestion exists


LoadAwareExp3
─────────────────────────────────────────────────────────────────────
✓ When: Congestion matters, have capacity information
✓ Use case: Load balancing with capacity constraints
✓ Parameters: gamma=0.15-0.25, L_max=P99, load_window=100
✓ Must set capacities: bandit.set_capacities({...})


CapacityProportionalBandit
─────────────────────────────────────────────────────────────────────
✓ When: Capacity is primary concern, heterogeneous capacities
✓ Use case: Simple load balancing, capacity-based routing
✓ Parameters: gamma=0.10, L_max=P99
✓ Simpler than LoadAwareExp3, less adaptive


╔══════════════════════════════════════════════════════════════════════╗
║                          QUICK START                                 ║
╚══════════════════════════════════════════════════════════════════════╝

1. Collect data:
   - Sample latencies from each arm (~100-1000 samples)
   - Measure or estimate capacities if relevant

2. Analyze:
   from literegistry.bandit_tuning import analyze_arm_characteristics
   
   arm_data = {
       'arm1': {'latencies': [...], 'capacity': 100},
       'arm2': {'latencies': [...], 'capacity': 150},
   }
   recommendations = analyze_arm_characteristics(arm_data)

3. Choose algorithm and parameters based on recommendations

4. Test in simulation before production

5. Monitor and adjust:
   - Track average latency over time
   - Check selection distribution
   - Adjust gamma if too concentrated or too spread out


╔══════════════════════════════════════════════════════════════════════╗
║                      MONITORING CHECKLIST                            ║
╚══════════════════════════════════════════════════════════════════════╝

Monitor these metrics:

□ Average latency (rolling window)
  → Should decrease over time and stabilize

□ Selection distribution
  → Check if reasonable given arm characteristics
  → In load balancing, should roughly match capacities

□ Per-arm utilization (if using LoadAwareExp3)
  → Should be balanced, avoid overload

□ Regret (cumulative difference from optimal)
  → Should grow sublinearly (slower over time)

□ P95/P99 latencies
  → Check for tail latency issues

Warnings:
⚠ If one arm gets >95%: increase gamma for more exploration
⚠ If all arms equal: may need to decrease gamma or check L_max
⚠ If latency increasing: check for overload, adjust capacity params
"""
    print(guide)


if __name__ == "__main__":
    quick_tuning_guide()

