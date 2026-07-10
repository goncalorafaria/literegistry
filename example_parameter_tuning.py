#!/usr/bin/env python3
"""
Example: How to tune bandit parameters for your specific use case.

This shows the complete workflow from data collection to parameter selection.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from literegistry.bandit_tuning import (
    analyze_arm_characteristics,
    estimate_L_max,
    recommend_gamma,
    ParameterTuner,
    quick_tuning_guide
)


def example_1_analyze_your_arms():
    """
    Step 1: Analyze your arm characteristics to get recommendations.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Analyze Your Arms")
    print("="*70)
    
    # Example: You have 3 models/endpoints with different characteristics
    arm_data = {
        'fast_model': {
            'latencies': np.random.normal(0.15, 0.03, 500).tolist(),  # mean=0.15s
            'capacity': 100  # requests/sec
        },
        'accurate_model': {
            'latencies': np.random.normal(0.45, 0.08, 500).tolist(),  # mean=0.45s
            'capacity': 50   # requests/sec
        },
        'cheap_model': {
            'latencies': np.random.normal(0.30, 0.05, 500).tolist(),  # mean=0.30s
            'capacity': 150  # requests/sec
        }
    }
    
    # Analyze and get recommendations
    recommendations = analyze_arm_characteristics(arm_data)
    
    return recommendations


def example_2_estimate_parameters():
    """
    Step 2: Estimate individual parameters from data.
    """
    print("\n\n" + "="*70)
    print("EXAMPLE 2: Estimate Individual Parameters")
    print("="*70)
    
    # Collect latency samples
    print("\n--- Estimating L_max ---")
    all_latencies = np.concatenate([
        np.random.normal(0.15, 0.03, 200),
        np.random.normal(0.45, 0.08, 200),
        np.random.normal(0.30, 0.05, 200),
    ])
    
    L_max = estimate_L_max(all_latencies, percentile=99)
    
    # Recommend gamma
    print("\n--- Recommending Gamma ---")
    gamma = recommend_gamma(
        num_arms=3,
        exploration_need='medium',
        time_horizon=10000
    )
    
    return {'L_max': L_max, 'gamma': gamma}


def example_3_grid_search():
    """
    Step 3: Run grid search to find best parameters empirically.
    """
    print("\n\n" + "="*70)
    print("EXAMPLE 3: Grid Search for Optimal Parameters")
    print("="*70)
    
    # Define a simulator for your specific workload
    def my_simulator(params, num_iterations):
        """
        Simulates your workload with given parameters.
        Replace this with your actual simulation or test environment.
        """
        from literegistry.bandit import Exp3Dynamic
        
        # Create bandit with test parameters
        bandit = Exp3Dynamic(
            gamma=params['gamma'],
            L_max=params['L_max'],
            init_weight=params.get('init_weight', 1.0)
        )
        
        # Simulate arms with different characteristics
        arms = {
            'fast': {'mean': 0.15, 'std': 0.03},
            'medium': {'mean': 0.30, 'std': 0.05},
            'slow': {'mean': 0.45, 'std': 0.08},
        }
        arm_names = list(arms.keys())
        
        latencies = []
        total_latency = 0
        optimal_latency = 0
        
        for i in range(num_iterations):
            # Select arm
            chosen, _ = bandit.get_arm(arm_names, k=1)
            if not chosen:
                continue
            
            arm = chosen[0]
            
            # Simulate latency
            latency = max(0.01, np.random.normal(
                arms[arm]['mean'],
                arms[arm]['std']
            ))
            success = np.random.random() < 0.98
            
            # Update
            bandit.update(arm, success, latency)
            
            # Track metrics
            latencies.append(latency)
            total_latency += latency
            optimal_latency += arms['fast']['mean']  # Optimal is always fast
        
        # Calculate metrics
        recent_latencies = latencies[-1000:] if len(latencies) >= 1000 else latencies
        
        return {
            'avg_latency': np.mean(latencies),
            'recent_avg_latency': np.mean(recent_latencies),
            'total_regret': total_latency - optimal_latency,
            'final_probs': bandit.get_probabilities()
        }
    
    # Run grid search
    tuner = ParameterTuner(my_simulator, num_iterations=3000)
    
    results = tuner.tune_exp3(
        gamma_range=[0.05, 0.10, 0.15, 0.20],
        L_max_range=[0.8, 1.0, 1.5],
        init_weight_range=[1.0]
    )
    
    return results


def example_4_practical_workflow():
    """
    Step 4: Complete practical workflow.
    """
    print("\n\n" + "="*70)
    print("EXAMPLE 4: Complete Practical Workflow")
    print("="*70)
    
    print("""
STEP-BY-STEP GUIDE TO TUNE YOUR BANDIT:

1. COLLECT DATA (do this first!)
   ────────────────────────────────────────────────────────────────
   Run a short A/B test or collect metrics from existing system:
   
   from collections import defaultdict
   
   arm_data = defaultdict(lambda: {'latencies': [], 'capacity': None})
   
   # For each request:
   #   arm = choose_arm_uniformly()  # or current routing
   #   latency = measure_latency(arm)
   #   arm_data[arm]['latencies'].append(latency)
   
   # Estimate capacities (optional but recommended):
   #   arm_data['arm1']['capacity'] = 100  # requests/sec
   

2. ANALYZE DATA
   ────────────────────────────────────────────────────────────────
   from literegistry.bandit_tuning import analyze_arm_characteristics
   
   recommendations = analyze_arm_characteristics(arm_data)
   
   This will tell you:
   - Which algorithm to use
   - Recommended gamma and L_max
   - Whether to use capacity-aware routing


3. START WITH RECOMMENDED PARAMETERS
   ────────────────────────────────────────────────────────────────
   Example based on analysis:
   
   from literegistry.bandit import Exp3Dynamic  # or LoadAwareExp3
   
   bandit = Exp3Dynamic(
       gamma=recommendations['recommended_gamma'],
       L_max=recommendations['recommended_L_max'],
       init_weight=1.0
   )
   
   # If using LoadAwareExp3:
   # bandit.set_capacities(recommendations['capacities'])


4. TEST IN SIMULATION (recommended!)
   ────────────────────────────────────────────────────────────────
   Before production, test with your collected data:
   
   - Replay historical requests
   - Simulate different scenarios
   - Compare different parameter settings
   - Use ParameterTuner for grid search


5. DEPLOY WITH MONITORING
   ────────────────────────────────────────────────────────────────
   Deploy with careful monitoring:
   
   metrics = {
       'avg_latency': rolling_average(window=1000),
       'selection_distribution': Counter(recent_selections),
       'p95_latency': percentile(latencies, 95),
       'regret': cumulative_latency - optimal_cumulative
   }
   
   Watch for:
   ⚠ Latency increasing → may be overloading
   ⚠ One arm >95% → increase gamma
   ⚠ All arms ~equal → may need to decrease gamma


6. TUNE BASED ON PRODUCTION METRICS
   ────────────────────────────────────────────────────────────────
   Adjust parameters based on observed behavior:
   
   If too concentrated (one arm >95%):
      gamma_new = gamma * 1.5  # Increase exploration
   
   If too spread out (uniform distribution):
      gamma_new = gamma * 0.7  # Decrease exploration
   
   If average latency not improving:
      - Check L_max (may be too high or low)
      - Check if arms are actually different
      - Consider load-aware algorithm


EXAMPLE CODE:
────────────────────────────────────────────────────────────────

# 1. Initial setup with recommendations
from literegistry.bandit import Exp3Dynamic

bandit = Exp3Dynamic(gamma=0.15, L_max=1.2, init_weight=1.0)
arms = ['model_a', 'model_b', 'model_c']

# 2. In your request handler
def handle_request(request):
    # Select arm
    chosen, prob = bandit.get_arm(arms, k=1)
    arm = chosen[0]
    
    # Route request
    start = time.time()
    success, response = route_to_arm(arm, request)
    latency = time.time() - start
    
    # Update bandit
    bandit.update(arm, success, latency)
    
    # Log metrics
    log_metrics({
        'arm': arm,
        'latency': latency,
        'success': success,
        'probability': prob[0]
    })
    
    return response

# 3. Periodic monitoring (every N requests or time interval)
def check_metrics():
    probs = bandit.get_probabilities()
    print(f"Current distribution: {probs}")
    
    recent_latency = get_recent_avg_latency(window=1000)
    print(f"Recent avg latency: {recent_latency:.4f}")
    
    # Alert if needed
    if max(probs.values()) > 0.95:
        print("⚠ Warning: Very concentrated, consider increasing gamma")

""")


def main():
    """Run all examples."""
    
    # Show quick reference guide first
    quick_tuning_guide()
    
    input("\n\nPress Enter to see examples...")
    
    # Run examples
    recommendations = example_1_analyze_your_arms()
    
    input("\n\nPress Enter to continue...")
    
    params = example_2_estimate_parameters()
    
    input("\n\nPress Enter to run grid search (this takes ~30 seconds)...")
    
    results = example_3_grid_search()
    
    input("\n\nPress Enter to see practical workflow guide...")
    
    example_4_practical_workflow()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nBased on examples above, here's what to do:")
    print("\n1. Collect latency samples from your arms")
    print("2. Run: analyze_arm_characteristics(your_data)")
    print("3. Use recommended parameters")
    print("4. Monitor and adjust as needed")
    print("\nFor more details, see: BANDIT_TEST_SUMMARY.md")


if __name__ == "__main__":
    main()

