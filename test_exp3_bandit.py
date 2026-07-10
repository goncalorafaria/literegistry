#!/usr/bin/env python3
"""
Test script to verify Exp3Dynamic bandit learns to minimize latency.
Tests if it converges to preferring faster arms over time.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict

# Import directly from the bandit module
import importlib.util
spec = importlib.util.spec_from_file_location("bandit", "literegistry/bandit.py")
bandit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bandit_module)
Exp3Dynamic = bandit_module.Exp3Dynamic


class ArmSimulator:
    """Simulates arms with different latency characteristics."""
    
    def __init__(self, name, mean_latency, std_dev=0.1, success_rate=0.95):
        self.name = name
        self.mean_latency = mean_latency
        self.std_dev = std_dev
        self.success_rate = success_rate
        self.total_calls = 0
        self.total_latency = 0
    
    def call(self):
        """Simulate calling this arm."""
        self.total_calls += 1
        
        # Random success/failure
        success = np.random.random() < self.success_rate
        
        # Random latency (normal distribution, clipped to positive)
        latency = max(0.01, np.random.normal(self.mean_latency, self.std_dev))
        
        if success:
            self.total_latency += latency
        
        return success, latency
    
    def get_avg_latency(self):
        """Get average latency for successful calls."""
        return self.total_latency / max(1, self.total_calls)
    
    def __repr__(self):
        return f"{self.name}(μ={self.mean_latency:.2f})"


def test_exp3_convergence(arms, num_samples=5000, gamma=0.2, L_max=2.0):
    """
    Test if Exp3 learns to prefer faster arms.
    
    Args:
        arms: Dictionary of arm_name -> ArmSimulator
        num_samples: Number of iterations
        gamma: Exploration parameter (lower = more exploitation)
        L_max: Maximum latency for normalization
    """
    arm_names = list(arms.keys())
    bandit = Exp3Dynamic(gamma=gamma, L_max=L_max, init_weight=1.0)
    
    # Track metrics over time
    history = {
        'selections': [],
        'latencies': [],
        'cumulative_latency': [],
        'probabilities': defaultdict(list),
        'cumulative_regret': [],
    }
    
    # Find optimal arm (lowest mean latency)
    optimal_arm = min(arms.values(), key=lambda a: a.mean_latency)
    
    cumulative_latency = 0
    optimal_cumulative_latency = 0
    
    print(f"\nTesting Exp3Dynamic with gamma={gamma}, L_max={L_max}")
    print(f"Arms: {arm_names}")
    print(f"Optimal arm: {optimal_arm.name} (mean latency: {optimal_arm.mean_latency:.3f})")
    print(f"\nRunning {num_samples} iterations...\n")
    
    for i in range(num_samples):
        # Get current probabilities
        probs = bandit.get_probabilities()
        for arm_name in arm_names:
            history['probabilities'][arm_name].append(probs.get(arm_name, 0))
        
        # Select arm
        chosen_arms, _ = bandit.get_arm(arm_names, k=1)
        if not chosen_arms:
            continue
        
        chosen_arm = chosen_arms[0]
        
        # Simulate call to chosen arm
        success, latency = arms[chosen_arm].call()
        
        # Update bandit
        bandit.update(chosen_arm, success, latency)
        
        # Track metrics
        history['selections'].append(chosen_arm)
        history['latencies'].append(latency if success else L_max)
        cumulative_latency += (latency if success else L_max)
        history['cumulative_latency'].append(cumulative_latency)
        
        # Track optimal cumulative latency (what we'd get if always choosing optimal)
        optimal_success, optimal_latency = optimal_arm.mean_latency, optimal_arm.mean_latency
        optimal_cumulative_latency += optimal_arm.mean_latency
        
        # Regret = difference from optimal
        history['cumulative_regret'].append(cumulative_latency - optimal_cumulative_latency)
    
    return history, bandit


def analyze_results(history, arms, bandit):
    """Analyze and print results."""
    arm_names = list(arms.keys())
    selections = history['selections']
    
    print("="*70)
    print("RESULTS ANALYSIS")
    print("="*70)
    
    # Selection counts
    counts = Counter(selections)
    total_selections = len(selections)
    
    print(f"\nSelection Distribution (last 1000 iterations):")
    recent_counts = Counter(selections[-1000:])
    
    # Sort by mean latency
    sorted_arms = sorted(arms.items(), key=lambda x: x[1].mean_latency)
    
    for arm_name, arm_sim in sorted_arms:
        total_count = counts.get(arm_name, 0)
        recent_count = recent_counts.get(arm_name, 0)
        total_pct = (total_count / total_selections) * 100
        recent_pct = (recent_count / 1000) * 100
        avg_latency = arm_sim.get_avg_latency()
        
        is_optimal = arm_sim.mean_latency == min(a.mean_latency for a in arms.values())
        marker = "⭐" if is_optimal else "  "
        
        print(f"{marker} {arm_name}: mean_latency={arm_sim.mean_latency:.3f}")
        print(f"     Total: {total_count:5d} ({total_pct:5.2f}%)")
        print(f"     Recent: {recent_count:4d} ({recent_pct:5.2f}%)")
        print(f"     Observed avg latency: {avg_latency:.3f}")
    
    # Final probabilities
    print(f"\nFinal Arm Probabilities (from Exp3):")
    final_probs = bandit.get_probabilities()
    for arm_name, arm_sim in sorted_arms:
        prob = final_probs.get(arm_name, 0)
        is_optimal = arm_sim.mean_latency == min(a.mean_latency for a in arms.values())
        marker = "⭐" if is_optimal else "  "
        print(f"{marker} {arm_name}: {prob*100:5.2f}%")
    
    # Performance metrics
    total_latency = history['cumulative_latency'][-1]
    avg_latency = total_latency / len(selections)
    
    optimal_arm = min(arms.values(), key=lambda a: a.mean_latency)
    optimal_avg_latency = optimal_arm.mean_latency
    
    print(f"\nPerformance Metrics:")
    print(f"  Average latency (Exp3): {avg_latency:.4f}")
    print(f"  Average latency (Optimal): {optimal_avg_latency:.4f}")
    print(f"  Difference: {avg_latency - optimal_avg_latency:.4f}")
    print(f"  Total regret: {history['cumulative_regret'][-1]:.2f}")
    
    # Check convergence - look at last 1000 iterations
    if len(selections) >= 1000:
        recent_avg_latency = sum(history['latencies'][-1000:]) / 1000
        print(f"  Recent avg latency (last 1000): {recent_avg_latency:.4f}")
        
        # Check if converged to optimal
        optimal_arm_name = optimal_arm.name
        optimal_selection_rate = recent_counts.get(optimal_arm_name, 0) / 1000
        print(f"  Optimal arm selection rate (recent): {optimal_selection_rate*100:.2f}%")


def plot_exp3_results(history, arms, output_file='exp3_bandit_results.png'):
    """Create comprehensive visualization of Exp3 learning."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    arm_names = list(arms.keys())
    iterations = range(len(history['selections']))
    
    # Identify optimal arm
    optimal_arm = min(arms.items(), key=lambda x: x[1].mean_latency)
    
    # Color scheme for arms (optimal in green)
    colors = {}
    for arm_name, arm_sim in arms.items():
        if arm_name == optimal_arm[0]:
            colors[arm_name] = 'green'
        else:
            colors[arm_name] = plt.cm.tab10(hash(arm_name) % 10)
    
    # 1. Probability evolution
    ax1 = fig.add_subplot(gs[0, 0])
    for arm_name in arm_names:
        probs = history['probabilities'][arm_name]
        ax1.plot(probs, label=f"{arm_name} (μ={arms[arm_name].mean_latency:.2f})",
                color=colors[arm_name], linewidth=2 if arm_name == optimal_arm[0] else 1)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Selection Probability')
    ax1.set_title('Arm Selection Probabilities Over Time', fontweight='bold')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(alpha=0.3)
    
    # 2. Selection counts over time (cumulative)
    ax2 = fig.add_subplot(gs[0, 1])
    cumulative_counts = {arm: [] for arm in arm_names}
    for arm in arm_names:
        count = 0
        for selection in history['selections']:
            if selection == arm:
                count += 1
            cumulative_counts[arm].append(count)
    
    for arm_name in arm_names:
        ax2.plot(cumulative_counts[arm_name], label=f"{arm_name}",
                color=colors[arm_name], linewidth=2 if arm_name == optimal_arm[0] else 1)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Cumulative Selections')
    ax2.set_title('Cumulative Selection Counts', fontweight='bold')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(alpha=0.3)
    
    # 3. Average latency over time (rolling window)
    ax3 = fig.add_subplot(gs[1, 0])
    window_size = 100
    rolling_avg = []
    for i in range(len(history['latencies'])):
        start = max(0, i - window_size + 1)
        rolling_avg.append(np.mean(history['latencies'][start:i+1]))
    
    ax3.plot(rolling_avg, color='blue', linewidth=2, label='Exp3 (rolling avg)')
    ax3.axhline(y=optimal_arm[1].mean_latency, color='green', linestyle='--',
               linewidth=2, label=f'Optimal ({optimal_arm[1].mean_latency:.3f})')
    
    # Add other arms' mean latencies for reference
    for arm_name, arm_sim in arms.items():
        if arm_name != optimal_arm[0]:
            ax3.axhline(y=arm_sim.mean_latency, color=colors[arm_name], 
                       linestyle=':', alpha=0.5, linewidth=1)
    
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Average Latency')
    ax3.set_title(f'Learning Progress (window={window_size})', fontweight='bold')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(alpha=0.3)
    
    # 4. Cumulative regret
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(history['cumulative_regret'], color='red', linewidth=2)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Cumulative Regret')
    ax4.set_title('Cumulative Regret vs Optimal', fontweight='bold')
    ax4.grid(alpha=0.3)
    
    # 5. Selection distribution (pie chart - recent)
    ax5 = fig.add_subplot(gs[2, 0])
    recent_selections = history['selections'][-1000:]
    recent_counts = Counter(recent_selections)
    
    labels = []
    sizes = []
    pie_colors = []
    for arm_name in arm_names:
        count = recent_counts.get(arm_name, 0)
        if count > 0:
            labels.append(f"{arm_name}\n(μ={arms[arm_name].mean_latency:.2f})")
            sizes.append(count)
            pie_colors.append(colors[arm_name])
    
    ax5.pie(sizes, labels=labels, colors=pie_colors, autopct='%1.1f%%',
           startangle=90)
    ax5.set_title('Selection Distribution (Last 1000 Iterations)', fontweight='bold')
    
    # 6. Latency histogram by arm
    ax6 = fig.add_subplot(gs[2, 1])
    
    # Collect latencies by arm
    latencies_by_arm = {arm: [] for arm in arm_names}
    for selection, latency in zip(history['selections'], history['latencies']):
        latencies_by_arm[selection].append(latency)
    
    positions = np.arange(len(arm_names))
    bp = ax6.boxplot([latencies_by_arm[arm] for arm in arm_names],
                      positions=positions,
                      labels=[f"{arm}\n(μ={arms[arm].mean_latency:.2f})" for arm in arm_names],
                      patch_artist=True)
    
    # Color the boxes
    for patch, arm_name in zip(bp['boxes'], arm_names):
        patch.set_facecolor(colors[arm_name])
        patch.set_alpha(0.6)
    
    ax6.set_ylabel('Latency')
    ax6.set_title('Observed Latency Distribution by Arm', fontweight='bold')
    ax6.grid(alpha=0.3, axis='y')
    
    plt.suptitle('Exp3 Bandit Learning Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")


def test_scenario_1():
    """Scenario 1: Clear winner - one arm is much faster."""
    print("\n" + "="*70)
    print("SCENARIO 1: Clear Winner")
    print("One arm significantly faster than others")
    print("="*70)
    
    arms = {
        'fast_arm': ArmSimulator('fast_arm', mean_latency=0.1, std_dev=0.02),
        'medium_arm': ArmSimulator('medium_arm', mean_latency=0.5, std_dev=0.1),
        'slow_arm': ArmSimulator('slow_arm', mean_latency=1.0, std_dev=0.2),
    }
    
    history, bandit = test_exp3_convergence(arms, num_samples=3000, gamma=0.1, L_max=2.0)
    analyze_results(history, arms, bandit)
    plot_exp3_results(history, arms, 'exp3_scenario1_clear_winner.png')


def test_scenario_2():
    """Scenario 2: Close competition - arms have similar latencies."""
    print("\n" + "="*70)
    print("SCENARIO 2: Close Competition")
    print("Multiple arms with similar latencies")
    print("="*70)
    
    arms = {
        'arm1': ArmSimulator('arm1', mean_latency=0.30, std_dev=0.05),
        'arm2': ArmSimulator('arm2', mean_latency=0.32, std_dev=0.05),
        'arm3': ArmSimulator('arm3', mean_latency=0.35, std_dev=0.05),
        'arm4': ArmSimulator('arm4', mean_latency=0.50, std_dev=0.08),
    }
    
    history, bandit = test_exp3_convergence(arms, num_samples=5000, gamma=0.15, L_max=1.0)
    analyze_results(history, arms, bandit)
    plot_exp3_results(history, arms, 'exp3_scenario2_close_competition.png')


def test_scenario_3():
    """Scenario 3: Many arms with one optimal."""
    print("\n" + "="*70)
    print("SCENARIO 3: Many Arms")
    print("One optimal arm among many mediocre ones")
    print("="*70)
    
    arms = {
        f'arm{i}': ArmSimulator(f'arm{i}', mean_latency=0.5 + i*0.1, std_dev=0.05)
        for i in range(7)
    }
    # Add one clearly optimal arm
    arms['optimal'] = ArmSimulator('optimal', mean_latency=0.2, std_dev=0.03)
    
    history, bandit = test_exp3_convergence(arms, num_samples=8000, gamma=0.2, L_max=2.0)
    analyze_results(history, arms, bandit)
    plot_exp3_results(history, arms, 'exp3_scenario3_many_arms.png')


def test_gamma_comparison():
    """Compare different gamma values (exploration vs exploitation)."""
    print("\n" + "="*70)
    print("SCENARIO 4: Gamma Comparison")
    print("Testing different exploration parameters")
    print("="*70)
    
    # Fixed set of arms
    base_arms = {
        'fast': ArmSimulator('fast', mean_latency=0.2, std_dev=0.03),
        'medium': ArmSimulator('medium', mean_latency=0.5, std_dev=0.05),
        'slow': ArmSimulator('slow', mean_latency=0.9, std_dev=0.1),
    }
    
    gammas = [0.05, 0.1, 0.2, 0.4]
    results = {}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, gamma in enumerate(gammas):
        print(f"\nTesting gamma = {gamma}")
        
        # Create fresh arms for each test
        arms = {
            name: ArmSimulator(name, sim.mean_latency, sim.std_dev, sim.success_rate)
            for name, sim in base_arms.items()
        }
        
        history, bandit = test_exp3_convergence(arms, num_samples=3000, gamma=gamma, L_max=1.5)
        results[gamma] = (history, arms, bandit)
        
        # Plot probability evolution
        ax = axes[idx]
        for arm_name in arms.keys():
            probs = history['probabilities'][arm_name]
            ax.plot(probs, label=f"{arm_name} (μ={arms[arm_name].mean_latency:.2f})", linewidth=2)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Selection Probability')
        ax.set_title(f'γ = {gamma}', fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)
        
        # Print final stats
        final_probs = bandit.get_probabilities()
        print(f"  Final probabilities: {final_probs}")
        recent_selections = Counter(history['selections'][-500:])
        print(f"  Recent selections (last 500): {dict(recent_selections)}")
    
    plt.suptitle('Effect of Gamma (Exploration Parameter) on Exp3', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('exp3_gamma_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nGamma comparison saved to: exp3_gamma_comparison.png")


def main():
    print("="*70)
    print("EXP3 BANDIT LATENCY MINIMIZATION TEST")
    print("="*70)
    
    # Run different scenarios
    test_scenario_1()  # Clear winner
    test_scenario_2()  # Close competition
    test_scenario_3()  # Many arms
    test_gamma_comparison()  # Exploration parameter tuning
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - exp3_scenario1_clear_winner.png")
    print("  - exp3_scenario2_close_competition.png")
    print("  - exp3_scenario3_many_arms.png")
    print("  - exp3_gamma_comparison.png")


if __name__ == "__main__":
    main()

