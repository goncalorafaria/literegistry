#!/usr/bin/env python3
"""
Test Exp3 bandit with realistic load balancing - where arms have capacity limits
and latency increases under load (congestion).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, deque

# Import directly from the bandit module
import importlib.util
spec = importlib.util.spec_from_file_location("bandit", "literegistry/bandit.py")
bandit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bandit_module)
Exp3Dynamic = bandit_module.Exp3Dynamic


class CongestionArmSimulator:
    """
    Simulates an arm with congestion/queueing effects.
    Latency increases as more concurrent requests are being served.
    """
    
    def __init__(self, name, base_latency, capacity, congestion_factor=1.0):
        """
        Args:
            name: Arm identifier
            base_latency: Base latency when unloaded (seconds)
            capacity: Optimal capacity (requests/sec)
            congestion_factor: How much latency increases under load
        """
        self.name = name
        self.base_latency = base_latency
        self.capacity = capacity
        self.congestion_factor = congestion_factor
        
        # Track recent load (sliding window)
        self.recent_requests = deque(maxlen=100)  # Track last 100 requests
        self.total_calls = 0
        self.total_latency = 0
        
    def get_current_load(self):
        """Estimate current load based on recent request rate."""
        if len(self.recent_requests) < 2:
            return 0
        # Approximate requests per second from recent history
        return len(self.recent_requests)
    
    def get_latency(self):
        """
        Calculate latency based on current load.
        Latency increases as load approaches/exceeds capacity.
        
        Uses queueing theory approximation:
        latency ≈ base_latency * (1 + congestion_factor * (load/capacity)^2)
        """
        load = self.get_current_load()
        utilization = load / self.capacity
        
        # Latency increases quadratically as utilization increases
        congestion_multiplier = 1 + self.congestion_factor * (utilization ** 2)
        
        # Add some noise
        noise = np.random.normal(1.0, 0.1)
        latency = self.base_latency * congestion_multiplier * noise
        
        return max(0.01, latency)
    
    def call(self, current_time):
        """Simulate calling this arm."""
        self.total_calls += 1
        self.recent_requests.append(current_time)
        
        # Get latency based on current congestion
        latency = self.get_latency()
        self.total_latency += latency
        
        # Success rate might decrease under heavy load
        base_success_rate = 0.98
        load = self.get_current_load()
        utilization = load / self.capacity
        # Success rate drops if overloaded
        success_rate = base_success_rate * (1 - 0.3 * max(0, utilization - 1.0))
        success = np.random.random() < success_rate
        
        return success, latency
    
    def get_avg_latency(self):
        """Get average observed latency."""
        return self.total_latency / max(1, self.total_calls)
    
    def get_theoretical_optimal_latency(self):
        """What latency would be at optimal load."""
        optimal_utilization = 0.7  # Rule of thumb: keep utilization around 70%
        congestion_multiplier = 1 + self.congestion_factor * (optimal_utilization ** 2)
        return self.base_latency * congestion_multiplier
    
    def __repr__(self):
        return f"{self.name}(base={self.base_latency:.3f}, cap={self.capacity})"


def test_load_balancing(arms, num_samples=5000, gamma=0.2, L_max=2.0, window_size=10):
    """
    Test Exp3 with congestion-aware arms.
    
    Args:
        arms: Dictionary of arm_name -> CongestionArmSimulator
        num_samples: Number of requests to simulate
        gamma: Exploration parameter
        L_max: Maximum latency for reward normalization
        window_size: Time window for simulating concurrent requests
    """
    arm_names = list(arms.keys())
    bandit = Exp3Dynamic(gamma=gamma, L_max=L_max, init_weight=1.0)
    
    history = {
        'selections': [],
        'latencies': [],
        'loads': {arm: [] for arm in arm_names},
        'probabilities': {arm: [] for arm in arm_names},
        'cumulative_latency': [],
    }
    
    cumulative_latency = 0
    
    print(f"\nTesting Exp3 with Load Balancing")
    print(f"Arms: {list(arms.values())}")
    print(f"Gamma: {gamma}, L_max: {L_max}")
    print(f"Running {num_samples} requests...\n")
    
    for t in range(num_samples):
        # Get current probabilities
        probs = bandit.get_probabilities()
        for arm_name in arm_names:
            history['probabilities'][arm_name].append(probs.get(arm_name, 0))
        
        # Record current loads
        for arm_name in arm_names:
            history['loads'][arm_name].append(arms[arm_name].get_current_load())
        
        # Select arm
        chosen_arms, _ = bandit.get_arm(arm_names, k=1)
        if not chosen_arms:
            continue
        
        chosen_arm = chosen_arms[0]
        
        # Call the arm (with congestion effects)
        success, latency = arms[chosen_arm].call(t)
        
        # Update bandit
        bandit.update(chosen_arm, success, latency)
        
        # Track metrics
        history['selections'].append(chosen_arm)
        history['latencies'].append(latency)
        cumulative_latency += latency
        history['cumulative_latency'].append(cumulative_latency)
    
    return history, bandit


def analyze_load_balancing_results(history, arms, bandit):
    """Analyze and print load balancing results."""
    arm_names = list(arms.keys())
    selections = history['selections']
    
    print("="*70)
    print("LOAD BALANCING RESULTS")
    print("="*70)
    
    # Selection distribution
    counts = Counter(selections)
    total = len(selections)
    
    print(f"\nArm Characteristics and Performance:")
    print(f"{'Arm':<12} {'Base':<8} {'Capacity':<10} {'Selections':<12} {'Avg Latency':<12} {'Avg Load':<10}")
    print("-" * 70)
    
    total_theoretical_capacity = sum(arm.capacity for arm in arms.values())
    
    for arm_name, arm in arms.items():
        count = counts.get(arm_name, 0)
        pct = (count / total) * 100
        avg_latency = arm.get_avg_latency()
        avg_load = np.mean(history['loads'][arm_name][1000:])  # After warmup
        theoretical = arm.get_theoretical_optimal_latency()
        
        # Ideal load distribution based on capacity
        ideal_pct = (arm.capacity / total_theoretical_capacity) * 100
        
        print(f"{arm_name:<12} {arm.base_latency:<8.3f} {arm.capacity:<10.0f} "
              f"{count:>5d} ({pct:>5.1f}%) {avg_latency:<12.3f} {avg_load:<10.1f}")
        print(f"  └─ Ideal%: {ideal_pct:5.1f}%  Theoretical optimal latency: {theoretical:.3f}")
    
    # Final probabilities from Exp3
    print(f"\nFinal Exp3 Probabilities:")
    final_probs = bandit.get_probabilities()
    for arm_name in arm_names:
        prob = final_probs.get(arm_name, 0)
        ideal_pct = (arms[arm_name].capacity / total_theoretical_capacity)
        print(f"  {arm_name}: {prob*100:5.2f}% (ideal: {ideal_pct*100:5.2f}%)")
    
    # Overall performance
    avg_latency = sum(history['latencies']) / len(history['latencies'])
    recent_avg_latency = np.mean(history['latencies'][-1000:])
    
    # Calculate theoretical optimal average latency
    theoretical_avg = sum(
        arms[arm].get_theoretical_optimal_latency() * (arms[arm].capacity / total_theoretical_capacity)
        for arm in arm_names
    )
    
    print(f"\nOverall Performance:")
    print(f"  Average latency (all time): {avg_latency:.4f}")
    print(f"  Average latency (recent 1000): {recent_avg_latency:.4f}")
    print(f"  Theoretical optimal (if perfectly balanced): {theoretical_avg:.4f}")
    print(f"  Difference from optimal: {recent_avg_latency - theoretical_avg:.4f} "
          f"({(recent_avg_latency/theoretical_avg - 1)*100:+.1f}%)")
    
    # Check load balance
    print(f"\nLoad Balance Analysis (recent 1000 requests):")
    recent_counts = Counter(selections[-1000:])
    for arm_name, arm in arms.items():
        count = recent_counts.get(arm_name, 0)
        ideal_count = (arm.capacity / total_theoretical_capacity) * 1000
        deviation = count - ideal_count
        print(f"  {arm_name}: {count:4d} requests (ideal: {ideal_count:6.1f}, "
              f"deviation: {deviation:+6.1f})")


def plot_load_balancing_results(history, arms, output_file='exp3_load_balancing.png'):
    """Visualize load balancing behavior."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    arm_names = list(arms.keys())
    iterations = range(len(history['selections']))
    
    colors = {arm: plt.cm.tab10(i) for i, arm in enumerate(arm_names)}
    
    # 1. Probability evolution
    ax1 = fig.add_subplot(gs[0, 0])
    for arm_name in arm_names:
        probs = history['probabilities'][arm_name]
        ax1.plot(probs, label=f"{arm_name}", color=colors[arm_name], linewidth=2, alpha=0.8)
    
    # Add ideal capacity-based distribution
    total_cap = sum(arm.capacity for arm in arms.values())
    for arm_name in arm_names:
        ideal = arms[arm_name].capacity / total_cap
        ax1.axhline(y=ideal, color=colors[arm_name], linestyle='--', alpha=0.3, linewidth=1)
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Selection Probability')
    ax1.set_title('Arm Selection Probabilities (dashed = ideal capacity ratio)', fontweight='bold')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(alpha=0.3)
    
    # 2. Load over time
    ax2 = fig.add_subplot(gs[0, 1])
    for arm_name in arm_names:
        loads = history['loads'][arm_name]
        # Smooth with rolling average
        window = 50
        if len(loads) >= window:
            smoothed = np.convolve(loads, np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(loads)), smoothed, 
                    label=f"{arm_name} (cap={arms[arm_name].capacity:.0f})",
                    color=colors[arm_name], linewidth=2)
        
        # Show capacity as horizontal line
        ax2.axhline(y=arms[arm_name].capacity, color=colors[arm_name], 
                   linestyle=':', alpha=0.3, linewidth=1)
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Load (recent requests)')
    ax2.set_title('Arm Load Over Time (smoothed, dotted = capacity)', fontweight='bold')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(alpha=0.3)
    
    # 3. Latency over time (rolling average)
    ax3 = fig.add_subplot(gs[1, 0])
    window_size = 100
    rolling_avg = []
    for i in range(len(history['latencies'])):
        start = max(0, i - window_size + 1)
        rolling_avg.append(np.mean(history['latencies'][start:i+1]))
    
    ax3.plot(rolling_avg, color='blue', linewidth=2, label='Observed latency')
    
    # Calculate theoretical optimal
    total_cap = sum(arm.capacity for arm in arms.values())
    theoretical_optimal = sum(
        arms[arm].get_theoretical_optimal_latency() * (arms[arm].capacity / total_cap)
        for arm in arm_names
    )
    ax3.axhline(y=theoretical_optimal, color='green', linestyle='--',
               linewidth=2, label=f'Theoretical optimal ({theoretical_optimal:.3f})')
    
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Average Latency')
    ax3.set_title(f'Average Latency Over Time (window={window_size})', fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(alpha=0.3)
    
    # 4. Cumulative selections by arm
    ax4 = fig.add_subplot(gs[1, 1])
    cumulative_counts = {arm: [] for arm in arm_names}
    for arm in arm_names:
        count = 0
        for selection in history['selections']:
            if selection == arm:
                count += 1
            cumulative_counts[arm].append(count)
    
    for arm_name in arm_names:
        ax4.plot(cumulative_counts[arm_name], label=f"{arm_name}",
                color=colors[arm_name], linewidth=2)
        
        # Ideal line based on capacity
        ideal_rate = arms[arm_name].capacity / total_cap
        ideal_line = [i * ideal_rate for i in range(len(history['selections']))]
        ax4.plot(ideal_line, color=colors[arm_name], linestyle='--', alpha=0.3, linewidth=1)
    
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Cumulative Selections')
    ax4.set_title('Cumulative Selections (dashed = ideal)', fontweight='bold')
    ax4.legend(loc='best', fontsize=8)
    ax4.grid(alpha=0.3)
    
    # 5. Latency distribution by arm (box plot)
    ax5 = fig.add_subplot(gs[2, 0])
    latencies_by_arm = {arm: [] for arm in arm_names}
    for selection, latency in zip(history['selections'], history['latencies']):
        latencies_by_arm[selection].append(latency)
    
    data = [latencies_by_arm[arm] for arm in arm_names if latencies_by_arm[arm]]
    labels = [f"{arm}\n(base={arms[arm].base_latency:.2f})" for arm in arm_names if latencies_by_arm[arm]]
    
    bp = ax5.boxplot(data, labels=labels, patch_artist=True)
    for patch, arm_name in zip(bp['boxes'], [arm for arm in arm_names if latencies_by_arm[arm]]):
        patch.set_facecolor(colors[arm_name])
        patch.set_alpha(0.6)
    
    ax5.set_ylabel('Latency')
    ax5.set_title('Observed Latency Distribution by Arm', fontweight='bold')
    ax5.grid(alpha=0.3, axis='y')
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # 6. Selection distribution pie chart (recent)
    ax6 = fig.add_subplot(gs[2, 1])
    recent_selections = history['selections'][-1000:]
    recent_counts = Counter(recent_selections)
    
    sizes = [recent_counts.get(arm, 0) for arm in arm_names]
    pie_colors = [colors[arm] for arm in arm_names]
    labels_with_capacity = [f"{arm}\n(cap={arms[arm].capacity:.0f})" for arm in arm_names]
    
    ax6.pie(sizes, labels=labels_with_capacity, colors=pie_colors, autopct='%1.1f%%',
           startangle=90)
    ax6.set_title('Selection Distribution (Last 1000)', fontweight='bold')
    
    # 7. Utilization over time
    ax7 = fig.add_subplot(gs[3, :])
    for arm_name in arm_names:
        loads = history['loads'][arm_name]
        utilization = [load / arms[arm_name].capacity for load in loads]
        # Smooth
        window = 50
        if len(utilization) >= window:
            smoothed = np.convolve(utilization, np.ones(window)/window, mode='valid')
            ax7.plot(range(window-1, len(utilization)), smoothed, 
                    label=f"{arm_name}", color=colors[arm_name], linewidth=2)
    
    ax7.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, 
               linewidth=2, label='100% capacity (overload)')
    ax7.axhline(y=0.7, color='green', linestyle='--', alpha=0.3, 
               linewidth=1, label='70% (optimal)')
    
    ax7.set_xlabel('Iteration')
    ax7.set_ylabel('Utilization (load / capacity)')
    ax7.set_title('Arm Utilization Over Time (smoothed)', fontweight='bold')
    ax7.legend(loc='best', fontsize=8)
    ax7.grid(alpha=0.3)
    ax7.set_ylim(bottom=0)
    
    plt.suptitle('Exp3 Load Balancing with Congestion', fontsize=16, fontweight='bold')
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")


def test_scenario_balanced_arms():
    """Arms with different base latencies but similar capacities."""
    print("\n" + "="*70)
    print("SCENARIO 1: Balanced Arms (similar capacities)")
    print("="*70)
    
    arms = {
        'fast': CongestionArmSimulator('fast', base_latency=0.1, capacity=50, congestion_factor=2.0),
        'medium': CongestionArmSimulator('medium', base_latency=0.2, capacity=50, congestion_factor=2.0),
        'slow': CongestionArmSimulator('slow', base_latency=0.3, capacity=50, congestion_factor=2.0),
    }
    
    history, bandit = test_load_balancing(arms, num_samples=5000, gamma=0.15, L_max=2.0)
    analyze_load_balancing_results(history, arms, bandit)
    plot_load_balancing_results(history, arms, 'exp3_balanced_arms.png')


def test_scenario_heterogeneous_capacity():
    """Arms with different capacities - optimal strategy is capacity-proportional."""
    print("\n" + "="*70)
    print("SCENARIO 2: Heterogeneous Capacity")
    print("Fast but low capacity vs slow but high capacity")
    print("="*70)
    
    arms = {
        'fast_small': CongestionArmSimulator('fast_small', base_latency=0.1, capacity=30, congestion_factor=3.0),
        'medium': CongestionArmSimulator('medium', base_latency=0.2, capacity=60, congestion_factor=2.0),
        'slow_large': CongestionArmSimulator('slow_large', base_latency=0.3, capacity=90, congestion_factor=1.5),
    }
    
    history, bandit = test_load_balancing(arms, num_samples=8000, gamma=0.2, L_max=2.0)
    analyze_load_balancing_results(history, arms, bandit)
    plot_load_balancing_results(history, arms, 'exp3_heterogeneous_capacity.png')


def test_scenario_many_arms():
    """Many arms with varying characteristics."""
    print("\n" + "="*70)
    print("SCENARIO 3: Many Arms with Varying Characteristics")
    print("="*70)
    
    arms = {
        'ultra_fast': CongestionArmSimulator('ultra_fast', base_latency=0.05, capacity=20, congestion_factor=4.0),
        'fast1': CongestionArmSimulator('fast1', base_latency=0.1, capacity=40, congestion_factor=2.5),
        'fast2': CongestionArmSimulator('fast2', base_latency=0.12, capacity=45, congestion_factor=2.0),
        'medium': CongestionArmSimulator('medium', base_latency=0.2, capacity=60, congestion_factor=1.8),
        'slow_large': CongestionArmSimulator('slow_large', base_latency=0.3, capacity=80, congestion_factor=1.5),
    }
    
    history, bandit = test_load_balancing(arms, num_samples=10000, gamma=0.2, L_max=2.0)
    analyze_load_balancing_results(history, arms, bandit)
    plot_load_balancing_results(history, arms, 'exp3_many_arms_congestion.png')


def main():
    print("="*70)
    print("EXP3 LOAD BALANCING WITH CONGESTION TEST")
    print("="*70)
    print("\nThis test simulates realistic scenarios where:")
    print("- Arms have limited capacity")
    print("- Latency increases under load (congestion)")
    print("- Optimal strategy is load balancing, not just picking fastest")
    
    test_scenario_balanced_arms()
    test_scenario_heterogeneous_capacity()
    test_scenario_many_arms()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - exp3_balanced_arms.png")
    print("  - exp3_heterogeneous_capacity.png")
    print("  - exp3_many_arms_congestion.png")


if __name__ == "__main__":
    main()

