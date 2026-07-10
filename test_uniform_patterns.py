#!/usr/bin/env python3
"""
Test script to check for sequential patterns and autocorrelation in UniformBandit.
This tests if there are any non-random patterns in the sequence of selections.
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
UniformBandit = bandit_module.UniformBandit


def test_sequential_patterns(num_arms=5, num_samples=10000):
    """
    Test for sequential patterns - does selecting arm X make arm Y more likely next?
    """
    active_arms = [f"arm{i}" for i in range(num_arms)]
    bandit = UniformBandit()
    
    selections = []
    
    print(f"Testing sequential patterns with {num_arms} arms and {num_samples} samples...")
    
    for i in range(num_samples):
        chosen_arm, prob = bandit.get_arm(active_arms, k=1)
        if chosen_arm:
            selections.append(chosen_arm[0])
            bandit.update(chosen_arm[0], success=True, latency=0.5)
    
    # Build transition matrix
    transitions = defaultdict(lambda: defaultdict(int))
    for i in range(len(selections) - 1):
        current = selections[i]
        next_arm = selections[i + 1]
        transitions[current][next_arm] += 1
    
    # Calculate transition probabilities
    print("\nTransition Probabilities (Current -> Next):")
    print("=" * 70)
    
    transition_matrix = np.zeros((num_arms, num_arms))
    
    for i, current_arm in enumerate(active_arms):
        total_from_current = sum(transitions[current_arm].values())
        if total_from_current > 0:
            print(f"\nFrom {current_arm}:")
            for j, next_arm in enumerate(active_arms):
                count = transitions[current_arm][next_arm]
                prob = count / total_from_current
                transition_matrix[i][j] = prob
                expected = 1.0 / num_arms
                deviation = (prob - expected) * 100
                marker = "⚠" if abs(deviation) > 2 else "✓"
                print(f"  -> {next_arm}: {prob*100:5.2f}% ({count:4d} times) "
                      f"[Expected: {expected*100:5.2f}%, Deviation: {deviation:+5.2f}%] {marker}")
    
    # Calculate autocorrelation
    print("\n" + "=" * 70)
    print("Autocorrelation Analysis:")
    print("=" * 70)
    
    # Convert arms to numerical indices
    arm_to_idx = {arm: i for i, arm in enumerate(active_arms)}
    selection_indices = [arm_to_idx[arm] for arm in selections]
    
    lags = [1, 2, 3, 5, 10, 20, 50]
    for lag in lags:
        if lag < len(selection_indices):
            corr = np.corrcoef(selection_indices[:-lag], selection_indices[lag:])[0, 1]
            print(f"Lag {lag:3d}: correlation = {corr:+.6f}")
    
    # Test for runs (consecutive selections of same arm)
    print("\n" + "=" * 70)
    print("Run Analysis (consecutive selections of same arm):")
    print("=" * 70)
    
    runs = []
    current_run_length = 1
    for i in range(1, len(selections)):
        if selections[i] == selections[i-1]:
            current_run_length += 1
        else:
            runs.append(current_run_length)
            current_run_length = 1
    runs.append(current_run_length)
    
    run_counter = Counter(runs)
    print(f"\nRun length distribution:")
    for length in sorted(run_counter.keys())[:10]:  # Show first 10
        count = run_counter[length]
        percentage = (count / len(runs)) * 100
        # For uniform random, P(run of length k) = (1/n)^(k-1) * (n-1)/n
        expected_prob = (1/num_arms)**(length-1) * (num_arms-1)/num_arms
        expected_count = expected_prob * len(runs)
        print(f"  Length {length}: {count:5d} runs ({percentage:5.2f}%) "
              f"[Expected: {expected_count:5.0f} runs]")
    
    max_run = max(runs)
    avg_run = np.mean(runs)
    print(f"\nMax run length: {max_run}")
    print(f"Average run length: {avg_run:.2f} (Expected for uniform: {num_arms/(num_arms-1):.2f})")
    
    return transition_matrix, selections, active_arms


def plot_transition_matrix(transition_matrix, active_arms):
    """Plot the transition matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(transition_matrix, cmap='RdYlGn', vmin=0, vmax=max(0.3, transition_matrix.max()))
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(active_arms)))
    ax.set_yticks(np.arange(len(active_arms)))
    ax.set_xticklabels(active_arms)
    ax.set_yticklabels(active_arms)
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Transition Probability", rotation=-90, va="bottom")
    
    # Add text annotations
    expected = 1.0 / len(active_arms)
    for i in range(len(active_arms)):
        for j in range(len(active_arms)):
            value = transition_matrix[i, j]
            # Color text based on deviation from expected
            color = "black" if abs(value - expected) < 0.02 else "red"
            text = ax.text(j, i, f"{value*100:.1f}%",
                          ha="center", va="center", color=color, fontsize=8)
    
    ax.set_title(f"Transition Matrix for UniformBandit\n(Expected: {expected*100:.1f}% for all transitions)")
    ax.set_xlabel("Next Arm")
    ax.set_ylabel("Current Arm")
    
    plt.tight_layout()
    plt.savefig('uniform_bandit_transitions.png', dpi=150, bbox_inches='tight')
    print(f"\nTransition matrix saved to: uniform_bandit_transitions.png")


def test_dynamic_arms(num_samples=5000):
    """Test behavior when arms are dynamically added/removed."""
    print("\n" + "=" * 70)
    print("Testing Dynamic Arm Addition/Removal:")
    print("=" * 70)
    
    bandit = UniformBandit()
    selections = []
    
    # Start with 3 arms
    active_arms = ["arm0", "arm1", "arm2"]
    
    for i in range(num_samples):
        # At iteration 1000, add two more arms
        if i == 1000:
            active_arms.append("arm3")
            active_arms.append("arm4")
            print(f"\n[Iteration {i}] Added arm3 and arm4")
        
        # At iteration 3000, remove arm1
        if i == 3000:
            active_arms.remove("arm1")
            print(f"[Iteration {i}] Removed arm1")
        
        chosen_arm, prob = bandit.get_arm(active_arms, k=1)
        if chosen_arm:
            selections.append((i, chosen_arm[0], prob[0]))
            bandit.update(chosen_arm[0], success=True, latency=0.5)
    
    # Analyze selections in each phase
    phase1 = [s for s in selections if s[0] < 1000]
    phase2 = [s for s in selections if 1000 <= s[0] < 3000]
    phase3 = [s for s in selections if s[0] >= 3000]
    
    print("\n\nPhase 1 (0-999): 3 arms [arm0, arm1, arm2]")
    counts1 = Counter([s[1] for s in phase1])
    for arm in ["arm0", "arm1", "arm2"]:
        count = counts1.get(arm, 0)
        print(f"  {arm}: {count:4d} ({count/len(phase1)*100:5.2f}%) [Expected: 33.33%]")
    
    print("\nPhase 2 (1000-2999): 5 arms [arm0, arm1, arm2, arm3, arm4]")
    counts2 = Counter([s[1] for s in phase2])
    for arm in ["arm0", "arm1", "arm2", "arm3", "arm4"]:
        count = counts2.get(arm, 0)
        print(f"  {arm}: {count:4d} ({count/len(phase2)*100:5.2f}%) [Expected: 20.00%]")
    
    print("\nPhase 3 (3000-4999): 4 arms [arm0, arm2, arm3, arm4]")
    counts3 = Counter([s[1] for s in phase3])
    for arm in ["arm0", "arm2", "arm3", "arm4"]:
        count = counts3.get(arm, 0)
        print(f"  {arm}: {count:4d} ({count/len(phase3)*100:5.2f}%) [Expected: 25.00%]")


def main():
    print("="*70)
    print("UNIFORM BANDIT PATTERN ANALYSIS")
    print("="*70 + "\n")
    
    # Test sequential patterns
    transition_matrix, selections, active_arms = test_sequential_patterns(
        num_arms=5, 
        num_samples=10000
    )
    
    # Plot transition matrix
    plot_transition_matrix(transition_matrix, active_arms)
    
    # Test dynamic arm changes
    test_dynamic_arms(num_samples=5000)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

