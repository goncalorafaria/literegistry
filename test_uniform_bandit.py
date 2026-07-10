#!/usr/bin/env python3
"""
Test script to verify the uniformity of UniformBandit distribution.
Creates a histogram showing the selection frequency of each arm.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Import directly from the bandit module
import importlib.util
spec = importlib.util.spec_from_file_location("bandit", "literegistry/bandit.py")
bandit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bandit_module)
UniformBandit = bandit_module.UniformBandit

def test_uniform_distribution(num_arms=5, num_samples=10000):
    """
    Test the UniformBandit with multiple arms and check distribution.
    
    Args:
        num_arms: Number of arms to test
        num_samples: Number of samples to draw
    """
    # Create active arms
    active_arms = [f"arm{i}" for i in range(num_arms)]
    
    # Initialize bandit
    bandit = UniformBandit()
    
    # Collect samples
    selections = []
    probabilities = []
    
    print(f"Testing UniformBandit with {num_arms} arms and {num_samples} samples...")
    print(f"Active arms: {active_arms}\n")
    
    for i in range(num_samples):
        chosen_arm, prob = bandit.get_arm(active_arms, k=1)
        if chosen_arm:  # Handle empty returns
            selections.append(chosen_arm[0])
            probabilities.append(prob[0])
            
            # Update with neutral feedback (shouldn't affect uniform distribution)
            bandit.update(chosen_arm[0], success=True, latency=0.5)
    
    # Analyze results
    counts = Counter(selections)
    
    print("Selection counts:")
    for arm in active_arms:
        count = counts.get(arm, 0)
        percentage = (count / num_samples) * 100
        print(f"  {arm}: {count:6d} times ({percentage:5.2f}%)")
    
    print(f"\nExpected percentage per arm: {100/num_arms:.2f}%")
    
    # Calculate chi-square statistic
    expected_count = num_samples / num_arms
    chi_square = sum((counts.get(arm, 0) - expected_count)**2 / expected_count 
                     for arm in active_arms)
    print(f"Chi-square statistic: {chi_square:.4f}")
    print(f"(Lower values indicate more uniform distribution)")
    
    # Calculate standard deviation of counts
    count_values = [counts.get(arm, 0) for arm in active_arms]
    std_dev = np.std(count_values)
    print(f"Standard deviation of counts: {std_dev:.2f}")
    
    # Sample some probabilities reported
    print(f"\nSample of reported probabilities (first 10):")
    for i in range(min(10, len(probabilities))):
        print(f"  Sample {i+1}: {probabilities[i]:.6f}")
    
    return selections, probabilities, active_arms


def plot_histogram(selections, active_arms, num_samples):
    """Create and save a histogram of the selections."""
    counts = Counter(selections)
    
    # Prepare data for plotting
    arms = active_arms
    arm_counts = [counts.get(arm, 0) for arm in arms]
    expected_count = num_samples / len(active_arms)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Bar chart with counts
    x_pos = np.arange(len(arms))
    ax1.bar(x_pos, arm_counts, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axhline(y=expected_count, color='red', linestyle='--', 
                linewidth=2, label=f'Expected (uniform): {expected_count:.0f}')
    ax1.set_xlabel('Arm', fontsize=12)
    ax1.set_ylabel('Selection Count', fontsize=12)
    ax1.set_title(f'UniformBandit Selection Counts\n({num_samples:,} samples)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(arms, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Percentage deviation from expected
    percentages = [(count / num_samples * 100) for count in arm_counts]
    expected_pct = 100 / len(active_arms)
    deviations = [(pct - expected_pct) for pct in percentages]
    
    colors = ['green' if abs(d) < 0.5 else 'orange' if abs(d) < 1.0 else 'red' 
              for d in deviations]
    ax2.bar(x_pos, deviations, alpha=0.7, color=colors, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Arm', fontsize=12)
    ax2.set_ylabel('Deviation from Expected (%)', fontsize=12)
    ax2.set_title('Deviation from Uniform Distribution', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(arms, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'uniform_bandit_test.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nHistogram saved to: {output_file}")
    
    # Also try to show it
    try:
        plt.show()
    except:
        print("(Cannot display plot in this environment, but file is saved)")


def main():
    # Test with different numbers of arms
    test_cases = [
        (3, 10000),
        (5, 10000),
        (10, 10000),
    ]
    
    print("="*70)
    print("UNIFORM BANDIT DISTRIBUTION TEST")
    print("="*70 + "\n")
    
    for num_arms, num_samples in test_cases:
        print(f"\n{'='*70}")
        selections, probabilities, active_arms = test_uniform_distribution(
            num_arms=num_arms, 
            num_samples=num_samples
        )
        print(f"{'='*70}\n")
    
    # Create detailed visualization for 5 arms
    print("\nCreating detailed histogram for 5 arms...")
    selections, probabilities, active_arms = test_uniform_distribution(
        num_arms=5, 
        num_samples=10000
    )
    plot_histogram(selections, active_arms, 10000)


if __name__ == "__main__":
    main()

