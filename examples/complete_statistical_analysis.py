"""
Complete Statistical Analysis Example

This example demonstrates the full capabilities of the
Advanced Statistical Analysis Toolkit.

Author: Gabriel Demetrios Lafis
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import pandas as pd
from r_integration import RStatisticalAnalyzer, monte_carlo_simulation
from statistical_visualizations import StatisticalVisualizer
import matplotlib.pyplot as plt

def main():
    """Run complete statistical analysis example."""
    print("="*70)
    print("Advanced Statistical Analysis Toolkit - Complete Example")
    print("="*70)
    
    # Initialize analyzer and visualizer
    try:
        analyzer = RStatisticalAnalyzer()
        print("✓ R integration initialized successfully")
    except ImportError as e:
        print(f"✗ Error initializing R integration: {e}")
        print("  Please ensure rpy2 is installed and R is available")
        return
    
    viz = StatisticalVisualizer()
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # =========================================================================
    # Example 1: T-Test Analysis
    # =========================================================================
    print("\n" + "="*70)
    print("Example 1: Independent Samples T-Test")
    print("="*70)
    
    # Generate sample data
    np.random.seed(42)
    treatment_group = np.random.normal(100, 15, 50)
    control_group = np.random.normal(95, 15, 50)
    
    print(f"\nTreatment group: mean={treatment_group.mean():.2f}, std={treatment_group.std():.2f}")
    print(f"Control group: mean={control_group.mean():.2f}, std={control_group.std():.2f}")
    
    # Perform t-test
    t_result = analyzer.t_test(treatment_group, control_group, paired=False)
    
    print(f"\nT-Test Results:")
    print(f"  t-statistic: {t_result['statistic']:.4f}")
    print(f"  p-value: {t_result['p_value']:.4f}")
    print(f"  95% CI: [{t_result['conf_int'][0]:.2f}, {t_result['conf_int'][1]:.2f}]")
    print(f"  Mean difference: {t_result['mean_diff']:.2f}")
    
    if t_result['p_value'] < 0.05:
        print("  → Significant difference detected (p < 0.05)")
    else:
        print("  → No significant difference (p ≥ 0.05)")
    
    # =========================================================================
    # Example 2: ANOVA Analysis
    # =========================================================================
    print("\n" + "="*70)
    print("Example 2: One-Way ANOVA")
    print("="*70)
    
    # Generate sample data with three groups
    group_a = np.random.normal(100, 15, 30)
    group_b = np.random.normal(105, 15, 30)
    group_c = np.random.normal(110, 15, 30)
    
    # Create DataFrame
    anova_data = pd.DataFrame({
        'value': np.concatenate([group_a, group_b, group_c]),
        'group': ['A']*30 + ['B']*30 + ['C']*30
    })
    
    print(f"\nGroup means:")
    print(anova_data.groupby('group')['value'].mean())
    
    # Perform ANOVA
    anova_result = analyzer.anova(anova_data, 'value ~ group')
    print(f"\n✓ ANOVA completed")
    
    # Visualize groups
    fig = viz.plot_boxplot_comparison(
        anova_data, 'group', 'value',
        title='Comparison of Groups (ANOVA)',
        save_path='results/anova_comparison.png'
    )
    plt.close()
    print("✓ Boxplot saved to results/anova_comparison.png")
    
    # =========================================================================
    # Example 3: Chi-Square Test
    # =========================================================================
    print("\n" + "="*70)
    print("Example 3: Chi-Square Test of Independence")
    print("="*70)
    
    # Create contingency table
    contingency_table = np.array([[30, 20], [15, 35]])
    print("\nContingency table:")
    print(contingency_table)
    
    # Perform chi-square test
    chi_result = analyzer.chi_square_test(contingency_table)
    print(f"\nChi-Square Test Results:")
    print(f"  Chi-square statistic: {chi_result['statistic']:.4f}")
    print(f"  p-value: {chi_result['p_value']:.4f}")
    print(f"  Degrees of freedom: {chi_result['df']:.0f}")
    
    if chi_result['p_value'] < 0.05:
        print("  → Variables are significantly associated (p < 0.05)")
    else:
        print("  → No significant association (p ≥ 0.05)")
    
    # =========================================================================
    # Example 4: Normality Test
    # =========================================================================
    print("\n" + "="*70)
    print("Example 4: Normality Testing")
    print("="*70)
    
    # Test normal data
    normal_data = np.random.normal(50, 10, 100)
    norm_result = analyzer.test_normality(normal_data)
    
    print(f"\nNormality Test Results (Normal data):")
    print(f"  Shapiro-Wilk: W={norm_result['shapiro_statistic']:.4f}, p={norm_result['shapiro_p_value']:.4f}")
    print(f"  Kolmogorov-Smirnov: D={norm_result['ks_statistic']:.4f}, p={norm_result['ks_p_value']:.4f}")
    print(f"  → Data is {'normal' if norm_result['is_normal'] else 'not normal'}")
    
    # Visualize
    fig = viz.plot_distribution(
        normal_data,
        title='Distribution of Normal Data',
        save_path='results/normal_distribution.png'
    )
    plt.close()
    
    fig = viz.plot_qq(
        normal_data,
        title='Q-Q Plot for Normal Data',
        save_path='results/qq_plot_normal.png'
    )
    plt.close()
    print("✓ Plots saved to results/")
    
    # =========================================================================
    # Example 5: Bootstrap Confidence Intervals
    # =========================================================================
    print("\n" + "="*70)
    print("Example 5: Bootstrap Confidence Intervals")
    print("="*70)
    
    # Generate sample data
    sample_data = np.random.exponential(scale=20, size=100)
    
    print(f"\nOriginal data:")
    print(f"  Mean: {sample_data.mean():.2f}")
    print(f"  Median: {np.median(sample_data):.2f}")
    
    # Bootstrap for mean
    ci_mean = analyzer.bootstrap_ci(sample_data, statistic_func='mean', n_bootstrap=10000)
    print(f"\nBootstrap 95% CI for mean: [{ci_mean[0]:.2f}, {ci_mean[1]:.2f}]")
    
    # Bootstrap for median
    ci_median = analyzer.bootstrap_ci(sample_data, statistic_func='median', n_bootstrap=10000)
    print(f"Bootstrap 95% CI for median: [{ci_median[0]:.2f}, {ci_median[1]:.2f}]")
    
    # Visualize bootstrap distribution
    bootstrap_samples = []
    for _ in range(10000):
        resample = np.random.choice(sample_data, size=len(sample_data), replace=True)
        bootstrap_samples.append(np.mean(resample))
    bootstrap_samples = np.array(bootstrap_samples)
    
    fig = viz.plot_bootstrap_distribution(
        bootstrap_samples,
        sample_data.mean(),
        ci_mean,
        statistic_name='Mean',
        save_path='results/bootstrap_distribution.png'
    )
    plt.close()
    print("✓ Bootstrap plot saved to results/bootstrap_distribution.png")
    
    # =========================================================================
    # Example 6: Monte Carlo Simulation
    # =========================================================================
    print("\n" + "="*70)
    print("Example 6: Monte Carlo Simulation")
    print("="*70)
    
    # Simulate sampling from normal distribution
    print("\nRunning Monte Carlo simulation (10,000 iterations)...")
    mc_results = monte_carlo_simulation(
        n_simulations=10000,
        sample_size=50,
        distribution='normal',
        mean=100,
        std=15
    )
    
    print(f"\nSimulation Results:")
    print(f"  Mean of sample means: {mc_results.mean():.2f}")
    print(f"  Std of sample means: {mc_results.std():.2f}")
    print(f"  95% CI: [{np.percentile(mc_results, 2.5):.2f}, {np.percentile(mc_results, 97.5):.2f}]")
    
    # Visualize
    fig = viz.plot_distribution(
        mc_results,
        title='Monte Carlo Simulation: Distribution of Sample Means',
        save_path='results/monte_carlo_simulation.png'
    )
    plt.close()
    print("✓ Simulation plot saved to results/monte_carlo_simulation.png")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print("\nAll results and visualizations saved to 'results/' directory")
    print("\nGenerated files:")
    print("  - anova_comparison.png")
    print("  - normal_distribution.png")
    print("  - qq_plot_normal.png")
    print("  - bootstrap_distribution.png")
    print("  - monte_carlo_simulation.png")
    print("\n✓ Example completed successfully!")


if __name__ == "__main__":
    main()

