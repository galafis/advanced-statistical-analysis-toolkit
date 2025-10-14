"""
Complete Statistical Analysis Example

Demonstrates hypothesis testing, regression, and Monte Carlo simulation.

Author: Gabriel Demetrios Lafis
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import pandas as pd

def main():
    """Run simple statistical analysis examples."""
    print("Simple Statistical Analysis Example")
    print("="*50)
    
    # Example: Basic statistics
    data = np.random.normal(100, 15, 100)
    
    print(f"\nBasic Statistics:")
    print(f"  Mean: {np.mean(data):.2f}")
    print(f"  Median: {np.median(data):.2f}")
    print(f"  Std Dev: {np.std(data):.2f}")
    print(f"  Min: {np.min(data):.2f}")
    print(f"  Max: {np.max(data):.2f}")
    
    # Example: Confidence interval
    from scipy import stats
    ci = stats.t.interval(0.95, len(data)-1, 
                          loc=np.mean(data), 
                          scale=stats.sem(data))
    print(f"\n95% Confidence Interval: [{ci[0]:.2f}, {ci[1]:.2f}]")
    
    print("\n✓ Example completed!")

if __name__ == "__main__":
    main()
