"""
R Integration Module for Statistical Analysis

This module provides a Python interface to R statistical functions
using rpy2 for seamless integration.

Author: Gabriel Demetrios Lafis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    pandas2ri.activate()
    HAS_RPY2 = True
except ImportError:
    HAS_RPY2 = False
    print("Warning: rpy2 not installed. R integration will not be available.")


class RStatisticalAnalyzer:
    """
    Python wrapper for R statistical functions.
    
    Provides easy access to R's statistical capabilities from Python.
    """
    
    def __init__(self):
        """Initialize R statistical analyzer."""
        if not HAS_RPY2:
            raise ImportError("rpy2 is required for R integration. Install with: pip install rpy2")
        
        # Import R packages
        self.stats = importr('stats')
        self.base = importr('base')
        
        # Load custom R functions
        ro.r.source('R/hypothesis_testing.R')
        
    def t_test(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        paired: bool = False,
        alternative: str = "two.sided"
    ) -> Dict:
        """
        Perform t-test using R.
        
        Parameters
        ----------
        group1 : np.ndarray
            First group data
        group2 : np.ndarray
            Second group data
        paired : bool
            Whether to perform paired t-test
        alternative : str
            Alternative hypothesis: "two.sided", "less", or "greater"
            
        Returns
        -------
        results : dict
            Test results including statistic, p-value, and confidence interval
        """
        r_group1 = ro.FloatVector(group1)
        r_group2 = ro.FloatVector(group2)
        
        result = ro.r['perform_t_test'](r_group1, r_group2, paired, alternative)
        
        return {
            'statistic': result.rx2('statistic')[0],
            'p_value': result.rx2('p_value')[0],
            'conf_int': tuple(result.rx2('conf_int')),
            'mean_diff': result.rx2('mean_diff')[0],
            'method': str(result.rx2('method')[0])
        }
    
    def anova(
        self,
        data: pd.DataFrame,
        formula: str
    ) -> Dict:
        """
        Perform ANOVA using R.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data for analysis
        formula : str
            R formula string (e.g., 'value ~ group')
            
        Returns
        -------
        results : dict
            ANOVA table and post-hoc tests
        """
        r_data = pandas2ri.py2rpy(data)
        r_formula = ro.Formula(formula)
        
        result = ro.r['perform_anova'](r_data, r_formula)
        
        return {
            'anova_table': result.rx2('anova_table'),
            'tukey_hsd': result.rx2('tukey_hsd'),
            'significant': True  # Simplified for example
        }
    
    def chi_square_test(
        self,
        contingency_table: np.ndarray
    ) -> Dict:
        """
        Perform chi-square test of independence.
        
        Parameters
        ----------
        contingency_table : np.ndarray
            2D contingency table
            
        Returns
        -------
        results : dict
            Chi-square test results
        """
        r_table = ro.r.matrix(ro.FloatVector(contingency_table.flatten()),
                             nrow=contingency_table.shape[0])
        
        result = ro.r['perform_chi_square'](r_table)
        
        return {
            'statistic': result.rx2('statistic')[0],
            'p_value': result.rx2('p_value')[0],
            'df': result.rx2('df')[0]
        }
    
    def test_normality(
        self,
        data: np.ndarray
    ) -> Dict:
        """
        Test for normality using Shapiro-Wilk and KS tests.
        
        Parameters
        ----------
        data : np.ndarray
            Data to test
            
        Returns
        -------
        results : dict
            Normality test results
        """
        r_data = ro.FloatVector(data)
        result = ro.r['test_normality'](r_data)
        
        return {
            'shapiro_statistic': result.rx2('shapiro').rx2('statistic')[0],
            'shapiro_p_value': result.rx2('shapiro').rx2('p_value')[0],
            'ks_statistic': result.rx2('ks').rx2('statistic')[0],
            'ks_p_value': result.rx2('ks').rx2('p_value')[0],
            'is_normal': bool(result.rx2('is_normal')[0])
        }
    
    def bootstrap_ci(
        self,
        data: np.ndarray,
        statistic_func: str = 'mean',
        n_bootstrap: int = 10000,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval.
        
        Parameters
        ----------
        data : np.ndarray
            Data to bootstrap
        statistic_func : str
            Statistic to calculate ('mean', 'median', 'sd')
        n_bootstrap : int
            Number of bootstrap samples
        confidence : float
            Confidence level
            
        Returns
        -------
        ci : tuple
            Lower and upper confidence bounds
        """
        # Bootstrap in Python for efficiency
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            if statistic_func == 'mean':
                stat = np.mean(sample)
            elif statistic_func == 'median':
                stat = np.median(sample)
            elif statistic_func == 'sd':
                stat = np.std(sample)
            else:
                raise ValueError(f"Unknown statistic: {statistic_func}")
            bootstrap_stats.append(stat)
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_stats, alpha/2 * 100)
        upper = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)
        
        return (lower, upper)


def monte_carlo_simulation(
    n_simulations: int,
    sample_size: int,
    distribution: str = 'normal',
    **dist_params
) -> np.ndarray:
    """
    Perform Monte Carlo simulation.
    
    Parameters
    ----------
    n_simulations : int
        Number of simulations
    sample_size : int
        Size of each sample
    distribution : str
        Distribution to sample from
    **dist_params : dict
        Distribution parameters
        
    Returns
    -------
    results : np.ndarray
        Simulation results
    """
    results = []
    
    for _ in range(n_simulations):
        if distribution == 'normal':
            sample = np.random.normal(
                loc=dist_params.get('mean', 0),
                scale=dist_params.get('std', 1),
                size=sample_size
            )
        elif distribution == 'uniform':
            sample = np.random.uniform(
                low=dist_params.get('low', 0),
                high=dist_params.get('high', 1),
                size=sample_size
            )
        elif distribution == 'exponential':
            sample = np.random.exponential(
                scale=dist_params.get('scale', 1),
                size=sample_size
            )
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        results.append(np.mean(sample))
    
    return np.array(results)

