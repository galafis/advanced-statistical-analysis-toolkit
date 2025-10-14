"""
Statistical Tests Module

Provides Python wrappers for common statistical tests.

Author: Gabriel Demetrios Lafis
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional, Union
import warnings


class StatisticalTests:
    """Collection of statistical test functions."""
    
    @staticmethod
    def t_test_independent(
        group1: np.ndarray,
        group2: np.ndarray,
        equal_var: bool = True,
        alternative: str = 'two-sided'
    ) -> Dict:
        """
        Perform independent samples t-test.
        
        Parameters
        ----------
        group1 : np.ndarray
            First group data
        group2 : np.ndarray
            Second group data
        equal_var : bool
            Assume equal variances (default: True)
        alternative : str
            Alternative hypothesis ('two-sided', 'less', 'greater')
            
        Returns
        -------
        results : dict
            Test results
        """
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var, alternative=alternative)
        
        # Calculate confidence interval
        n1, n2 = len(group1), len(group2)
        mean_diff = np.mean(group1) - np.mean(group2)
        
        if equal_var:
            pooled_se = np.sqrt(np.var(group1, ddof=1)/n1 + np.var(group2, ddof=1)/n2)
            df = n1 + n2 - 2
        else:
            pooled_se = np.sqrt(np.var(group1, ddof=1)/n1 + np.var(group2, ddof=1)/n2)
            # Welch-Satterthwaite df
            df = (np.var(group1, ddof=1)/n1 + np.var(group2, ddof=1)/n2)**2 / \
                 ((np.var(group1, ddof=1)/n1)**2/(n1-1) + (np.var(group2, ddof=1)/n2)**2/(n2-1))
        
        ci = stats.t.interval(0.95, df, loc=mean_diff, scale=pooled_se)
        
        return {
            'statistic': t_stat,
            'p_value': p_value,
            'df': df,
            'mean_diff': mean_diff,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def t_test_paired(
        pre: np.ndarray,
        post: np.ndarray,
        alternative: str = 'two-sided'
    ) -> Dict:
        """
        Perform paired samples t-test.
        
        Parameters
        ----------
        pre : np.ndarray
            Pre-treatment measurements
        post : np.ndarray
            Post-treatment measurements
        alternative : str
            Alternative hypothesis
            
        Returns
        -------
        results : dict
            Test results
        """
        t_stat, p_value = stats.ttest_rel(pre, post, alternative=alternative)
        
        differences = pre - post
        mean_diff = np.mean(differences)
        se_diff = stats.sem(differences)
        df = len(differences) - 1
        ci = stats.t.interval(0.95, df, loc=mean_diff, scale=se_diff)
        
        return {
            'statistic': t_stat,
            'p_value': p_value,
            'df': df,
            'mean_diff': mean_diff,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def one_way_anova(data: pd.DataFrame, value_col: str, group_col: str) -> Dict:
        """
        Perform one-way ANOVA.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data frame
        value_col : str
            Name of value column
        group_col : str
            Name of group column
            
        Returns
        -------
        results : dict
            ANOVA results
        """
        groups = [group[value_col].values for name, group in data.groupby(group_col)]
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Calculate eta squared (effect size)
        grand_mean = data[value_col].mean()
        ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in groups)
        ss_total = sum((x - grand_mean)**2 for group in groups for x in group)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        return {
            'statistic': f_stat,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def chi_square_test(contingency_table: Union[np.ndarray, pd.DataFrame]) -> Dict:
        """
        Perform chi-square test of independence.
        
        Parameters
        ----------
        contingency_table : np.ndarray or pd.DataFrame
            Contingency table
            
        Returns
        -------
        results : dict
            Test results
        """
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Calculate Cramér's V (effect size)
        n = np.sum(contingency_table)
        min_dim = min(contingency_table.shape[0], contingency_table.shape[1]) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
        
        return {
            'statistic': chi2,
            'p_value': p_value,
            'df': dof,
            'expected': expected,
            'cramers_v': cramers_v,
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def mann_whitney_u(group1: np.ndarray, group2: np.ndarray, alternative: str = 'two-sided') -> Dict:
        """
        Perform Mann-Whitney U test (non-parametric alternative to t-test).
        
        Parameters
        ----------
        group1 : np.ndarray
            First group data
        group2 : np.ndarray
            Second group data
        alternative : str
            Alternative hypothesis
            
        Returns
        -------
        results : dict
            Test results
        """
        u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative=alternative)
        
        # Calculate rank-biserial correlation (effect size)
        n1, n2 = len(group1), len(group2)
        rank_biserial = 1 - (2*u_stat) / (n1 * n2)
        
        return {
            'statistic': u_stat,
            'p_value': p_value,
            'rank_biserial': rank_biserial,
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def kruskal_wallis(data: pd.DataFrame, value_col: str, group_col: str) -> Dict:
        """
        Perform Kruskal-Wallis H test (non-parametric alternative to ANOVA).
        
        Parameters
        ----------
        data : pd.DataFrame
            Data frame
        value_col : str
            Name of value column
        group_col : str
            Name of group column
            
        Returns
        -------
        results : dict
            Test results
        """
        groups = [group[value_col].values for name, group in data.groupby(group_col)]
        h_stat, p_value = stats.kruskal(*groups)
        
        return {
            'statistic': h_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def shapiro_wilk(data: np.ndarray) -> Dict:
        """
        Perform Shapiro-Wilk test for normality.
        
        Parameters
        ----------
        data : np.ndarray
            Data to test
            
        Returns
        -------
        results : dict
            Test results
        """
        if len(data) > 5000:
            warnings.warn("Shapiro-Wilk test may not be reliable for n > 5000")
        
        w_stat, p_value = stats.shapiro(data)
        
        return {
            'statistic': w_stat,
            'p_value': p_value,
            'is_normal': p_value > 0.05,
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def kolmogorov_smirnov(data: np.ndarray, distribution: str = 'norm') -> Dict:
        """
        Perform Kolmogorov-Smirnov test.
        
        Parameters
        ----------
        data : np.ndarray
            Data to test
        distribution : str
            Distribution to test against ('norm', 'uniform', 'expon')
            
        Returns
        -------
        results : dict
            Test results
        """
        if distribution == 'norm':
            ks_stat, p_value = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data, ddof=1)))
        elif distribution == 'uniform':
            ks_stat, p_value = stats.kstest(data, 'uniform', args=(np.min(data), np.max(data) - np.min(data)))
        elif distribution == 'expon':
            ks_stat, p_value = stats.kstest(data, 'expon', args=(np.min(data), np.mean(data)))
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        return {
            'statistic': ks_stat,
            'p_value': p_value,
            'distribution': distribution,
            'fits_distribution': p_value > 0.05,
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def levene_test(data: pd.DataFrame, value_col: str, group_col: str) -> Dict:
        """
        Perform Levene's test for equality of variances.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data frame
        value_col : str
            Name of value column
        group_col : str
            Name of group column
            
        Returns
        -------
        results : dict
            Test results
        """
        groups = [group[value_col].values for name, group in data.groupby(group_col)]
        w_stat, p_value = stats.levene(*groups)
        
        return {
            'statistic': w_stat,
            'p_value': p_value,
            'equal_variances': p_value > 0.05,
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def pearson_correlation(x: np.ndarray, y: np.ndarray) -> Dict:
        """
        Calculate Pearson correlation coefficient.
        
        Parameters
        ----------
        x : np.ndarray
            First variable
        y : np.ndarray
            Second variable
            
        Returns
        -------
        results : dict
            Correlation results
        """
        r, p_value = stats.pearsonr(x, y)
        
        return {
            'coefficient': r,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def spearman_correlation(x: np.ndarray, y: np.ndarray) -> Dict:
        """
        Calculate Spearman rank correlation coefficient.
        
        Parameters
        ----------
        x : np.ndarray
            First variable
        y : np.ndarray
            Second variable
            
        Returns
        -------
        results : dict
            Correlation results
        """
        rho, p_value = stats.spearmanr(x, y)
        
        return {
            'coefficient': rho,
            'p_value': p_value,
            'significant': p_value < 0.05
        }


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.
    
    Parameters
    ----------
    group1 : np.ndarray
        First group
    group2 : np.ndarray
        Second group
        
    Returns
    -------
    d : float
        Cohen's d
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def power_analysis_t_test(
    effect_size: float,
    n: int,
    alpha: float = 0.05,
    alternative: str = 'two-sided'
) -> float:
    """
    Calculate statistical power for t-test.
    
    Parameters
    ----------
    effect_size : float
        Effect size (Cohen's d)
    n : int
        Sample size per group
    alpha : float
        Significance level
    alternative : str
        Alternative hypothesis
        
    Returns
    -------
    power : float
        Statistical power
    """
    from statsmodels.stats.power import TTestIndPower
    
    analysis = TTestIndPower()
    power = analysis.solve_power(
        effect_size=effect_size,
        nobs1=n,
        alpha=alpha,
        ratio=1.0,
        alternative=alternative
    )
    
    return power


def sample_size_t_test(
    effect_size: float,
    power: float = 0.80,
    alpha: float = 0.05,
    alternative: str = 'two-sided'
) -> int:
    """
    Calculate required sample size for t-test.
    
    Parameters
    ----------
    effect_size : float
        Expected effect size (Cohen's d)
    power : float
        Desired statistical power
    alpha : float
        Significance level
    alternative : str
        Alternative hypothesis
        
    Returns
    -------
    n : int
        Required sample size per group
    """
    from statsmodels.stats.power import TTestIndPower
    
    analysis = TTestIndPower()
    n = analysis.solve_power(
        effect_size=effect_size,
        power=power,
        alpha=alpha,
        ratio=1.0,
        alternative=alternative
    )
    
    return int(np.ceil(n))
