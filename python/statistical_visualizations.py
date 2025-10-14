"""
Statistical Visualizations Module

Provides comprehensive visualization functions for statistical analysis.

Author: Gabriel Demetrios Lafis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Optional, Tuple, List


class StatisticalVisualizer:
    """
    Class for creating statistical visualizations.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        Initialize the visualizer.
        
        Parameters
        ----------
        style : str
            Matplotlib style to use
        """
        plt.style.use(style)
        sns.set_palette("husl")
        
    def plot_distribution(
        self,
        data: np.ndarray,
        title: str = "Distribution Plot",
        bins: int = 30,
        show_normal: bool = True,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot distribution with histogram and KDE.
        
        Parameters
        ----------
        data : np.ndarray
            Data to plot
        title : str
            Plot title
        bins : int
            Number of histogram bins
        show_normal : bool
            Whether to overlay normal distribution
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histogram and KDE
        ax.hist(data, bins=bins, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        
        # KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 100)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        
        # Normal distribution overlay
        if show_normal:
            mu, sigma = data.mean(), data.std()
            normal_curve = stats.norm.pdf(x_range, mu, sigma)
            ax.plot(x_range, normal_curve, 'g--', linewidth=2, label='Normal Distribution')
        
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_qq(
        self,
        data: np.ndarray,
        title: str = "Q-Q Plot",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create Q-Q plot to assess normality.
        
        Parameters
        ----------
        data : np.ndarray
            Data to plot
        title : str
            Plot title
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        stats.probplot(data, dist="norm", plot=ax)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_residuals(
        self,
        fitted: np.ndarray,
        residuals: np.ndarray,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create residual plots for regression diagnostics.
        
        Parameters
        ----------
        fitted : np.ndarray
            Fitted values
        residuals : np.ndarray
            Residuals
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Residuals vs Fitted
        axes[0, 0].scatter(fitted, residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Normal Q-Q')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Scale-Location plot
        standardized_residuals = np.sqrt(np.abs(residuals / np.std(residuals)))
        axes[1, 0].scatter(fitted, standardized_residuals, alpha=0.5)
        axes[1, 0].set_xlabel('Fitted Values')
        axes[1, 0].set_ylabel('√|Standardized Residuals|')
        axes[1, 0].set_title('Scale-Location')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[1, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Histogram of Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_correlation_matrix(
        self,
        data: pd.DataFrame,
        method: str = 'pearson',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create correlation matrix heatmap.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data frame with numeric variables
        method : str
            Correlation method ('pearson', 'spearman', 'kendall')
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        """
        # Calculate correlation matrix
        corr_matrix = data.corr(method=method)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   fmt='.2f', ax=ax)
        ax.set_title(f'Correlation Matrix ({method.capitalize()})')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_pca_results(
        self,
        scores: np.ndarray,
        variance_explained: np.ndarray,
        loadings: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive PCA visualization.
        
        Parameters
        ----------
        scores : np.ndarray
            PCA scores (transformed data)
        variance_explained : np.ndarray
            Variance explained by each component
        loadings : np.ndarray, optional
            Component loadings
        feature_names : List[str], optional
            Names of original features
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Scree plot
        cumulative_variance = np.cumsum(variance_explained)
        axes[0, 0].bar(range(1, len(variance_explained) + 1), variance_explained,
                      alpha=0.7, label='Individual')
        axes[0, 0].plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
                       'ro-', linewidth=2, label='Cumulative')
        axes[0, 0].axhline(y=0.95, color='g', linestyle='--', label='95% threshold')
        axes[0, 0].set_xlabel('Principal Component')
        axes[0, 0].set_ylabel('Variance Explained')
        axes[0, 0].set_title('Scree Plot')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # PC1 vs PC2 scatter plot
        axes[0, 1].scatter(scores[:, 0], scores[:, 1], alpha=0.5, s=50)
        axes[0, 1].set_xlabel(f'PC1 ({variance_explained[0]*100:.1f}%)')
        axes[0, 1].set_ylabel(f'PC2 ({variance_explained[1]*100:.1f}%)')
        axes[0, 1].set_title('PCA Biplot: PC1 vs PC2')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Loadings heatmap
        if loadings is not None:
            n_components = min(5, loadings.shape[1])
            loading_subset = loadings[:, :n_components]
            
            if feature_names is not None:
                loading_df = pd.DataFrame(
                    loading_subset,
                    index=feature_names,
                    columns=[f'PC{i+1}' for i in range(n_components)]
                )
            else:
                loading_df = pd.DataFrame(
                    loading_subset,
                    columns=[f'PC{i+1}' for i in range(n_components)]
                )
            
            sns.heatmap(loading_df, annot=True, cmap='coolwarm', center=0,
                       ax=axes[1, 0], cbar_kws={'label': 'Loading'}, fmt='.2f')
            axes[1, 0].set_title('Component Loadings')
        
        # Cumulative variance plot
        axes[1, 1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
                       'bo-', linewidth=2, markersize=8)
        axes[1, 1].fill_between(range(1, len(cumulative_variance) + 1),
                                cumulative_variance, alpha=0.3)
        axes[1, 1].set_xlabel('Number of Components')
        axes[1, 1].set_ylabel('Cumulative Variance Explained')
        axes[1, 1].set_title('Cumulative Variance Explained')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_boxplot_comparison(
        self,
        data: pd.DataFrame,
        x_var: str,
        y_var: str,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create boxplot for group comparison.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data frame
        x_var : str
            Grouping variable
        y_var : str
            Numeric variable
        title : str, optional
            Plot title
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.boxplot(data=data, x=x_var, y=y_var, ax=ax)
        sns.stripplot(data=data, x=x_var, y=y_var, color='black', 
                     alpha=0.3, size=3, ax=ax)
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'{y_var} by {x_var}')
        
        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_bootstrap_distribution(
        self,
        bootstrap_samples: np.ndarray,
        original_statistic: float,
        confidence_interval: Tuple[float, float],
        statistic_name: str = "Statistic",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot bootstrap distribution with confidence interval.
        
        Parameters
        ----------
        bootstrap_samples : np.ndarray
            Bootstrap sample statistics
        original_statistic : float
            Original statistic value
        confidence_interval : Tuple[float, float]
            Lower and upper CI bounds
        statistic_name : str
            Name of the statistic
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histogram
        ax.hist(bootstrap_samples, bins=50, alpha=0.7, color='skyblue',
               edgecolor='black', density=True)
        
        # Original statistic
        ax.axvline(original_statistic, color='red', linestyle='--',
                  linewidth=2, label=f'Original {statistic_name}')
        
        # Confidence interval
        ax.axvline(confidence_interval[0], color='green', linestyle='--',
                  linewidth=2, label='95% CI')
        ax.axvline(confidence_interval[1], color='green', linestyle='--',
                  linewidth=2)
        
        # KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(bootstrap_samples)
        x_range = np.linspace(bootstrap_samples.min(), bootstrap_samples.max(), 100)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2, alpha=0.7)
        
        ax.set_xlabel(statistic_name)
        ax.set_ylabel('Density')
        ax.set_title(f'Bootstrap Distribution of {statistic_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def create_diagnostic_report(
    data: pd.DataFrame,
    target_variable: str,
    save_dir: str = 'results'
) -> None:
    """
    Create comprehensive diagnostic report with multiple visualizations.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data frame for analysis
    target_variable : str
        Name of target variable
    save_dir : str
        Directory to save plots
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    viz = StatisticalVisualizer()
    
    # Distribution of target variable
    viz.plot_distribution(
        data[target_variable].values,
        title=f'Distribution of {target_variable}',
        save_path=f'{save_dir}/target_distribution.png'
    )
    
    # Q-Q plot
    viz.plot_qq(
        data[target_variable].values,
        title=f'Q-Q Plot for {target_variable}',
        save_path=f'{save_dir}/qq_plot.png'
    )
    
    # Correlation matrix
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        viz.plot_correlation_matrix(
            data[numeric_cols],
            save_path=f'{save_dir}/correlation_matrix.png'
        )
    
    print(f"✓ Diagnostic plots saved to {save_dir}/")

