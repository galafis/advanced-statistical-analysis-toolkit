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
        ax.plot
