"""
Python Integration Tests
Advanced Statistical Analysis Toolkit

Author: Gabriel Demetrios Lafis
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

# Import modules to test
try:
    from r_integration import RStatisticalAnalyzer, monte_carlo_simulation
    from statistical_visualizations import StatisticalVisualizer, create_diagnostic_report
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    print(f"Warning: Could not import modules: {e}")


@pytest.mark.skipif(not MODULES_AVAILABLE, reason="Required modules not available")
class TestRIntegration:
    """Test R integration functionality."""
    
    def test_monte_carlo_simulation(self):
        """Test Monte Carlo simulation."""
        results = monte_carlo_simulation(
            n_simulations=100,
            sample_size=30,
            distribution='normal',
            mean=0,
            std=1
        )
        
        assert len(results) == 100
        assert np.abs(results.mean()) < 0.5  # Should be close to 0
        assert results.std() > 0
    
    def test_monte_carlo_uniform(self):
        """Test Monte Carlo with uniform distribution."""
        results = monte_carlo_simulation(
            n_simulations=100,
            sample_size=30,
            distribution='uniform',
            low=0,
            high=1
        )
        
        assert len(results) == 100
        assert 0 < results.mean() < 1
    
    def test_monte_carlo_exponential(self):
        """Test Monte Carlo with exponential distribution."""
        results = monte_carlo_simulation(
            n_simulations=100,
            sample_size=30,
            distribution='exponential',
            scale=2
        )
        
        assert len(results) == 100
        assert results.mean() > 0
    
    def test_monte_carlo_invalid_distribution(self):
        """Test Monte Carlo with invalid distribution."""
        with pytest.raises(ValueError):
            monte_carlo_simulation(
                n_simulations=10,
                sample_size=10,
                distribution='invalid'
            )


@pytest.mark.skipif(not MODULES_AVAILABLE, reason="Required modules not available")
class TestStatisticalVisualizations:
    """Test statistical visualization functionality."""
    
    def test_visualizer_init(self):
        """Test visualizer initialization."""
        viz = StatisticalVisualizer()
        assert viz is not None
    
    def test_plot_distribution(self):
        """Test distribution plotting."""
        viz = StatisticalVisualizer()
        data = np.random.normal(0, 1, 100)
        
        fig = viz.plot_distribution(data, title="Test Distribution")
        assert fig is not None
    
    def test_plot_qq(self):
        """Test Q-Q plot."""
        viz = StatisticalVisualizer()
        data = np.random.normal(0, 1, 100)
        
        fig = viz.plot_qq(data, title="Test Q-Q Plot")
        assert fig is not None
    
    def test_plot_correlation_matrix(self):
        """Test correlation matrix plotting."""
        viz = StatisticalVisualizer()
        data = pd.DataFrame({
            'a': np.random.randn(50),
            'b': np.random.randn(50),
            'c': np.random.randn(50)
        })
        
        fig = viz.plot_correlation_matrix(data)
        assert fig is not None
    
    def test_plot_boxplot_comparison(self):
        """Test boxplot comparison."""
        viz = StatisticalVisualizer()
        data = pd.DataFrame({
            'group': ['A']*25 + ['B']*25,
            'value': np.random.randn(50)
        })
        
        fig = viz.plot_boxplot_comparison(data, 'group', 'value')
        assert fig is not None


class TestDataGeneration:
    """Test basic data generation and statistics."""
    
    def test_numpy_available(self):
        """Test that numpy is available."""
        assert np is not None
    
    def test_pandas_available(self):
        """Test that pandas is available."""
        assert pd is not None
    
    def test_random_normal(self):
        """Test random normal generation."""
        data = np.random.normal(0, 1, 100)
        assert len(data) == 100
        assert -5 < data.mean() < 5
        assert 0 < data.std() < 5
    
    def test_dataframe_creation(self):
        """Test DataFrame creation."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        assert df.shape == (3, 2)
        assert list(df.columns) == ['a', 'b']


class TestStatisticalFunctions:
    """Test basic statistical functions."""
    
    def test_mean(self):
        """Test mean calculation."""
        data = np.array([1, 2, 3, 4, 5])
        assert np.mean(data) == 3.0
    
    def test_std(self):
        """Test standard deviation."""
        data = np.array([1, 2, 3, 4, 5])
        std = np.std(data, ddof=1)
        assert std > 0
    
    def test_percentile(self):
        """Test percentile calculation."""
        data = np.arange(1, 101)
        p50 = np.percentile(data, 50)
        assert p50 == 50.5
    
    def test_correlation(self):
        """Test correlation calculation."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        corr = np.corrcoef(x, y)[0, 1]
        assert corr == pytest.approx(1.0)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
