"""
Data Preprocessing Module

Provides functions for data cleaning, transformation, and preparation.

Author: Gabriel Demetrios Lafis
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Union
from scipy import stats
import warnings


class DataPreprocessor:
    """Class for data preprocessing and cleaning."""
    
    @staticmethod
    def handle_missing_values(
        data: pd.DataFrame,
        strategy: str = 'mean',
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in DataFrame.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        strategy : str
            Strategy for handling missing values:
            'mean', 'median', 'mode', 'drop', 'forward_fill', 'backward_fill'
        columns : List[str], optional
            Columns to apply strategy to (default: all numeric columns)
            
        Returns
        -------
        data : pd.DataFrame
            Data with missing values handled
        """
        data = data.copy()
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if data[col].isnull().any():
                if strategy == 'mean':
                    data[col].fillna(data[col].mean(), inplace=True)
                elif strategy == 'median':
                    data[col].fillna(data[col].median(), inplace=True)
                elif strategy == 'mode':
                    data[col].fillna(data[col].mode()[0], inplace=True)
                elif strategy == 'drop':
                    data.dropna(subset=[col], inplace=True)
                elif strategy == 'forward_fill':
                    data[col].fillna(method='ffill', inplace=True)
                elif strategy == 'backward_fill':
                    data[col].fillna(method='bfill', inplace=True)
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
        
        return data
    
    @staticmethod
    def detect_outliers(
        data: Union[pd.Series, np.ndarray],
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> np.ndarray:
        """
        Detect outliers in data.
        
        Parameters
        ----------
        data : pd.Series or np.ndarray
            Input data
        method : str
            Method for outlier detection: 'iqr', 'zscore', 'modified_zscore'
        threshold : float
            Threshold for outlier detection
            
        Returns
        -------
        outliers : np.ndarray
            Boolean array indicating outliers
        """
        if isinstance(data, pd.Series):
            data = data.values
        
        if method == 'iqr':
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outliers = (data < lower_bound) | (data > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
            outliers = z_scores > threshold
            
        elif method == 'modified_zscore':
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad if mad != 0 else np.zeros_like(data)
            outliers = np.abs(modified_z_scores) > threshold
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return outliers
    
    @staticmethod
    def remove_outliers(
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Remove outliers from DataFrame.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        columns : List[str], optional
            Columns to check for outliers
        method : str
            Method for outlier detection
        threshold : float
            Threshold for outlier detection
            
        Returns
        -------
        data : pd.DataFrame
            Data with outliers removed
        """
        data = data.copy()
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        mask = np.ones(len(data), dtype=bool)
        
        for col in columns:
            outliers = DataPreprocessor.detect_outliers(data[col], method, threshold)
            mask &= ~outliers
        
        return data[mask]
    
    @staticmethod
    def standardize(
        data: Union[pd.DataFrame, np.ndarray],
        columns: Optional[List[str]] = None
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Standardize data (z-score normalization).
        
        Parameters
        ----------
        data : pd.DataFrame or np.ndarray
            Input data
        columns : List[str], optional
            Columns to standardize
            
        Returns
        -------
        standardized : pd.DataFrame or np.ndarray
            Standardized data
        """
        if isinstance(data, pd.DataFrame):
            result = data.copy()
            if columns is None:
                columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            for col in columns:
                result[col] = (data[col] - data[col].mean()) / data[col].std()
            
            return result
        else:
            return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    
    @staticmethod
    def normalize(
        data: Union[pd.DataFrame, np.ndarray],
        method: str = 'minmax',
        columns: Optional[List[str]] = None
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Normalize data.
        
        Parameters
        ----------
        data : pd.DataFrame or np.ndarray
            Input data
        method : str
            Normalization method: 'minmax', 'maxabs'
        columns : List[str], optional
            Columns to normalize
            
        Returns
        -------
        normalized : pd.DataFrame or np.ndarray
            Normalized data
        """
        if isinstance(data, pd.DataFrame):
            result = data.copy()
            if columns is None:
                columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            for col in columns:
                if method == 'minmax':
                    min_val, max_val = data[col].min(), data[col].max()
                    result[col] = (data[col] - min_val) / (max_val - min_val) if max_val != min_val else 0
                elif method == 'maxabs':
                    max_abs = np.abs(data[col]).max()
                    result[col] = data[col] / max_abs if max_abs != 0 else 0
                else:
                    raise ValueError(f"Unknown method: {method}")
            
            return result
        else:
            if method == 'minmax':
                min_val, max_val = np.min(data, axis=0), np.max(data, axis=0)
                return (data - min_val) / (max_val - min_val)
            elif method == 'maxabs':
                max_abs = np.max(np.abs(data), axis=0)
                return data / max_abs
            else:
                raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def encode_categorical(
        data: pd.DataFrame,
        columns: List[str],
        method: str = 'onehot'
    ) -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        columns : List[str]
            Categorical columns to encode
        method : str
            Encoding method: 'onehot', 'label', 'ordinal'
            
        Returns
        -------
        encoded : pd.DataFrame
            Data with encoded categorical variables
        """
        data = data.copy()
        
        for col in columns:
            if method == 'onehot':
                dummies = pd.get_dummies(data[col], prefix=col, drop_first=False)
                data = pd.concat([data, dummies], axis=1)
                data.drop(col, axis=1, inplace=True)
                
            elif method == 'label':
                data[col] = pd.Categorical(data[col]).codes
                
            elif method == 'ordinal':
                # Assumes natural ordering
                categories = sorted(data[col].unique())
                data[col] = pd.Categorical(data[col], categories=categories, ordered=True).codes
                
            else:
                raise ValueError(f"Unknown method: {method}")
        
        return data
    
    @staticmethod
    def create_polynomial_features(
        data: pd.DataFrame,
        columns: List[str],
        degree: int = 2
    ) -> pd.DataFrame:
        """
        Create polynomial features.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        columns : List[str]
            Columns to create polynomial features for
        degree : int
            Polynomial degree
            
        Returns
        -------
        poly_data : pd.DataFrame
            Data with polynomial features
        """
        result = data.copy()
        
        for col in columns:
            for d in range(2, degree + 1):
                result[f'{col}^{d}'] = data[col] ** d
        
        return result
    
    @staticmethod
    def create_interaction_features(
        data: pd.DataFrame,
        column_pairs: List[Tuple[str, str]]
    ) -> pd.DataFrame:
        """
        Create interaction features.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        column_pairs : List[Tuple[str, str]]
            Pairs of columns to create interactions for
            
        Returns
        -------
        interaction_data : pd.DataFrame
            Data with interaction features
        """
        result = data.copy()
        
        for col1, col2 in column_pairs:
            result[f'{col1}_x_{col2}'] = data[col1] * data[col2]
        
        return result
    
    @staticmethod
    def winsorize(
        data: Union[pd.Series, np.ndarray],
        lower_percentile: float = 0.05,
        upper_percentile: float = 0.95
    ) -> Union[pd.Series, np.ndarray]:
        """
        Winsorize data (cap extreme values).
        
        Parameters
        ----------
        data : pd.Series or np.ndarray
            Input data
        lower_percentile : float
            Lower percentile to cap at
        upper_percentile : float
            Upper percentile to cap at
            
        Returns
        -------
        winsorized : pd.Series or np.ndarray
            Winsorized data
        """
        is_series = isinstance(data, pd.Series)
        values = data.values if is_series else data
        
        lower_val = np.percentile(values, lower_percentile * 100)
        upper_val = np.percentile(values, upper_percentile * 100)
        
        winsorized = np.clip(values, lower_val, upper_val)
        
        return pd.Series(winsorized, index=data.index) if is_series else winsorized
    
    @staticmethod
    def split_data(
        data: pd.DataFrame,
        target_col: str,
        test_size: float = 0.2,
        random_state: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        target_col : str
            Name of target column
        test_size : float
            Proportion of data to use for testing
        random_state : int, optional
            Random state for reproducibility
            
        Returns
        -------
        X_train, X_test, y_train, y_test : tuple
            Training and testing data
        """
        from sklearn.model_selection import train_test_split
        
        X = data.drop(target_col, axis=1)
        y = data[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def check_data_quality(data: pd.DataFrame) -> dict:
        """
        Perform data quality checks.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
            
        Returns
        -------
        report : dict
            Data quality report
        """
        report = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'missing_percentage': (data.isnull().sum() / len(data) * 100).to_dict(),
            'duplicates': data.duplicated().sum(),
            'numeric_columns': data.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': data.select_dtypes(include=['object', 'category']).columns.tolist()
        }
        
        # Summary statistics for numeric columns
        if len(report['numeric_columns']) > 0:
            report['numeric_summary'] = data[report['numeric_columns']].describe().to_dict()
        
        # Unique values for categorical columns
        if len(report['categorical_columns']) > 0:
            report['categorical_unique'] = {
                col: data[col].nunique() for col in report['categorical_columns']
            }
        
        return report
