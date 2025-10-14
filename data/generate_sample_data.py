"""
Generate Sample Datasets for Examples
Advanced Statistical Analysis Toolkit

Author: Gabriel Demetrios Lafis
"""

import numpy as np
import pandas as pd
import os

def create_clinical_trial_dataset():
    """Create sample clinical trial dataset."""
    np.random.seed(42)
    
    n_per_group = 50
    
    # Treatment group
    treatment_scores = np.random.normal(105, 15, n_per_group)
    treatment_outcome = np.random.choice(['success', 'failure'], n_per_group, p=[0.7, 0.3])
    
    # Control group
    control_scores = np.random.normal(95, 15, n_per_group)
    control_outcome = np.random.choice(['success', 'failure'], n_per_group, p=[0.5, 0.5])
    
    # Combine
    data = pd.DataFrame({
        'group': ['treatment'] * n_per_group + ['control'] * n_per_group,
        'score': np.concatenate([treatment_scores, control_scores]),
        'outcome': np.concatenate([treatment_outcome, control_outcome]),
        'age': np.random.randint(18, 75, n_per_group * 2),
        'gender': np.random.choice(['M', 'F'], n_per_group * 2)
    })
    
    return data


def create_housing_dataset():
    """Create sample housing dataset."""
    np.random.seed(42)
    
    n = 200
    
    # Generate features
    bedrooms = np.random.randint(1, 6, n)
    bathrooms = np.random.randint(1, 4, n)
    sqft = np.random.randint(800, 4000, n)
    age = np.random.randint(0, 50, n)
    location = np.random.choice(['urban', 'suburban', 'rural'], n)
    
    # Generate price with relationships
    base_price = 100000
    price = (base_price + 
             bedrooms * 30000 + 
             bathrooms * 20000 + 
             sqft * 100 +
             (50 - age) * 1000 +
             np.random.normal(0, 50000, n))
    
    # Adjust for location
    location_effect = {'urban': 1.3, 'suburban': 1.0, 'rural': 0.8}
    price = price * [location_effect[loc] for loc in location]
    
    data = pd.DataFrame({
        'price': price,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft': sqft,
        'age': age,
        'location': location
    })
    
    return data


def create_wine_quality_dataset():
    """Create sample wine quality dataset."""
    np.random.seed(42)
    
    n = 150
    
    data = pd.DataFrame({
        'fixed_acidity': np.random.uniform(4, 12, n),
        'volatile_acidity': np.random.uniform(0.1, 1.2, n),
        'citric_acid': np.random.uniform(0, 0.8, n),
        'residual_sugar': np.random.uniform(0.5, 15, n),
        'chlorides': np.random.uniform(0.01, 0.15, n),
        'free_sulfur_dioxide': np.random.uniform(1, 60, n),
        'total_sulfur_dioxide': np.random.uniform(6, 200, n),
        'density': np.random.uniform(0.99, 1.01, n),
        'pH': np.random.uniform(2.8, 4, n),
        'sulphates': np.random.uniform(0.3, 1.5, n),
        'alcohol': np.random.uniform(8, 15, n),
        'quality': np.random.randint(3, 9, n)
    })
    
    return data


def create_income_dataset():
    """Create sample income dataset."""
    np.random.seed(42)
    
    n = 300
    
    # Generate income data with some skewness
    income = np.random.lognormal(mean=10.8, sigma=0.5, size=n)
    
    data = pd.DataFrame({
        'income': income,
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n),
        'experience': np.random.randint(0, 40, n),
        'age': np.random.randint(22, 65, n)
    })
    
    return data


def create_survey_dataset():
    """Create sample survey dataset."""
    np.random.seed(42)
    
    n = 200
    
    # Generate Likert scale responses (1-5)
    questions = [f'Q{i}' for i in range(1, 11)]
    
    data = {}
    for q in questions:
        data[q] = np.random.randint(1, 6, n)
    
    data['age_group'] = np.random.choice(['18-25', '26-35', '36-45', '46-55', '56+'], n)
    data['satisfaction'] = np.random.randint(1, 6, n)
    
    return pd.DataFrame(data)


def create_timeseries_dataset():
    """Create sample time series dataset."""
    np.random.seed(42)
    
    # Generate daily data for 2 years
    dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
    n = len(dates)
    
    # Trend + seasonality + noise
    trend = np.linspace(100, 150, n)
    seasonality = 20 * np.sin(2 * np.pi * np.arange(n) / 365.25)
    noise = np.random.normal(0, 5, n)
    
    values = trend + seasonality + noise
    
    data = pd.DataFrame({
        'date': dates,
        'value': values,
        'category': np.random.choice(['A', 'B', 'C'], n)
    })
    
    return data


def main():
    """Generate all sample datasets."""
    print("Generating sample datasets...")
    print("="*70)
    
    output_dir = 'data/sample_datasets'
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = {
        'clinical_trial.csv': create_clinical_trial_dataset(),
        'housing.csv': create_housing_dataset(),
        'wine_quality.csv': create_wine_quality_dataset(),
        'income.csv': create_income_dataset(),
        'survey.csv': create_survey_dataset(),
        'timeseries.csv': create_timeseries_dataset()
    }
    
    for filename, data in datasets.items():
        filepath = os.path.join(output_dir, filename)
        data.to_csv(filepath, index=False)
        print(f"✓ Created {filename} ({data.shape[0]} rows, {data.shape[1]} columns)")
    
    print("="*70)
    print(f"All sample datasets saved to {output_dir}/")
    print("\nDataset descriptions:")
    print("- clinical_trial.csv: Treatment vs control group comparison")
    print("- housing.csv: Housing prices with multiple predictors")
    print("- wine_quality.csv: Wine quality with chemical properties")
    print("- income.csv: Income data with demographic variables")
    print("- survey.csv: Survey responses with Likert scale")
    print("- timeseries.csv: Time series data with trend and seasonality")


if __name__ == "__main__":
    main()
