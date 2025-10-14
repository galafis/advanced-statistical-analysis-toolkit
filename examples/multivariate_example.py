"""
Multivariate Analysis Example
Advanced Statistical Analysis Toolkit

Demonstrates PCA, LDA, and clustering analysis.

Author: Gabriel Demetrios Lafis
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from statistical_visualizations import StatisticalVisualizer

def main():
    """Run multivariate analysis examples."""
    print("="*70)
    print("Multivariate Analysis Example")
    print("="*70)
    
    # Create results directory
    os.makedirs('../data/results', exist_ok=True)
    
    # Load data
    print("\nLoading wine quality data...")
    data = pd.read_csv('../data/sample_datasets/wine_quality.csv')
    print(f"Dataset shape: {data.shape}")
    
    # Prepare data
    features = data.drop('quality', axis=1)
    target = data['quality']
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # =========================================================================
    # Principal Component Analysis (PCA)
    # =========================================================================
    print("\n" + "="*70)
    print("Principal Component Analysis (PCA)")
    print("="*70)
    
    # Perform PCA
    pca = PCA()
    pca_scores = pca.fit_transform(features_scaled)
    
    # Variance explained
    variance_explained = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_explained)
    
    print("\nVariance explained by each component:")
    for i, (var, cum_var) in enumerate(zip(variance_explained[:5], cumulative_variance[:5]), 1):
        print(f"  PC{i}: {var*100:.2f}% (Cumulative: {cum_var*100:.2f}%)")
    
    # Number of components for 95% variance
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"\nComponents needed for 95% variance: {n_components_95}")
    
    # Visualize PCA
    viz = StatisticalVisualizer()
    fig = viz.plot_pca_results(
        pca_scores,
        variance_explained,
        pca.components_.T,
        features.columns.tolist(),
        save_path='../data/results/pca_analysis.png'
    )
    plt.close()
    print("✓ PCA plots saved to ../data/results/pca_analysis.png")
    
    # =========================================================================
    # K-Means Clustering
    # =========================================================================
    print("\n" + "="*70)
    print("K-Means Clustering")
    print("="*70)
    
    # Determine optimal number of clusters (elbow method)
    inertias = []
    K_range = range(2, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features_scaled)
        inertias.append(kmeans.inertia_)
    
    # Plot elbow curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method for Optimal k')
    ax.grid(True, alpha=0.3)
    plt.savefig('../data/results/kmeans_elbow.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Elbow plot saved to ../data/results/kmeans_elbow.png")
    
    # Fit K-means with k=3
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_scaled)
    
    print(f"\nCluster sizes:")
    for i in range(3):
        print(f"  Cluster {i}: {np.sum(clusters == i)} observations")
    
    # Visualize clusters in PCA space
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(pca_scores[:, 0], pca_scores[:, 1], 
                        c=clusters, cmap='viridis', alpha=0.6, s=50)
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
              marker='X', s=300, c='red', edgecolors='black', linewidths=2,
              label='Centroids')
    ax.set_xlabel(f'PC1 ({variance_explained[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({variance_explained[1]*100:.1f}%)')
    ax.set_title('K-Means Clustering (k=3) in PCA Space')
    ax.legend()
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    plt.savefig('../data/results/kmeans_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Cluster plot saved to ../data/results/kmeans_clusters.png")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print("\nGenerated files:")
    print("  - pca_analysis.png")
    print("  - kmeans_elbow.png")
    print("  - kmeans_clusters.png")
    print("\n✓ Example completed successfully!")


if __name__ == "__main__":
    main()
