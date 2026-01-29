"""
Exploratory Data Analysis (EDA) Module - Phase 1
=================================================

Provides comprehensive analysis and visualization of the dataset.

Functions:
    - plot_time_series: Line charts for all columns
    - plot_correlation_matrix: Correlation heatmap
    - detect_seasonality: Autocorrelation analysis
    - plot_distributions: Histograms and box plots
    - generate_eda_report: Full EDA report with all visualizations
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_time_series(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create time series line plots for all numerical columns.
    
    Args:
        df: DataFrame with time series data
        columns: Specific columns to plot (default: all numeric)
        figsize: Figure size (width, height)
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib Figure object
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    n_cols = len(columns)
    n_rows = (n_cols + 1) // 2  # 2 columns per row
    
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    axes = axes.flatten() if n_cols > 2 else [axes] if n_cols == 1 else axes.flatten()
    
    for idx, col in enumerate(columns):
        ax = axes[idx]
        ax.plot(df.index, df[col], linewidth=0.8, alpha=0.9)
        ax.set_title(f'{col}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Index (Time)')
        ax.set_ylabel('Value')
        
        # Add trend line
        z = np.polyfit(range(len(df)), df[col].values, 1)
        p = np.poly1d(z)
        ax.plot(df.index, p(range(len(df))), "r--", alpha=0.5, 
                label=f'Trend (slope: {z[0]:.4f})')
        ax.legend(loc='upper right', fontsize=8)
    
    # Hide unused subplots
    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Time Series Analysis - All Columns', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Time series plot saved to {save_path}")
    
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    method: str = 'pearson',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Create a correlation heatmap for all numerical columns.
    
    Args:
        df: DataFrame with numerical data
        method: Correlation method ('pearson', 'spearman', 'kendall')
        figsize: Figure size (width, height)
        save_path: Path to save the figure (optional)
        
    Returns:
        Tuple of (Figure, correlation matrix DataFrame)
    """
    # Calculate correlation matrix
    corr_matrix = df.select_dtypes(include=[np.number]).corr(method=method)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.3f',
        cmap='RdYlBu_r',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation"},
        ax=ax,
        vmin=-1,
        vmax=1
    )
    
    ax.set_title(f'Correlation Matrix ({method.capitalize()})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Correlation matrix saved to {save_path}")
    
    return fig, corr_matrix


def detect_seasonality(
    df: pd.DataFrame,
    column: str,
    max_lag: int = 100,
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Analyze seasonality using autocorrelation function (ACF).
    
    Args:
        df: DataFrame with time series data
        column: Column to analyze
        max_lag: Maximum lag to compute
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Tuple of (Figure, autocorrelation values)
    """
    series = df[column].values
    n = len(series)
    max_lag = min(max_lag, n // 4)  # Limit lag to 1/4 of data length
    
    # Calculate autocorrelation
    acf_values = np.correlate(series - series.mean(), series - series.mean(), mode='full')
    acf_values = acf_values[n-1:n+max_lag] / acf_values[n-1]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.bar(range(len(acf_values)), acf_values, width=0.8, alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Confidence interval (95%)
    conf_int = 1.96 / np.sqrt(n)
    ax.axhline(y=conf_int, color='r', linestyle='--', alpha=0.5, label='95% CI')
    ax.axhline(y=-conf_int, color='r', linestyle='--', alpha=0.5)
    
    # Find significant peaks
    significant_lags = np.where(np.abs(acf_values[1:]) > conf_int)[0] + 1
    if len(significant_lags) > 0:
        ax.scatter(significant_lags, acf_values[significant_lags], 
                  color='red', s=50, zorder=5, label='Significant lags')
    
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(f'Autocorrelation Function: {column}', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Seasonality plot saved to {save_path}")
    
    return fig, acf_values


def plot_seasonality_all_columns(
    df: pd.DataFrame,
    max_lag: int = 50,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot autocorrelation for all numerical columns.
    
    Args:
        df: DataFrame with time series data
        max_lag: Maximum lag to compute
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib Figure object
    """
    columns = df.select_dtypes(include=[np.number]).columns.tolist()
    n_cols = len(columns)
    n_rows = (n_cols + 1) // 2
    
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    axes = axes.flatten() if n_cols > 2 else [axes] if n_cols == 1 else axes.flatten()
    
    for idx, col in enumerate(columns):
        ax = axes[idx]
        series = df[col].values
        n = len(series)
        actual_max_lag = min(max_lag, n // 4)
        
        # Calculate autocorrelation
        acf_values = np.correlate(series - series.mean(), series - series.mean(), mode='full')
        acf_values = acf_values[n-1:n+actual_max_lag] / acf_values[n-1]
        
        ax.bar(range(len(acf_values)), acf_values, width=0.8, alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        # Confidence interval
        conf_int = 1.96 / np.sqrt(n)
        ax.axhline(y=conf_int, color='r', linestyle='--', alpha=0.5)
        ax.axhline(y=-conf_int, color='r', linestyle='--', alpha=0.5)
        
        ax.set_title(f'{col}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Lag')
        ax.set_ylabel('ACF')
    
    # Hide unused subplots
    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Autocorrelation Analysis - Seasonality Detection', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Seasonality plots saved to {save_path}")
    
    return fig


def plot_distributions(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create distribution plots (histogram + KDE) for all columns.
    
    Args:
        df: DataFrame with numerical data
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib Figure object
    """
    columns = df.select_dtypes(include=[np.number]).columns.tolist()
    n_cols = len(columns)
    n_rows = (n_cols + 1) // 2
    
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    axes = axes.flatten() if n_cols > 2 else [axes] if n_cols == 1 else axes.flatten()
    
    for idx, col in enumerate(columns):
        ax = axes[idx]
        
        # Histogram with KDE
        sns.histplot(df[col], kde=True, ax=ax, bins=50, alpha=0.7)
        
        # Add statistics
        mean_val = df[col].mean()
        median_val = df[col].median()
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
        
        # Normality test
        _, p_value = stats.normaltest(df[col].dropna())
        normality = "Normal" if p_value > 0.05 else "Non-Normal"
        
        ax.set_title(f'{col} ({normality}, p={p_value:.3f})', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
    
    # Hide unused subplots
    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Distribution Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Distribution plots saved to {save_path}")
    
    return fig


def plot_box_plots(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create box plots for outlier detection.
    
    Args:
        df: DataFrame with numerical data
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize for comparison
    df_normalized = (df - df.min()) / (df.max() - df.min())
    
    df_normalized.boxplot(ax=ax, grid=True, notch=True)
    ax.set_title('Box Plots (Normalized) - Outlier Detection', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalized Value (0-1)')
    ax.set_xlabel('Columns')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Box plots saved to {save_path}")
    
    return fig


def plot_rolling_statistics(
    df: pd.DataFrame,
    window: int = 50,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot rolling mean and standard deviation to detect trends and volatility.
    
    Args:
        df: DataFrame with time series data
        window: Rolling window size
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib Figure object
    """
    columns = df.select_dtypes(include=[np.number]).columns.tolist()
    n_cols = len(columns)
    n_rows = (n_cols + 1) // 2
    
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    axes = axes.flatten() if n_cols > 2 else [axes] if n_cols == 1 else axes.flatten()
    
    for idx, col in enumerate(columns):
        ax = axes[idx]
        
        # Original data
        ax.plot(df.index, df[col], alpha=0.5, label='Original', linewidth=0.5)
        
        # Rolling mean
        rolling_mean = df[col].rolling(window=window).mean()
        ax.plot(df.index, rolling_mean, color='red', label=f'Rolling Mean ({window})')
        
        # Rolling std as shaded area
        rolling_std = df[col].rolling(window=window).std()
        ax.fill_between(
            df.index,
            rolling_mean - rolling_std,
            rolling_mean + rolling_std,
            alpha=0.2,
            color='red',
            label='±1 Std'
        )
        
        ax.set_title(f'{col}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
    
    # Hide unused subplots
    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Rolling Statistics - Trend & Volatility Analysis', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Rolling statistics plot saved to {save_path}")
    
    return fig


def generate_eda_report(
    df: pd.DataFrame,
    output_dir: str = "reports/figures/",
    show_plots: bool = False
) -> Dict[str, Any]:
    """
    Generate a complete EDA report with all visualizations.
    
    Args:
        df: DataFrame to analyze
        output_dir: Directory to save figures
        show_plots: Whether to display plots interactively
        
    Returns:
        Dictionary containing EDA results and file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        "data_shape": df.shape,
        "columns": list(df.columns),
        "figures": [],
        "correlation_matrix": None,
        "statistics": {}
    }
    
    logger.info("=" * 60)
    logger.info("STARTING EXPLORATORY DATA ANALYSIS (Phase 1)")
    logger.info("=" * 60)
    
    # 1. Time Series Plots
    logger.info("Generating time series plots...")
    fig_ts = plot_time_series(
        df, 
        save_path=str(output_dir / "01_time_series.png")
    )
    report["figures"].append("01_time_series.png")
    
    # 2. Correlation Matrix
    logger.info("Computing correlation matrix...")
    fig_corr, corr_matrix = plot_correlation_matrix(
        df,
        save_path=str(output_dir / "02_correlation_matrix.png")
    )
    report["figures"].append("02_correlation_matrix.png")
    report["correlation_matrix"] = corr_matrix.to_dict()
    
    # 3. Seasonality/Autocorrelation
    logger.info("Analyzing seasonality patterns...")
    fig_season = plot_seasonality_all_columns(
        df,
        save_path=str(output_dir / "03_seasonality_acf.png")
    )
    report["figures"].append("03_seasonality_acf.png")
    
    # 4. Distributions
    logger.info("Plotting distributions...")
    fig_dist = plot_distributions(
        df,
        save_path=str(output_dir / "04_distributions.png")
    )
    report["figures"].append("04_distributions.png")
    
    # 5. Box Plots (Outliers)
    logger.info("Creating box plots for outlier detection...")
    fig_box = plot_box_plots(
        df,
        save_path=str(output_dir / "05_box_plots.png")
    )
    report["figures"].append("05_box_plots.png")
    
    # 6. Rolling Statistics
    logger.info("Computing rolling statistics...")
    fig_roll = plot_rolling_statistics(
        df,
        save_path=str(output_dir / "06_rolling_statistics.png")
    )
    report["figures"].append("06_rolling_statistics.png")
    
    # Calculate summary statistics
    for col in df.select_dtypes(include=[np.number]).columns:
        report["statistics"][col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "skew": float(df[col].skew()),
            "kurtosis": float(df[col].kurtosis())
        }
    
    if show_plots:
        plt.show()
    else:
        plt.close('all')
    
    logger.info("=" * 60)
    logger.info("EDA COMPLETE - All figures saved to: %s", output_dir)
    logger.info("=" * 60)
    
    return report


def print_correlation_insights(corr_matrix: pd.DataFrame, threshold: float = 0.5) -> None:
    """
    Print insights about strongly correlated variables.
    
    Args:
        corr_matrix: Correlation matrix DataFrame
        threshold: Correlation threshold for "strong" correlation
    """
    print("\n" + "=" * 50)
    print("CORRELATION INSIGHTS")
    print("=" * 50)
    
    # Find strong correlations (excluding diagonal)
    strong_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                strong_corr.append({
                    "col1": corr_matrix.columns[i],
                    "col2": corr_matrix.columns[j],
                    "correlation": corr_val
                })
    
    if strong_corr:
        print(f"\nStrong correlations (|r| >= {threshold}):")
        for item in sorted(strong_corr, key=lambda x: abs(x["correlation"]), reverse=True):
            direction = "positive" if item["correlation"] > 0 else "negative"
            print(f"  • {item['col1']} ↔ {item['col2']}: {item['correlation']:.3f} ({direction})")
        
        print("\nInterpretation:")
        print("  - Positive correlation: Variables move together")
        print("  - Negative correlation: Variables move in opposite directions")
        print("  - This suggests multivariate modeling may capture relationships")
    else:
        print(f"\nNo strong correlations found (|r| >= {threshold})")
        print("  - Variables appear relatively independent")
        print("  - Individual models might perform as well as multivariate")
    
    print("=" * 50 + "\n")


if __name__ == "__main__":
    # Test EDA module with sample data
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create sample data for testing
    np.random.seed(42)
    n_samples = 100
    
    # Simulated correlated time series
    t = np.arange(n_samples)
    base_signal = np.sin(2 * np.pi * t / 20) + 0.1 * np.random.randn(n_samples)
    
    sample_df = pd.DataFrame({
        'col_1': base_signal + np.random.randn(n_samples) * 0.2,
        'col_2': base_signal * 0.8 + np.random.randn(n_samples) * 0.3 + 5,
        'col_3': -base_signal + np.random.randn(n_samples) * 0.25 + 10,
        'col_4': np.random.randn(n_samples) + 15,  # Independent
        'col_5': base_signal * 0.5 + t * 0.01 + np.random.randn(n_samples) * 0.2 + 20,  # With trend
        'col_6': base_signal * 0.3 + np.random.randn(n_samples) * 0.4 + 25
    })
    
    print("Running EDA on sample data...")
    report = generate_eda_report(sample_df, show_plots=False)
    
    # Print correlation insights
    corr_df = pd.DataFrame(report["correlation_matrix"])
    print_correlation_insights(corr_df)
    
    print(f"Generated {len(report['figures'])} figures")
