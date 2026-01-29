"""
Model Evaluation Module - Phase 4
==================================

Provides comprehensive evaluation metrics and visualizations for model performance.

Features:
    - RMSE, MAE, R² calculation per target
    - Actual vs Predicted plots
    - Residual analysis
    - Error distribution plots
    - Evaluation report generation
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    column_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics for each target.
    
    Args:
        y_true: Ground truth values of shape (n_samples, n_targets)
        y_pred: Predicted values of shape (n_samples, n_targets)
        column_names: Names of target columns
        
    Returns:
        Dictionary containing metrics for each column and overall
    """
    n_targets = y_true.shape[1] if len(y_true.shape) > 1 else 1
    
    if column_names is None:
        column_names = [f"target_{i+1}" for i in range(n_targets)]
    
    metrics = {
        'per_column': {},
        'overall': {}
    }
    
    all_rmse = []
    all_mae = []
    all_r2 = []
    
    for i, col in enumerate(column_names):
        true_col = y_true[:, i] if len(y_true.shape) > 1 else y_true
        pred_col = y_pred[:, i] if len(y_pred.shape) > 1 else y_pred
        
        rmse = np.sqrt(mean_squared_error(true_col, pred_col))
        mae = mean_absolute_error(true_col, pred_col)
        r2 = r2_score(true_col, pred_col)
        mape = np.mean(np.abs((true_col - pred_col) / (true_col + 1e-10))) * 100
        
        metrics['per_column'][col] = {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape),
            'std_error': float(np.std(true_col - pred_col)),
            'mean_error': float(np.mean(true_col - pred_col)),
            'max_error': float(np.max(np.abs(true_col - pred_col))),
            'min_error': float(np.min(np.abs(true_col - pred_col)))
        }
        
        all_rmse.append(rmse)
        all_mae.append(mae)
        all_r2.append(r2)
    
    # Overall metrics
    metrics['overall'] = {
        'mean_rmse': float(np.mean(all_rmse)),
        'mean_mae': float(np.mean(all_mae)),
        'mean_r2': float(np.mean(all_r2)),
        'total_rmse': float(np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))),
        'n_samples': int(len(y_true)),
        'n_targets': int(n_targets)
    }
    
    return metrics


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    column_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create actual vs predicted scatter plots for each target.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        column_names: Names of target columns
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib Figure object
    """
    n_targets = y_true.shape[1] if len(y_true.shape) > 1 else 1
    
    if column_names is None:
        column_names = [f"target_{i+1}" for i in range(n_targets)]
    
    n_rows = (n_targets + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    axes = axes.flatten() if n_targets > 2 else [axes] if n_targets == 1 else axes.flatten()
    
    for i, col in enumerate(column_names):
        ax = axes[i]
        
        true_col = y_true[:, i] if len(y_true.shape) > 1 else y_true
        pred_col = y_pred[:, i] if len(y_pred.shape) > 1 else y_pred
        
        # Scatter plot
        ax.scatter(true_col, pred_col, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(true_col.min(), pred_col.min())
        max_val = max(true_col.max(), pred_col.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        
        # Calculate R²
        r2 = r2_score(true_col, pred_col)
        rmse = np.sqrt(mean_squared_error(true_col, pred_col))
        
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{col}\nR²={r2:.4f}, RMSE={rmse:.4f}', fontsize=10, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
    
    # Hide unused subplots
    for idx in range(len(column_names), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Actual vs Predicted - Model Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Actual vs Predicted plot saved to {save_path}")
    
    return fig


def plot_prediction_timeline(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    column_names: Optional[List[str]] = None,
    n_samples: int = 100,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot actual and predicted values over time (last n samples).
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        column_names: Names of target columns
        n_samples: Number of samples to display
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib Figure object
    """
    n_targets = y_true.shape[1] if len(y_true.shape) > 1 else 1
    
    if column_names is None:
        column_names = [f"target_{i+1}" for i in range(n_targets)]
    
    # Use last n_samples
    y_true = y_true[-n_samples:]
    y_pred = y_pred[-n_samples:]
    
    n_rows = (n_targets + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    axes = axes.flatten() if n_targets > 2 else [axes] if n_targets == 1 else axes.flatten()
    
    x = np.arange(len(y_true))
    
    for i, col in enumerate(column_names):
        ax = axes[i]
        
        true_col = y_true[:, i] if len(y_true.shape) > 1 else y_true
        pred_col = y_pred[:, i] if len(y_pred.shape) > 1 else y_pred
        
        ax.plot(x, true_col, 'b-', linewidth=1.5, label='Actual', alpha=0.8)
        ax.plot(x, pred_col, 'r--', linewidth=1.5, label='Predicted', alpha=0.8)
        
        ax.fill_between(x, true_col, pred_col, alpha=0.2, color='gray')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.set_title(f'{col}', fontsize=10, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
    
    # Hide unused subplots
    for idx in range(len(column_names), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Prediction Timeline (Last {n_samples} Samples)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Prediction timeline plot saved to {save_path}")
    
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    column_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create residual distribution plots for model diagnostics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        column_names: Names of target columns
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib Figure object
    """
    residuals = y_true - y_pred
    n_targets = y_true.shape[1] if len(y_true.shape) > 1 else 1
    
    if column_names is None:
        column_names = [f"target_{i+1}" for i in range(n_targets)]
    
    n_rows = (n_targets + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    axes = axes.flatten() if n_targets > 2 else [axes] if n_targets == 1 else axes.flatten()
    
    for i, col in enumerate(column_names):
        ax = axes[i]
        
        res_col = residuals[:, i] if len(residuals.shape) > 1 else residuals
        
        # Histogram of residuals
        sns.histplot(res_col, kde=True, ax=ax, bins=50, alpha=0.7)
        
        # Add vertical line at 0
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
        ax.axvline(np.mean(res_col), color='green', linestyle='--', 
                  linewidth=2, label=f'Mean: {np.mean(res_col):.4f}')
        
        ax.set_xlabel('Residual (Actual - Predicted)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{col} (Std: {np.std(res_col):.4f})', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
    
    # Hide unused subplots
    for idx in range(len(column_names), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Residual Analysis - Error Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Residuals plot saved to {save_path}")
    
    return fig


def plot_error_summary(
    metrics: Dict[str, Any],
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a bar chart summary of RMSE and MAE for each target.
    
    Args:
        metrics: Metrics dictionary from calculate_metrics
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib Figure object
    """
    columns = list(metrics['per_column'].keys())
    rmse_values = [metrics['per_column'][col]['rmse'] for col in columns]
    mae_values = [metrics['per_column'][col]['mae'] for col in columns]
    r2_values = [metrics['per_column'][col]['r2'] for col in columns]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    x = np.arange(len(columns))
    width = 0.6
    
    # RMSE
    axes[0].bar(x, rmse_values, width, color='steelblue', alpha=0.8)
    axes[0].axhline(metrics['overall']['mean_rmse'], color='red', linestyle='--', 
                   label=f"Mean: {metrics['overall']['mean_rmse']:.4f}")
    axes[0].set_xlabel('Target Column')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('Root Mean Squared Error', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(columns, rotation=45, ha='right')
    axes[0].legend()
    
    # MAE
    axes[1].bar(x, mae_values, width, color='coral', alpha=0.8)
    axes[1].axhline(metrics['overall']['mean_mae'], color='red', linestyle='--',
                   label=f"Mean: {metrics['overall']['mean_mae']:.4f}")
    axes[1].set_xlabel('Target Column')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Mean Absolute Error', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(columns, rotation=45, ha='right')
    axes[1].legend()
    
    # R²
    colors = ['green' if r2 > 0.8 else 'orange' if r2 > 0.5 else 'red' for r2 in r2_values]
    axes[2].bar(x, r2_values, width, color=colors, alpha=0.8)
    axes[2].axhline(metrics['overall']['mean_r2'], color='blue', linestyle='--',
                   label=f"Mean: {metrics['overall']['mean_r2']:.4f}")
    axes[2].axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    axes[2].set_xlabel('Target Column')
    axes[2].set_ylabel('R² Score')
    axes[2].set_title('R² Score (Coefficient of Determination)', fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(columns, rotation=45, ha='right')
    axes[2].legend()
    axes[2].set_ylim([min(0, min(r2_values) - 0.1), 1.1])
    
    plt.suptitle('Model Performance Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Error summary plot saved to {save_path}")
    
    return fig


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    column_names: Optional[List[str]] = None,
    output_dir: str = "reports/",
    show_plots: bool = False
) -> Dict[str, Any]:
    """
    Run complete model evaluation and generate all reports.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        column_names: Names of target columns
        output_dir: Directory for output files
        show_plots: Whether to display plots interactively
        
    Returns:
        Dictionary containing metrics and file paths
    """
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    metrics_dir = output_dir / "metrics"
    
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("STARTING MODEL EVALUATION (Phase 4)")
    logger.info("=" * 60)
    
    # Calculate metrics
    logger.info("Calculating evaluation metrics...")
    metrics = calculate_metrics(y_true, y_pred, column_names)
    
    # Save metrics to JSON
    metrics_file = metrics_dir / "evaluation_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_file}")
    
    # Generate plots
    figures = []
    
    logger.info("Generating Actual vs Predicted plots...")
    fig1 = plot_actual_vs_predicted(
        y_true, y_pred, column_names,
        save_path=str(figures_dir / "eval_actual_vs_predicted.png")
    )
    figures.append("eval_actual_vs_predicted.png")
    
    logger.info("Generating prediction timeline...")
    fig2 = plot_prediction_timeline(
        y_true, y_pred, column_names,
        save_path=str(figures_dir / "eval_prediction_timeline.png")
    )
    figures.append("eval_prediction_timeline.png")
    
    logger.info("Generating residual analysis...")
    fig3 = plot_residuals(
        y_true, y_pred, column_names,
        save_path=str(figures_dir / "eval_residuals.png")
    )
    figures.append("eval_residuals.png")
    
    logger.info("Generating error summary...")
    fig4 = plot_error_summary(
        metrics,
        save_path=str(figures_dir / "eval_error_summary.png")
    )
    figures.append("eval_error_summary.png")
    
    if show_plots:
        plt.show()
    else:
        plt.close('all')
    
    result = {
        'metrics': metrics,
        'figures': figures,
        'metrics_file': str(metrics_file)
    }
    
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info(f"  Mean RMSE: {metrics['overall']['mean_rmse']:.6f}")
    logger.info(f"  Mean MAE: {metrics['overall']['mean_mae']:.6f}")
    logger.info(f"  Mean R²: {metrics['overall']['mean_r2']:.6f}")
    logger.info("=" * 60)
    
    return result


def print_evaluation_report(metrics: Dict[str, Any]) -> None:
    """
    Print a formatted evaluation report to console.
    
    Args:
        metrics: Metrics dictionary from calculate_metrics
    """
    print("\n" + "=" * 70)
    print("MODEL EVALUATION REPORT")
    print("=" * 70)
    
    print("\nPer-Column Metrics:")
    print("-" * 70)
    print(f"{'Column':<15} {'RMSE':<12} {'MAE':<12} {'R²':<12} {'MAPE (%)':<12}")
    print("-" * 70)
    
    for col, col_metrics in metrics['per_column'].items():
        print(f"{col:<15} {col_metrics['rmse']:<12.6f} {col_metrics['mae']:<12.6f} "
              f"{col_metrics['r2']:<12.6f} {col_metrics['mape']:<12.2f}")
    
    print("-" * 70)
    print("\nOverall Metrics:")
    print(f"  • Mean RMSE: {metrics['overall']['mean_rmse']:.6f}")
    print(f"  • Mean MAE: {metrics['overall']['mean_mae']:.6f}")
    print(f"  • Mean R²: {metrics['overall']['mean_r2']:.6f}")
    print(f"  • Total RMSE: {metrics['overall']['total_rmse']:.6f}")
    print(f"  • Samples evaluated: {metrics['overall']['n_samples']}")
    
    # Interpretation
    mean_r2 = metrics['overall']['mean_r2']
    print("\nInterpretation:")
    if mean_r2 > 0.9:
        print("  ✓ Excellent model performance (R² > 0.9)")
    elif mean_r2 > 0.7:
        print("  ✓ Good model performance (R² > 0.7)")
    elif mean_r2 > 0.5:
        print("  ⚠ Moderate model performance (R² > 0.5)")
    else:
        print("  ✗ Poor model performance (R² < 0.5) - consider different approach")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Test evaluation with sample data
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create sample actual and predicted data
    np.random.seed(42)
    n_samples = 200
    n_targets = 6
    
    column_names = [f"col_{i+1}" for i in range(n_targets)]
    
    # Simulated actual values
    y_true = np.random.randn(n_samples, n_targets) * 10 + 50
    
    # Simulated predictions (with some error)
    noise = np.random.randn(n_samples, n_targets) * 2
    y_pred = y_true + noise
    
    print("Testing evaluation module...")
    result = evaluate_model(
        y_true, y_pred,
        column_names=column_names,
        output_dir="reports/",
        show_plots=False
    )
    
    print_evaluation_report(result['metrics'])
