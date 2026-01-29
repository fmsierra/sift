"""
Prediction Module - Phase 5
============================

Handles final prediction generation for new data points.

Features:
    - Generate predictions for Row 2719 (next step)
    - Export predictions to CSV
    - Confidence intervals (based on historical error)
    - Prediction report generation
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

from .model import MultivariatePredictionModel
from .preprocessing import TimeSeriesPreprocessor

logger = logging.getLogger(__name__)


def predict_next_step(
    model: MultivariatePredictionModel,
    preprocessor: TimeSeriesPreprocessor,
    final_window: np.ndarray
) -> np.ndarray:
    """
    Generate prediction for the next time step.
    
    Args:
        model: Trained prediction model
        preprocessor: Fitted preprocessor with scaler
        final_window: Prepared input window (normalized)
        
    Returns:
        Predicted values in original scale
    """
    # Make prediction (normalized scale)
    prediction_normalized = model.predict(final_window)
    
    # Inverse transform to original scale
    prediction_original = preprocessor.inverse_transform(prediction_normalized)
    
    return prediction_original


def calculate_confidence_intervals(
    prediction: np.ndarray,
    historical_rmse: Dict[str, float],
    column_names: List[str],
    confidence_level: float = 0.95
) -> Dict[str, Dict[str, float]]:
    """
    Calculate confidence intervals based on historical RMSE.
    
    Args:
        prediction: Predicted values
        historical_rmse: RMSE for each column from evaluation
        column_names: Names of target columns
        confidence_level: Confidence level (default 95%)
        
    Returns:
        Dictionary with lower and upper bounds for each column
    """
    # Z-score for 95% confidence
    z_score = 1.96 if confidence_level == 0.95 else 1.645  # 90% alternative
    
    intervals = {}
    pred_flat = prediction.flatten()
    
    for i, col in enumerate(column_names):
        rmse = historical_rmse.get(col, 0)
        margin = z_score * rmse
        
        intervals[col] = {
            'prediction': float(pred_flat[i]),
            'lower_bound': float(pred_flat[i] - margin),
            'upper_bound': float(pred_flat[i] + margin),
            'margin_of_error': float(margin),
            'confidence_level': confidence_level
        }
    
    return intervals


def export_predictions(
    predictions: np.ndarray,
    column_names: List[str],
    output_path: str,
    row_index: int = 2719,
    include_timestamp: bool = True
) -> str:
    """
    Export predictions to CSV file.
    
    Args:
        predictions: Predicted values array
        column_names: Names of columns
        output_path: Directory to save the file
        row_index: Index of the predicted row
        include_timestamp: Whether to add timestamp to filename
        
    Returns:
        Path to the saved file
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame
    df = pd.DataFrame(
        predictions.reshape(1, -1),
        columns=column_names,
        index=[row_index]
    )
    df.index.name = 'row_index'
    
    # Generate filename
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"prediction_{row_index}_{timestamp}.csv"
    else:
        filename = f"prediction_{row_index}.csv"
    
    filepath = output_path / filename
    df.to_csv(filepath)
    
    logger.info(f"Predictions exported to {filepath}")
    return str(filepath)


def generate_prediction_report(
    predictions: np.ndarray,
    column_names: List[str],
    confidence_intervals: Optional[Dict[str, Dict[str, float]]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a comprehensive prediction report.
    
    Args:
        predictions: Predicted values
        column_names: Names of columns
        confidence_intervals: Confidence intervals (optional)
        metrics: Historical evaluation metrics (optional)
        output_path: Path to save the report (optional)
        
    Returns:
        Report dictionary
    """
    report = {
        'generated_at': datetime.now().isoformat(),
        'predictions': {},
        'summary': {}
    }
    
    pred_flat = predictions.flatten()
    
    for i, col in enumerate(column_names):
        col_report = {
            'predicted_value': float(pred_flat[i])
        }
        
        if confidence_intervals and col in confidence_intervals:
            col_report.update(confidence_intervals[col])
        
        if metrics and 'per_column' in metrics and col in metrics['per_column']:
            col_report['historical_rmse'] = metrics['per_column'][col]['rmse']
            col_report['historical_r2'] = metrics['per_column'][col]['r2']
        
        report['predictions'][col] = col_report
    
    # Summary
    report['summary'] = {
        'n_columns_predicted': len(column_names),
        'column_names': column_names,
        'values': pred_flat.tolist()
    }
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Prediction report saved to {output_path}")
    
    return report


def run_final_prediction(
    model: MultivariatePredictionModel,
    preprocessor: TimeSeriesPreprocessor,
    df: pd.DataFrame,
    column_names: List[str],
    metrics: Optional[Dict[str, Any]] = None,
    output_dir: str = "data/predictions/",
    row_index: int = 2719
) -> Dict[str, Any]:
    """
    Execute the complete final prediction workflow.
    
    This function:
    1. Retrains the model on 100% of the data
    2. Prepares the final window
    3. Generates predictions
    4. Calculates confidence intervals
    5. Exports results
    
    Args:
        model: Trained model (will be used as-is if retrain=False)
        preprocessor: Fitted preprocessor
        df: Full dataset
        column_names: Names of target columns
        metrics: Historical evaluation metrics
        output_dir: Directory for output files
        row_index: Index of the row being predicted
        
    Returns:
        Dictionary containing predictions, intervals, and file paths
    """
    logger.info("=" * 60)
    logger.info("STARTING FINAL PREDICTION (Phase 5)")
    logger.info("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Normalize full dataset and prepare final window
    data_normalized = preprocessor.transform(df)
    final_window = preprocessor.prepare_final_window(data_normalized)
    
    logger.info(f"Final window shape: {final_window.shape}")
    logger.info(f"Using last {preprocessor.window_size} rows for prediction")
    
    # Make prediction
    logger.info("Generating prediction for next step...")
    prediction = predict_next_step(model, preprocessor, final_window)
    
    logger.info(f"Raw prediction: {prediction.flatten()}")
    
    # Calculate confidence intervals if metrics available
    confidence_intervals = None
    if metrics and 'per_column' in metrics:
        historical_rmse = {
            col: metrics['per_column'][col]['rmse'] 
            for col in metrics['per_column']
        }
        confidence_intervals = calculate_confidence_intervals(
            prediction, historical_rmse, column_names
        )
    
    # Export predictions
    csv_path = export_predictions(
        prediction, column_names, str(output_dir), row_index
    )
    
    # Generate report
    report_path = output_dir / f"prediction_report_{row_index}.json"
    report = generate_prediction_report(
        prediction, column_names, confidence_intervals, metrics,
        output_path=str(report_path)
    )
    
    result = {
        'predictions': prediction,
        'column_names': column_names,
        'row_index': row_index,
        'confidence_intervals': confidence_intervals,
        'csv_path': csv_path,
        'report_path': str(report_path),
        'report': report
    }
    
    logger.info("=" * 60)
    logger.info("PREDICTION COMPLETE")
    logger.info(f"  Predicted Row: {row_index}")
    logger.info(f"  Values: {prediction.flatten()}")
    logger.info(f"  Output: {csv_path}")
    logger.info("=" * 60)
    
    return result


def print_prediction_results(result: Dict[str, Any]) -> None:
    """
    Print formatted prediction results to console.
    
    Args:
        result: Result dictionary from run_final_prediction
    """
    print("\n" + "=" * 70)
    print(f"PREDICTION RESULTS - ROW {result['row_index']}")
    print("=" * 70)
    
    predictions = result['predictions'].flatten()
    columns = result['column_names']
    intervals = result.get('confidence_intervals', {})
    
    print(f"\n{'Column':<15} {'Prediction':<15} {'95% CI Lower':<15} {'95% CI Upper':<15}")
    print("-" * 70)
    
    for i, col in enumerate(columns):
        pred = predictions[i]
        if col in intervals:
            lower = intervals[col]['lower_bound']
            upper = intervals[col]['upper_bound']
            print(f"{col:<15} {pred:<15.6f} {lower:<15.6f} {upper:<15.6f}")
        else:
            print(f"{col:<15} {pred:<15.6f} {'N/A':<15} {'N/A':<15}")
    
    print("-" * 70)
    print(f"\nPredictions exported to: {result['csv_path']}")
    print(f"Full report saved to: {result['report_path']}")
    
    if intervals:
        print("\nNote: Confidence intervals are based on historical RMSE from evaluation.")
        print("      The model is typically within ± margin of error from the true value.")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Test prediction module
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("Prediction module loaded successfully.")
    print("This module requires trained model and preprocessor to run.")
    print("Use the main.py pipeline script to execute the full workflow.")
