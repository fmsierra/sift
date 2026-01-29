#!/usr/bin/env python3
"""
Predictive Analytics System - Main Pipeline
=============================================

Orchestrates the complete ML pipeline for multivariate time series prediction.

Phases:
    1. EDA - Exploratory Data Analysis
    2. Preprocessing - Data transformation and windowing
    3. Training - Model training with HistGradientBoostingRegressor
    4. Evaluation - Model performance assessment
    5. Prediction - Generate final predictions

Usage:
    # Run complete pipeline
    python main.py --data data/raw/dataset.csv
    
    # Run specific phase
    python main.py --data data/raw/dataset.csv --phase eda
    
    # Run with custom config
    python main.py --data data/raw/dataset.csv --config config/config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_config, load_data, validate_data, print_data_summary
from src.eda import generate_eda_report, print_correlation_insights
from src.preprocessing import preprocess_pipeline, print_preprocessing_summary, TimeSeriesPreprocessor
from src.model import train_model, print_model_summary, MultivariatePredictionModel
from src.evaluation import evaluate_model, print_evaluation_report
from src.prediction import run_final_prediction, print_prediction_results


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


def run_eda(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute Phase 1: Exploratory Data Analysis.
    
    Args:
        df: Raw data
        config: Configuration dictionary
        
    Returns:
        EDA report dictionary
    """
    print("\n" + "=" * 70)
    print("PHASE 1: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    
    output_dir = config.get('output', {}).get('figures_path', 'reports/figures/')
    
    report = generate_eda_report(df, output_dir=output_dir, show_plots=False)
    
    # Print correlation insights
    import pandas as pd
    corr_df = pd.DataFrame(report["correlation_matrix"])
    print_correlation_insights(corr_df)
    
    print(f"\n✓ EDA complete. {len(report['figures'])} figures saved to {output_dir}")
    
    return report


def run_preprocessing(
    df: pd.DataFrame, 
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute Phase 2: Data Preprocessing.
    
    Args:
        df: Raw data
        config: Configuration dictionary
        
    Returns:
        Preprocessing result dictionary
    """
    print("\n" + "=" * 70)
    print("PHASE 2: DATA PREPROCESSING")
    print("=" * 70)
    
    prep_config = config.get('preprocessing', {})
    
    result = preprocess_pipeline(
        df,
        window_size=prep_config.get('window_size', 15),
        normalize=prep_config.get('normalize', True),
        train_split=prep_config.get('train_split', 0.9),
        save_preprocessor='models/preprocessor.joblib'
    )
    
    print_preprocessing_summary(result)
    
    return result


def run_training(
    prep_result: Dict[str, Any],
    config: Dict[str, Any]
) -> MultivariatePredictionModel:
    """
    Execute Phase 3: Model Training.
    
    Args:
        prep_result: Preprocessing result dictionary
        config: Configuration dictionary
        
    Returns:
        Trained model
    """
    print("\n" + "=" * 70)
    print("PHASE 3: MODEL TRAINING")
    print("=" * 70)
    
    model_path = config.get('output', {}).get('model_path', 'models/predictor.joblib')
    
    model = train_model(
        prep_result['X_train'],
        prep_result['y_train'],
        config,
        save_path=model_path
    )
    
    print_model_summary(model)
    
    return model


def run_evaluation(
    model: MultivariatePredictionModel,
    prep_result: Dict[str, Any],
    config: Dict[str, Any],
    column_names: list
) -> Dict[str, Any]:
    """
    Execute Phase 4: Model Evaluation.
    
    Args:
        model: Trained model
        prep_result: Preprocessing result dictionary
        config: Configuration dictionary
        column_names: Names of target columns
        
    Returns:
        Evaluation result dictionary
    """
    print("\n" + "=" * 70)
    print("PHASE 4: MODEL EVALUATION")
    print("=" * 70)
    
    # Make predictions on test set
    y_pred = model.predict(prep_result['X_test'])
    
    # Inverse transform to original scale for meaningful metrics
    y_true_original = prep_result['preprocessor'].inverse_transform(prep_result['y_test'])
    y_pred_original = prep_result['preprocessor'].inverse_transform(y_pred)
    
    output_dir = config.get('output', {}).get('figures_path', 'reports/').replace('/figures/', '/')
    
    result = evaluate_model(
        y_true_original,
        y_pred_original,
        column_names=column_names,
        output_dir=output_dir,
        show_plots=False
    )
    
    print_evaluation_report(result['metrics'])
    
    return result


def run_final_prediction_phase(
    df: pd.DataFrame,
    config: Dict[str, Any],
    eval_result: Dict[str, Any],
    column_names: list
) -> Dict[str, Any]:
    """
    Execute Phase 5: Final Prediction.
    
    Retrains on full data and predicts the next row.
    
    Args:
        df: Full dataset
        config: Configuration dictionary
        eval_result: Evaluation result with metrics
        column_names: Names of target columns
        
    Returns:
        Prediction result dictionary
    """
    print("\n" + "=" * 70)
    print("PHASE 5: FINAL PREDICTION")
    print("=" * 70)
    
    prep_config = config.get('preprocessing', {})
    
    # Create fresh preprocessor and fit on ALL data
    preprocessor = TimeSeriesPreprocessor(
        window_size=prep_config.get('window_size', 15),
        normalize=prep_config.get('normalize', True),
        train_split=1.0  # Use all data
    )
    
    # Fit and transform ALL data
    data_normalized = preprocessor.fit_transform(df)
    
    # Create windowed dataset from ALL data
    X_full, y_full = preprocessor.create_sliding_window(data_normalized)
    
    print(f"Retraining model on 100% of data ({len(X_full)} samples)...")
    
    # Retrain model on full dataset
    model = train_model(X_full, y_full, config)
    
    # Prepare final window and predict
    output_dir = config.get('data', {}).get('predictions_path', 'data/predictions/')
    
    result = run_final_prediction(
        model=model,
        preprocessor=preprocessor,
        df=df,
        column_names=column_names,
        metrics=eval_result['metrics'],
        output_dir=output_dir,
        row_index=len(df) + 1  # Next row index
    )
    
    print_prediction_results(result)
    
    return result


def run_full_pipeline(
    data_path: str,
    config_path: str = "config/config.yaml"
) -> Dict[str, Any]:
    """
    Execute the complete 5-phase pipeline.
    
    Args:
        data_path: Path to input CSV file
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing all phase results
    """
    print("\n" + "=" * 70)
    print("PREDICTIVE ANALYTICS PIPELINE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Load configuration
    config = load_config(config_path)
    setup_logging(config.get('logging', {}).get('level', 'INFO'))
    
    # Load and validate data
    print("\n📊 Loading data...")
    df = load_data(data_path)
    print_data_summary(df)
    
    is_valid, validation_report = validate_data(df, strict=False)
    if not is_valid:
        print("⚠️  Data validation warnings detected. Proceeding anyway...")
    
    column_names = df.columns.tolist()
    
    results = {
        'config': config,
        'data_shape': df.shape,
        'column_names': column_names
    }
    
    # Phase 1: EDA
    results['eda'] = run_eda(df, config)
    
    # Phase 2: Preprocessing
    results['preprocessing'] = run_preprocessing(df, config)
    
    # Phase 3: Training
    results['model'] = run_training(results['preprocessing'], config)
    
    # Phase 4: Evaluation
    results['evaluation'] = run_evaluation(
        results['model'],
        results['preprocessing'],
        config,
        column_names
    )
    
    # Phase 5: Final Prediction
    results['prediction'] = run_final_prediction_phase(
        df, config, results['evaluation'], column_names
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  • Input data: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  • Model R²: {results['evaluation']['metrics']['overall']['mean_r2']:.4f}")
    print(f"  • Predicted Row: {results['prediction']['row_index']}")
    print(f"  • Output: {results['prediction']['csv_path']}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")
    
    return results


def run_single_phase(
    phase: str,
    data_path: str,
    config_path: str = "config/config.yaml"
) -> Dict[str, Any]:
    """
    Execute a single phase of the pipeline.
    
    Args:
        phase: Phase to run ('eda', 'preprocess', 'train', 'evaluate', 'predict')
        data_path: Path to input CSV file
        config_path: Path to configuration file
        
    Returns:
        Phase result dictionary
    """
    config = load_config(config_path)
    setup_logging(config.get('logging', {}).get('level', 'INFO'))
    
    df = load_data(data_path)
    column_names = df.columns.tolist()
    
    if phase == 'eda':
        return run_eda(df, config)
    
    elif phase == 'preprocess':
        return run_preprocessing(df, config)
    
    elif phase == 'train':
        prep_result = run_preprocessing(df, config)
        return {'model': run_training(prep_result, config), 'preprocessing': prep_result}
    
    elif phase == 'evaluate':
        prep_result = run_preprocessing(df, config)
        model = run_training(prep_result, config)
        return run_evaluation(model, prep_result, config, column_names)
    
    elif phase == 'predict':
        # Need to run full pipeline for prediction
        return run_full_pipeline(data_path, config_path)
    
    else:
        raise ValueError(f"Unknown phase: {phase}. Choose from: eda, preprocess, train, evaluate, predict")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Predictive Analytics Pipeline for Multivariate Time Series",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --data data/raw/dataset.csv
  python main.py --data data/raw/dataset.csv --phase eda
  python main.py --data data/raw/dataset.csv --config config/custom.yaml
        """
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Path to the input CSV file'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--phase', '-p',
        type=str,
        choices=['eda', 'preprocess', 'train', 'evaluate', 'predict', 'all'],
        default='all',
        help='Phase to run (default: all)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not Path(args.data).exists():
        print(f"Error: Data file not found: {args.data}")
        print("\nPlease place your CSV data file in the specified location.")
        print("Expected format: CSV with 6 numerical columns")
        sys.exit(1)
    
    # Check if config exists
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    try:
        if args.phase == 'all':
            results = run_full_pipeline(args.data, args.config)
        else:
            results = run_single_phase(args.phase, args.data, args.config)
        
        return 0
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"\n❌ Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
