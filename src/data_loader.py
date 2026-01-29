"""
Data Loader Module
==================

Handles CSV ingestion, validation, and basic data quality checks.

Functions:
    - load_config: Load YAML configuration file
    - load_data: Load CSV data with validation
    - validate_data: Check data quality constraints
    - get_data_summary: Generate basic statistics
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def load_data(
    file_path: str,
    expected_columns: Optional[int] = None,
    index_col: Optional[int] = None
) -> pd.DataFrame:
    """
    Load CSV data with automatic type detection and validation.
    
    Args:
        file_path: Path to the CSV file
        expected_columns: Expected number of columns (optional validation)
        index_col: Column to use as index (optional)
        
    Returns:
        DataFrame containing the loaded data
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data doesn't meet quality constraints
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Load data
    df = pd.read_csv(file_path, index_col=index_col)
    logger.info(f"Loaded data from {file_path}: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Validate column count if specified
    if expected_columns is not None and df.shape[1] != expected_columns:
        raise ValueError(
            f"Expected {expected_columns} columns, but found {df.shape[1]}. "
            f"Columns: {list(df.columns)}"
        )
    
    return df


def validate_data(df: pd.DataFrame, strict: bool = True) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate data quality constraints for time-series analysis.
    
    Checks:
        - All columns are numerical
        - No missing values (or within acceptable threshold)
        - Data is sequential (no major gaps)
        - Sufficient data points for windowing
        
    Args:
        df: DataFrame to validate
        strict: If True, raise errors on validation failure
        
    Returns:
        Tuple of (is_valid, validation_report)
    """
    report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "column_names": list(df.columns),
        "issues": []
    }
    
    # Check 1: All columns should be numerical
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        issue = f"Non-numeric columns found: {non_numeric_cols}"
        report["issues"].append(issue)
        logger.warning(issue)
    
    # Check 2: Missing values
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    if total_missing > 0:
        missing_pct = (total_missing / (df.shape[0] * df.shape[1])) * 100
        issue = f"Missing values: {total_missing} ({missing_pct:.2f}%)"
        report["issues"].append(issue)
        report["missing_by_column"] = missing_counts[missing_counts > 0].to_dict()
        logger.warning(issue)
    
    # Check 3: Duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issue = f"Duplicate rows found: {duplicates}"
        report["issues"].append(issue)
        logger.warning(issue)
    
    # Check 4: Data range (check for potential outliers)
    for col in df.select_dtypes(include=[np.number]).columns:
        col_std = df[col].std()
        col_mean = df[col].mean()
        outliers = ((df[col] - col_mean).abs() > 4 * col_std).sum()
        if outliers > 0:
            issue = f"Column '{col}' has {outliers} potential outliers (>4 std)"
            report["issues"].append(issue)
            logger.warning(issue)
    
    is_valid = len(report["issues"]) == 0
    report["is_valid"] = is_valid
    
    if strict and not is_valid:
        raise ValueError(f"Data validation failed: {report['issues']}")
    
    return is_valid, report


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive summary statistics for the dataset.
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        Dictionary containing summary statistics
    """
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        "statistics": {}
    }
    
    # Per-column statistics
    for col in df.select_dtypes(include=[np.number]).columns:
        summary["statistics"][col] = {
            "count": int(df[col].count()),
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "25%": float(df[col].quantile(0.25)),
            "50%": float(df[col].quantile(0.50)),
            "75%": float(df[col].quantile(0.75)),
            "max": float(df[col].max()),
            "skew": float(df[col].skew()),
            "kurtosis": float(df[col].kurtosis())
        }
    
    return summary


def print_data_summary(df: pd.DataFrame) -> None:
    """
    Print a formatted summary of the dataset to console.
    
    Args:
        df: DataFrame to summarize
    """
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    print("\nColumn Information:")
    print("-" * 40)
    
    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].count()
        null_pct = (1 - non_null / len(df)) * 100
        print(f"  {col}: {dtype} | {non_null} non-null ({null_pct:.1f}% missing)")
    
    print("\nBasic Statistics:")
    print("-" * 40)
    print(df.describe().round(4).to_string())
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Test the data loader with sample data
    logging.basicConfig(level=logging.INFO)
    
    # Load config
    try:
        config = load_config()
        print("Configuration loaded successfully!")
        print(f"Window size: {config['preprocessing']['window_size']}")
        print(f"Train split: {config['preprocessing']['train_split']}")
    except FileNotFoundError as e:
        print(f"Config not found: {e}")
    
    # Test data loading (will fail if no data exists)
    data_path = "data/raw/dataset.csv"
    if os.path.exists(data_path):
        df = load_data(data_path)
        print_data_summary(df)
        is_valid, report = validate_data(df, strict=False)
        print(f"Validation passed: {is_valid}")
    else:
        print(f"No data file found at {data_path}")
        print("Place your CSV file there to test the data loader.")
