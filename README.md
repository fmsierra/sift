# Predictive Analytics System

A machine learning pipeline for multivariate time series prediction using HistGradientBoostingRegressor.

## 🎯 Project Objective

Analyze ~2,718 historical records across 6 numerical categories to accurately predict the next set of data points (Row 2,719) using a sliding window approach and gradient boosting.

## 📁 Project Structure

```
sift/
├── config/
│   └── config.yaml          # Pipeline configuration
├── data/
│   ├── raw/                  # Place your input CSV here
│   ├── processed/            # Windowed datasets
│   └── predictions/          # Model predictions output
├── models/                   # Saved trained models
├── reports/
│   ├── figures/              # EDA and evaluation plots
│   └── metrics/              # Performance metrics JSON
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Data ingestion utilities
│   ├── eda.py                # Phase 1: Exploratory Data Analysis
│   ├── preprocessing.py      # Phase 2: Feature engineering
│   ├── model.py              # Phase 3: Model training
│   ├── evaluation.py         # Phase 4: Model evaluation
│   └── prediction.py         # Phase 5: Final prediction
├── tests/
│   └── test_preprocessing.py # Unit tests
├── main.py                   # Pipeline orchestration
├── requirements.txt          # Python dependencies
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Add Your Data

Place your CSV file in `data/raw/dataset.csv`. The file should have:
- ~2,718 rows (sequential time series)
- 6 numerical columns
- No header row issues

### 3. Run the Pipeline

```bash
# Run complete pipeline (all 5 phases)
python main.py --data data/raw/dataset.csv

# Run specific phase
python main.py --data data/raw/dataset.csv --phase eda
python main.py --data data/raw/dataset.csv --phase preprocess
python main.py --data data/raw/dataset.csv --phase train
python main.py --data data/raw/dataset.csv --phase evaluate
python main.py --data data/raw/dataset.csv --phase predict
```

## 📊 Pipeline Phases

### Phase 1: Exploratory Data Analysis (EDA)
- Time series plots for all columns
- Correlation matrix heatmap
- Seasonality detection (autocorrelation)
- Distribution analysis
- Outlier detection

**Output:** `reports/figures/01_*.png` through `06_*.png`

### Phase 2: Data Preprocessing
- MinMax normalization (0-1 scaling)
- Sliding window feature engineering (default: 15 lags)
- Chronological train/test split (90/10)

**Output:** Windowed datasets ready for training

### Phase 3: Model Training
- Algorithm: `HistGradientBoostingRegressor`
- Wrapper: `MultiOutputRegressor` for 6 simultaneous predictions
- Features: Early stopping, regularization

**Output:** `models/predictor.joblib`

### Phase 4: Evaluation
- RMSE, MAE, R² per column
- Actual vs Predicted plots
- Residual analysis
- Error distribution

**Output:** `reports/metrics/evaluation_metrics.json`, `reports/figures/eval_*.png`

### Phase 5: Final Prediction
- Retrain on 100% of data
- Predict Row 2,719
- Generate confidence intervals
- Export results

**Output:** `data/predictions/prediction_2719_*.csv`

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

```yaml
preprocessing:
  window_size: 15          # Lag features (look-back window)
  normalize: true          # MinMax scaling
  train_split: 0.90        # Train/test ratio

model:
  max_iter: 100            # Boosting iterations
  max_depth: 10            # Tree depth (prevents overfitting)
  learning_rate: 0.1       # Step size
  min_samples_leaf: 20     # Minimum leaf samples
  l2_regularization: 0.1   # Regularization strength
```

## 📈 Example Output

```
PREDICTION RESULTS - ROW 2719
======================================================================
Column          Prediction      95% CI Lower    95% CI Upper   
----------------------------------------------------------------------
col_1           23.456789       22.123456       24.790122      
col_2           45.678901       44.345678       47.012134      
...
======================================================================
```

## 🧪 Running Tests

```bash
# Install pytest
pip install pytest

# Run tests
pytest tests/ -v
```

## 📦 Dependencies

- Python 3.8+
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- pyyaml >= 6.0
- joblib >= 1.3.0

## ⚠️ Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Overfitting | Limited tree depth, L2 regularization, early stopping |
| Concept Drift | Recent data weighting, rolling window training |
| Missing Values | Validation checks, imputation options |

## 📝 License

MIT License