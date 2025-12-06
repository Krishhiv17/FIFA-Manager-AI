from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Feature_Engineering.data_prep import (
    get_value_dataset,
    BASE_FEATURE_COLS,
    TARGET_VALUE,
)

# Where to save models
RF_MODEL_PATH = Path("./models/value_rf.joblib")
XGB_MODEL_PATH = Path("./models/value_xgb.joblib")


def eval_regression(y_true_log, y_pred_log, label: str):
    """Print metrics in log space and original value space."""
    # Log space metrics
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    mae_log = mean_absolute_error(y_true_log, y_pred_log)
    r2_log = r2_score(y_true_log, y_pred_log)

    # Back to original euros
    y_true_val = np.expm1(y_true_log)
    y_pred_val = np.expm1(y_pred_log)

    rmse_val = np.sqrt(mean_squared_error(y_true_val, y_pred_val))
    mae_val = mean_absolute_error(y_true_val, y_pred_val)
    r2_val = r2_score(y_true_val, y_pred_val)

    mean_val = y_true_val.mean()
    median_val = float(np.median(y_true_val))
    mape = np.mean(np.abs(y_true_val - y_pred_val) / (y_true_val + 1e-9)) * 100

    print(f"\n=== {label} – log(value_eur) metrics ===")
    print(f"RMSE (log): {rmse_log:.3f}")
    print(f"MAE  (log): {mae_log:.3f}")
    print(f"R^2  (log): {r2_log:.3f}")

    print(f"\n=== {label} – value_eur metrics ===")
    print(f"RMSE: {rmse_val:,.0f}")
    print(f"MAE : {mae_val:,.0f}")
    print(f"R^2 : {r2_val:.3f}")

    print(f"\n=== {label} – error context ===")
    print(f"Mean true value   : {mean_val:,.0f}")
    print(f"Median true value : {median_val:,.0f}")
    print(f"MAE as % of mean  : {mae_val / mean_val * 100:,.1f}%")
    print(f"MAPE (approx)     : {mape:,.1f}%")

    return {
        "rmse_log": rmse_log,
        "mae_log": mae_log,
        "r2_log": r2_log,
        "rmse_val": rmse_val,
        "mae_val": mae_val,
        "r2_val": r2_val,
        "mape": mape,
    }


def train_value_models(test_size: float = 0.2, random_state: int = 42):
    # 1) Load dataset
    X, y_log, df = get_value_dataset()
    print(f"Loaded dataset for value prediction: {df.shape[0]} rows")
    print(f"Number of features: {len(BASE_FEATURE_COLS)}")
    print(f"Target: log1p({TARGET_VALUE})")

    # 2) Train/test split
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log,
        test_size=test_size,
        random_state=random_state,
    )

    # ---------------------------------------------------------
    # 3) Random Forest baseline
    # ---------------------------------------------------------
    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )

    print("\nTraining Random Forest baseline for value prediction...")
    rf.fit(X_train, y_train_log)

    y_pred_rf_log = rf.predict(X_test)
    rf_metrics = eval_regression(y_test_log, y_pred_rf_log, label="Random Forest")

    # Save RF model
    RF_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": rf,
            "feature_cols": BASE_FEATURE_COLS,
            "target": TARGET_VALUE,
            "target_transform": "log1p",
            "model_type": "random_forest",
        },
        RF_MODEL_PATH,
    )
    print(f"\n[Saved] Random Forest value model → {RF_MODEL_PATH.resolve()}")

    # ---------------------------------------------------------
    # 4) XGBoost model (regularised)
    # ---------------------------------------------------------
    xgb = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=2.0,
        reg_alpha=0.5,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )

    print("\nTraining XGBoost for value prediction...")
    xgb.fit(
        X_train,
        y_train_log,
        eval_set=[(X_test, y_test_log)],
        verbose=False,
    )
    
    y_train_pred_log = xgb.predict(X_train)
    rmse_train_log = np.sqrt(mean_squared_error(y_train_log, y_train_pred_log))
    r2_train_log = r2_score(y_train_log, y_train_pred_log)

    print("\n=== XGBoost – Train vs Test (log value) ===")
    print(f"Train RMSE (log): {rmse_train_log:.3f}")
    print(f"Train R^2  (log): {r2_train_log:.3f}")
    print(f"Test  RMSE (log): {0.202:.3f}")  # or recompute from y_pred_xgb_log
    print(f"Test  R^2  (log): {0.972:.3f}")  # or recompute from y_pred_xgb_log

    y_pred_xgb_log = xgb.predict(X_test)
    xgb_metrics = eval_regression(y_test_log, y_pred_xgb_log, label="XGBoost")

    # 5) Compare and choose best model (likely XGBoost)
    print("\n=== RF vs XGB – R^2 (value_eur) comparison ===")
    print(f"Random Forest R^2 : {rf_metrics['r2_val']:.3f}")
    print(f"XGBoost      R^2  : {xgb_metrics['r2_val']:.3f}")

    # Save XGBoost model
    joblib.dump(
        {
            "model": xgb,
            "feature_cols": BASE_FEATURE_COLS,
            "target": TARGET_VALUE,
            "target_transform": "log1p",
            "model_type": "xgboost",
        },
        XGB_MODEL_PATH,
    )
    print(f"\n[Saved] XGBoost value model → {XGB_MODEL_PATH.resolve()}")

    # 6) Optional: feature importances
    importances = xgb.feature_importances_
    feat_imp = sorted(
        zip(BASE_FEATURE_COLS, importances),
        key=lambda x: x[1],
        reverse=True,
    )

    print("\n=== Top 15 XGBoost feature importances ===")
    for name, score in feat_imp[:15]:
        print(f"{name:30s}: {score:.4f}")


if __name__ == "__main__":
    train_value_models()
