from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Feature_Engineering.data_prep import (
    get_overall_dataset,
    BASE_FEATURE_COLS,
    TARGET_OVERALL,
)

RF_MODEL_PATH = Path("./models/overall_rf.joblib")
XGB_MODEL_PATH = Path("./models/overall_xgb.joblib")


def eval_overall(y_true, y_pred, label: str):
    """Print metrics for overall rating prediction (0–99 scale)."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n=== {label} – overall metrics ===")
    print(f"RMSE (rating pts): {rmse:.3f}")
    print(f"MAE  (rating pts): {mae:.3f}")
    print(f"R^2              : {r2:.3f}")

    print(f"\n=== {label} – error context ===")
    print(f"Mean overall   : {y_true.mean():.2f}")
    print(f"Median overall : {np.median(y_true):.2f}")
    print(f"MAE as % of mean overall: {mae / (y_true.mean() + 1e-9) * 100:.1f}%")

    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_overall_models(test_size: float = 0.2, random_state: int = 42):
    # 1) Load dataset
    X, y, df = get_overall_dataset()
    print(f"Loaded dataset for overall prediction: {df.shape[0]} rows")
    print(f"Number of features: {len(BASE_FEATURE_COLS)}")
    print(f"Target: {TARGET_OVERALL}")

    # 2) Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
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

    print("\nTraining Random Forest baseline for overall prediction...")
    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)
    rf_metrics = eval_overall(y_test, y_pred_rf, label="Random Forest")

    RF_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": rf,
            "feature_cols": BASE_FEATURE_COLS,
            "target": TARGET_OVERALL,
            "model_type": "random_forest",
        },
        RF_MODEL_PATH,
    )
    print(f"\n[Saved] Random Forest overall model → {RF_MODEL_PATH.resolve()}")

    # ---------------------------------------------------------
    # 4) XGBoost model
    # ---------------------------------------------------------
    xgb = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=2.0,
        reg_alpha=0.5,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )

    print("\nTraining XGBoost for overall prediction...")
    xgb.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Train vs Test check
    y_train_pred = xgb.predict(X_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    r2_train = r2_score(y_train, y_train_pred)

    print("\n=== XGBoost – Train vs Test (overall) ===")
    print(f"Train RMSE: {rmse_train:.3f}")
    print(f"Train R^2 : {r2_train:.3f}")

    y_pred_xgb = xgb.predict(X_test)
    xgb_metrics = eval_overall(y_test, y_pred_xgb, label="XGBoost")

    # Compare
    print("\n=== RF vs XGB – R^2 (overall) comparison ===")
    print(f"Random Forest R^2 : {rf_metrics['r2']:.3f}")
    print(f"XGBoost      R^2  : {xgb_metrics['r2']:.3f}")

    # Save XGBoost model
    joblib.dump(
        {
            "model": xgb,
            "feature_cols": BASE_FEATURE_COLS,
            "target": TARGET_OVERALL,
            "model_type": "xgboost",
        },
        XGB_MODEL_PATH,
    )
    print(f"\n[Saved] XGBoost overall model → {XGB_MODEL_PATH.resolve()}")

    # Feature importances
    importances = xgb.feature_importances_
    feat_imp = sorted(
        zip(BASE_FEATURE_COLS, importances),
        key=lambda x: x[1],
        reverse=True,
    )

    print("\n=== Top 15 XGBoost feature importances (overall) ===")
    for name, score in feat_imp[:15]:
        print(f"{name:30s}: {score:.4f}")


if __name__ == "__main__":
    train_overall_models()
