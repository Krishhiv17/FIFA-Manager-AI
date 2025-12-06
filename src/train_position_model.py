from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Feature_Engineering.data_prep import (
    get_position_dataset,
    POSITION_FEATURE_COLS,
    TARGET_POSITION,
)

RF_MODEL_PATH = Path("./models/position_rf.joblib")
XGB_MODEL_PATH = Path("./models/position_xgb.joblib")


def eval_classifier(y_true, y_pred, label_names, label: str):
    """Print accuracy, classification report and confusion matrix."""
    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== {label} – position_10 metrics ===")
    print(f"Accuracy: {acc:.3f}")

    print("\nClassification report (macro & per-class F1):")
    print(classification_report(y_true, y_pred, target_names=label_names, digits=3))

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    print("\nConfusion matrix (rows = true, cols = predicted):")
    print(cm_df)

    return {"accuracy": acc, "confusion_matrix": cm_df}


def train_position_models(test_size: float = 0.2, random_state: int = 42):
    # 1) Load dataset
    X, y_raw, df = get_position_dataset()
    print(f"Loaded dataset for position prediction: {df.shape[0]} rows")
    print(f"Number of features: {len(POSITION_FEATURE_COLS)}")
    print(f"Target: {TARGET_POSITION}")

    print("\n=== Class distribution (position_10) ===")
    print(pd.Series(y_raw).value_counts())

    # 2) Encode string labels -> integers
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = list(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # ---------------------------------------------------------
    # 3) Random Forest baseline (unchanged, but uses new features)
    # ---------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    print("\nTraining Random Forest baseline for position prediction...")
    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)
    rf_metrics = eval_classifier(
        y_test, y_pred_rf, class_names, label="Random Forest"
    )

    RF_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": rf,
            "feature_cols": POSITION_FEATURE_COLS,
            "target": TARGET_POSITION,
            "model_type": "random_forest_classifier",
            "label_encoder": le,
        },
        RF_MODEL_PATH,
    )
    print(f"\n[Saved] Random Forest position model → {RF_MODEL_PATH.resolve()}")

    # ---------------------------------------------------------
    # 4) XGBoost classifier – optimised
    # ---------------------------------------------------------
    # Class-balanced sample weights for XGB
    class_counts = np.bincount(y_train)
    class_weights = {cls: len(y_train) / (len(class_counts) * count)
                     for cls, count in enumerate(class_counts)}
    sample_weight = np.array([class_weights[cls] for cls in y_train])

    xgb = XGBClassifier(
        n_estimators=600,      
        max_depth=6,           
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=2.0,
        reg_alpha=0.5,
        objective="multi:softprob",
        num_class=len(class_names),
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )

    print("\nTraining XGBoost for position prediction (with class weights)...")
    xgb.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_test, y_test)],
        verbose=True,
    )

    # Train vs Test accuracy
    y_train_pred = xgb.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print("\n=== XGBoost – Train vs Test (position_10) ===")
    print(f"Train accuracy: {train_acc:.3f}")

    y_pred_xgb = xgb.predict(X_test)
    xgb_metrics = eval_classifier(
        y_test, y_pred_xgb, class_names, label="XGBoost"
    )

    print("\n=== RF vs XGB – Accuracy comparison ===")
    print(f"Random Forest accuracy : {rf_metrics['accuracy']:.3f}")
    print(f"XGBoost      accuracy  : {xgb_metrics['accuracy']:.3f}")

    # Save XGB model
    XGB_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": xgb,
            "feature_cols": POSITION_FEATURE_COLS,
            "target": TARGET_POSITION,
            "model_type": "xgboost_classifier",
            "label_encoder": le,
            "classes_": class_names,
        },
        XGB_MODEL_PATH,
    )
    print(f"\n[Saved] XGBoost position model → {XGB_MODEL_PATH.resolve()}")

    # Feature importances
    importances = xgb.feature_importances_
    feat_imp = sorted(
        zip(POSITION_FEATURE_COLS, importances),
        key=lambda x: x[1],
        reverse=True,
    )

    print("\n=== Top 15 XGBoost feature importances (position) ===")
    for name, score in feat_imp[:15]:
        print(f"{name:30s}: {score:.4f}")


if __name__ == "__main__":
    train_position_models()
