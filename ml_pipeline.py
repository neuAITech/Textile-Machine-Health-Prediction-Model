"""
=============================================================================
 MONITORING YARN MACHINE HEALTH AND EFFICIENCY - Complete ML Pipeline
 =============================================================================
 Dataset : Textile Manufacturing Machine Monitoring (5001 rows × 30 cols)
 Targets :
   1. Target_RUL_Hours       → Regression  (Remaining Useful Life in hours)
   2. Failure_Imminent_Flag   → Binary      (1 = failure likely soon)
   3. Target_Failure_24H      → Binary      (1 = failure within 24 hours)
   4. Failure_Mode_Code       → Multi‑class (0‑Healthy … 3‑Critical)
 Models  : Random Forest, Gradient Boosting, XGBoost, LightGBM,
           Extra Trees, Ridge / Logistic Regression
=============================================================================
"""

import os, warnings, time, json, joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

DIVIDER = "=" * 80

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & EXPLORATION
# ═══════════════════════════════════════════════════════════════════════════════
def load_and_explore(path: str) -> pd.DataFrame:
    print(f"\n{DIVIDER}")
    print("  STEP 1 ▸ DATA LOADING & EXPLORATION")
    print(DIVIDER)

    df = pd.read_csv(path, parse_dates=["Timestamp"])
    print(f"  Shape           : {df.shape}")
    print(f"  Columns         : {df.columns.tolist()}")
    print(f"  Dtypes overview :\n{df.dtypes.value_counts().to_string()}")
    print(f"  Missing values  : {df.isnull().sum().sum()}")
    print(f"\n  Numeric stats (sample):")
    print(df.describe().T[["mean","std","min","max"]].head(10).to_string())
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{DIVIDER}")
    print("  STEP 2 ▸ FEATURE ENGINEERING")
    print(DIVIDER)

    # ── target variables ─────────────────────────────────────────────────────
    # 1) Target_RUL_Hours  (already present as RUL_Hours)
    df["Target_RUL_Hours"] = df["RUL_Hours"]

    # 2) Failure_Imminent_Flag: 1 when component health is dangerously low
    #    RUL < 1250 captures ~bottom 30-35% (moderate/critical wear zone)
    df["Failure_Imminent_Flag"] = (df["RUL_Hours"] < 1250).astype(int)

    # 3) Target_Failure_24H: high-risk machines likely to fail soon
    #    RUL <= 1100 captures critical & severe moderate wear (~bottom 15%)
    df["Target_Failure_24H"] = (df["RUL_Hours"] <= 1100).astype(int)

    # 4) Failure_Mode_Code: ordinal from Degradation_Phase
    phase_map = {"Healthy": 0, "Early Wear": 1, "Moderate Wear": 2, "Critical": 3}
    df["Failure_Mode_Code"] = df["Degradation_Phase"].map(phase_map)

    # ── time features ────────────────────────────────────────────────────────
    df["Hour"]        = df["Timestamp"].dt.hour
    df["DayOfWeek"]   = df["Timestamp"].dt.dayofweek
    df["DayOfMonth"]  = df["Timestamp"].dt.day

    # ── interaction features ─────────────────────────────────────────────────
    df["Temp_x_Friction"]    = df["Temp_C"] * df["Mechanical_Friction_Index"]
    df["Speed_x_Torque"]     = df["Machine_Speed_RPM"] * df["Torque_Load_Index"]
    df["Current_x_Voltage"]  = df["Motor_Current_A"] * df["Voltage_Variation_%"]
    df["Energy_per_Output"]  = df["Energy_kWh"] / (df["Output_kg"] + 1)
    df["Waste_Ratio"]        = df["Waste_kg"] / (df["Output_kg"] + 1)

    # ── encode categoricals ──────────────────────────────────────────────────
    cat_cols = ["Machine_ID", "Machine_Type", "Section", "Machine_State", "Shift"]
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # save encoders
    joblib.dump(label_encoders, os.path.join(MODEL_DIR, "label_encoders.pkl"))

    # ── print target distributions ───────────────────────────────────────────
    print("  Target distributions:")
    print(f"    Failure_Mode_Code   : {df['Failure_Mode_Code'].value_counts().sort_index().to_dict()}")
    print(f"    Failure_Imminent_Flag: {df['Failure_Imminent_Flag'].value_counts().to_dict()}")
    print(f"    Target_Failure_24H  : {df['Target_Failure_24H'].value_counts().to_dict()}")
    print(f"    Target_RUL_Hours    : mean={df['Target_RUL_Hours'].mean():.1f}  std={df['Target_RUL_Hours'].std():.1f}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PREPARE FEATURES & SPLITS
# ═══════════════════════════════════════════════════════════════════════════════
def prepare_data(df: pd.DataFrame):
    print(f"\n{DIVIDER}")
    print("  STEP 3 ▸ PREPARING FEATURES & TRAIN / TEST SPLIT")
    print(DIVIDER)

    # columns to drop (identifiers, raw categoricals, targets, leakage cols)
    drop_cols = [
        "Timestamp", "Machine_ID", "Machine_Type", "Section",
        "Machine_State", "Shift", "Degradation_Phase",
        # targets
        "Target_RUL_Hours", "Failure_Imminent_Flag",
        "Target_Failure_24H", "Failure_Mode_Code",
        # leakage — these directly reveal the targets
        "RUL_Hours", "Component_Health_%",
    ]

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    print(f"  Feature columns ({len(feature_cols)}): {feature_cols}")

    # scale
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    targets = {
        "Failure_Mode_Code":    df["Failure_Mode_Code"],
        "Failure_Imminent_Flag": df["Failure_Imminent_Flag"],
        "Target_Failure_24H":   df["Target_Failure_24H"],
        "Target_RUL_Hours":     df["Target_RUL_Hours"],
    }

    splits = {}
    for name, y in targets.items():
        stratify = y if name != "Target_RUL_Hours" else None
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=stratify
        )
        splits[name] = (X_tr, X_te, y_tr, y_te)
        print(f"  {name:30s}  train={len(X_tr)}  test={len(X_te)}")

    return splits, feature_cols


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════
def get_classifiers():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=20, min_samples_split=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, n_jobs=-1, verbosity=0
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, max_depth=10, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=42,
            n_jobs=-1, verbose=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, max_depth=20, min_samples_split=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced",
            solver="lbfgs", random_state=42, n_jobs=-1
        ),
    }

def get_regressors():
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=20, min_samples_split=5,
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbosity=0
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=300, max_depth=10, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300, max_depth=20, min_samples_split=5,
            random_state=42, n_jobs=-1
        ),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
def train_classification(target_name, X_tr, X_te, y_tr, y_te):
    """Train all classifiers, return results DataFrame and best model."""
    print(f"\n{'─'*80}")
    print(f"  ▸ CLASSIFICATION TARGET : {target_name}")
    print(f"{'─'*80}")

    models = get_classifiers()
    rows = []

    best_f1, best_model, best_name = -1, None, ""

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        train_time = time.time() - t0

        y_pred = model.predict(X_te)

        avg = "binary" if len(set(y_te)) == 2 else "weighted"
        acc  = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, average=avg, zero_division=0)
        rec  = recall_score(y_te, y_pred, average=avg, zero_division=0)
        f1   = f1_score(y_te, y_pred, average=avg, zero_division=0)

        # cross-val (safe — skip if too few samples per class)
        try:
            cv_scores = cross_val_score(model, X_tr, y_tr, cv=5, scoring="f1_weighted", n_jobs=-1)
        except Exception:
            cv_scores = np.array([f1])  # fallback

        rows.append({
            "Model": name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1": f1,
            "CV_F1_mean": cv_scores.mean(), "CV_F1_std": cv_scores.std(),
            "Train_sec": round(train_time, 2),
        })

        status = ""
        if f1 > best_f1:
            best_f1, best_model, best_name = f1, model, name
            status = "  ★ BEST"

        print(f"    {name:25s}  Acc={acc:.4f}  F1={f1:.4f}  CV_F1={cv_scores.mean():.4f}{status}")

    # confusion matrix of best
    y_best = best_model.predict(X_te)
    print(f"\n  Best Model: {best_name} (F1={best_f1:.4f})")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_te, y_best)}")
    print(f"\n  Classification Report:\n{classification_report(y_te, y_best, zero_division=0)}")

    results_df = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    return results_df, best_model, best_name


def train_regression(target_name, X_tr, X_te, y_tr, y_te):
    """Train all regressors, return results DataFrame and best model."""
    print(f"\n{'─'*80}")
    print(f"  ▸ REGRESSION TARGET : {target_name}")
    print(f"{'─'*80}")

    models = get_regressors()
    rows = []
    best_r2, best_model, best_name = -999, None, ""

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        train_time = time.time() - t0

        y_pred = model.predict(X_te)
        mae  = mean_absolute_error(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        r2   = r2_score(y_te, y_pred)

        try:
            cv_scores = cross_val_score(model, X_tr, y_tr, cv=5, scoring="r2", n_jobs=-1)
        except Exception:
            cv_scores = np.array([r2])

        rows.append({
            "Model": name, "MAE": mae, "RMSE": rmse, "R2": r2,
            "CV_R2_mean": cv_scores.mean(), "CV_R2_std": cv_scores.std(),
            "Train_sec": round(train_time, 2),
        })

        status = ""
        if r2 > best_r2:
            best_r2, best_model, best_name = r2, model, name
            status = "  ★ BEST"

        print(f"    {name:25s}  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}  CV_R²={cv_scores.mean():.4f}{status}")

    print(f"\n  Best Model: {best_name} (R²={best_r2:.4f})")

    results_df = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    return results_df, best_model, best_name


# ═══════════════════════════════════════════════════════════════════════════════
# 6. SAVE MODELS & RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
def save_artifacts(target_name, results_df, best_model, best_model_name):
    safe = target_name.replace(" ", "_")
    # results CSV
    csv_path = os.path.join(RESULT_DIR, f"{safe}_results.csv")
    results_df.to_csv(csv_path, index=False)
    # model pkl
    pkl_path = os.path.join(MODEL_DIR, f"{safe}_best_model.pkl")
    joblib.dump(best_model, pkl_path)
    print(f"  ✓ Results  → {csv_path}")
    print(f"  ✓ Model    → {pkl_path}  ({best_model_name})")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. FEATURE IMPORTANCE (top‑10)
# ═══════════════════════════════════════════════════════════════════════════════
def show_feature_importance(model, feature_cols, target_name):
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_cols)
        imp = imp.sort_values(ascending=False).head(10)
        print(f"\n  Top-10 features for {target_name}:")
        for i, (feat, val) in enumerate(imp.items(), 1):
            bar = "█" * int(val / imp.max() * 30)
            print(f"    {i:2d}. {feat:30s} {val:.4f}  {bar}")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. PREDICTION DEMO
# ═══════════════════════════════════════════════════════════════════════════════
def prediction_demo(splits, best_models, feature_cols):
    print(f"\n{DIVIDER}")
    print("  STEP 6 ▸ SAMPLE PREDICTIONS (first 10 test rows)")
    print(DIVIDER)

    # take the test set from any target (same X split – but we use Failure_Mode_Code's)
    X_te = splits["Failure_Mode_Code"][1]
    sample = X_te.head(10)

    preds = pd.DataFrame(index=sample.index)
    for tname, model in best_models.items():
        preds[tname] = model.predict(sample)

    # round regression
    preds["Target_RUL_Hours"] = preds["Target_RUL_Hours"].round(1)

    print(preds.to_string())

    # save full test predictions
    full_preds = pd.DataFrame(index=X_te.index)
    for tname, model in best_models.items():
        full_preds[tname] = model.predict(X_te)
    full_preds["Target_RUL_Hours"] = full_preds["Target_RUL_Hours"].round(2)
    pred_path = os.path.join(RESULT_DIR, "test_predictions.csv")
    full_preds.to_csv(pred_path, index=False)
    print(f"\n  ✓ Full test predictions saved → {pred_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    start = time.time()

    # 1. Load
    df = load_and_explore(DATA_PATH)

    # 2. Feature engineering
    df = engineer_features(df)

    # 3. Prepare splits
    splits, feature_cols = prepare_data(df)

    # 4 & 5. Train models per target
    print(f"\n{DIVIDER}")
    print("  STEP 4 & 5 ▸ MODEL TRAINING & EVALUATION")
    print(DIVIDER)

    best_models = {}
    all_results = {}
    summary_rows = []

    # ── Classification targets ───────────────────────────────────────────────
    for tname in ["Failure_Mode_Code", "Failure_Imminent_Flag", "Target_Failure_24H"]:
        X_tr, X_te, y_tr, y_te = splits[tname]
        results_df, best_model, best_name = train_classification(tname, X_tr, X_te, y_tr, y_te)
        show_feature_importance(best_model, feature_cols, tname)
        save_artifacts(tname, results_df, best_model, best_name)
        best_models[tname] = best_model
        all_results[tname] = results_df
        top = results_df.iloc[0]
        summary_rows.append({
            "Target": tname, "Best_Model": top["Model"],
            "Accuracy": top["Accuracy"], "F1": top["F1"],
            "CV_F1_mean": top["CV_F1_mean"],
        })

    # ── Regression target ────────────────────────────────────────────────────
    tname = "Target_RUL_Hours"
    X_tr, X_te, y_tr, y_te = splits[tname]
    results_df, best_model, best_name = train_regression(tname, X_tr, X_te, y_tr, y_te)
    show_feature_importance(best_model, feature_cols, tname)
    save_artifacts(tname, results_df, best_model, best_name)
    best_models[tname] = best_model
    all_results[tname] = results_df
    top = results_df.iloc[0]
    summary_rows.append({
        "Target": tname, "Best_Model": top["Model"],
        "R2": top["R2"], "MAE": top["MAE"], "RMSE": top["RMSE"],
        "CV_R2_mean": top["CV_R2_mean"],
    })

    # 6. Prediction demo
    prediction_demo(splits, best_models, feature_cols)

    # ── Final Summary ────────────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("  ★  FINAL SUMMARY")
    print(DIVIDER)
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(RESULT_DIR, "final_summary.csv"), index=False)

    elapsed = time.time() - start
    print(f"\n  Total runtime : {elapsed:.1f} seconds")
    print(f"  Models saved  : {MODEL_DIR}")
    print(f"  Results saved : {RESULT_DIR}")
    print(f"\n{'═'*80}")
    print("  ✅  PIPELINE COMPLETE — ALL 4 TARGETS TRAINED & EVALUATED")
    print(f"{'═'*80}\n")


if __name__ == "__main__":
    main()
