"""
Train 2 Energy Prediction Models with HIGH-CORRELATION targets:
  1. Energy_Stress_Index    → Energy × Friction / Efficiency  (corr ~0.70)
  2. Energy_Health_Score    → Efficiency × Power_Factor normalized by energy  (corr ~0.73)
"""
import os, warnings, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

df = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])

# ── feature engineering (same as ml_pipeline.py) ────────────────────────────
df["Hour"]       = df["Timestamp"].dt.hour
df["DayOfWeek"]  = df["Timestamp"].dt.dayofweek
df["DayOfMonth"] = df["Timestamp"].dt.day
df["Temp_x_Friction"]   = df["Temp_C"] * df["Mechanical_Friction_Index"]
df["Speed_x_Torque"]    = df["Machine_Speed_RPM"] * df["Torque_Load_Index"]
df["Current_x_Voltage"] = df["Motor_Current_A"] * df["Voltage_Variation_%"]
df["Energy_per_Output"]  = df["Energy_kWh"] / (df["Output_kg"] + 1)
df["Waste_Ratio"]        = df["Waste_kg"] / (df["Output_kg"] + 1)

cat_cols = ["Machine_ID", "Machine_Type", "Section", "Machine_State", "Shift"]
label_encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
for col in cat_cols:
    le = label_encoders[col]
    df[col + "_enc"] = le.transform(df[col].astype(str))

# ═════════════════════════════════════════════════════════════════════════════
# CREATE TARGETS
# ═════════════════════════════════════════════════════════════════════════════
# 1. Energy Stress Index = how much stress energy puts on the machine
#    Formula: Energy × Friction / Efficiency  (higher = worse)
df["Target_Energy_Stress"] = df["Energy_kWh"] * df["Mechanical_Friction_Index"] / (df["Efficiency_Index"] + 0.01)

# 2. Energy Health Score = machine health from energy perspective
#    Formula: Efficiency × PowerFactor × 100 / (1 + Energy/MaxEnergy)  (higher = better)
df["Target_Energy_Health"] = (df["Efficiency_Index"] * df["Power_Factor"] * 100) / (1 + df["Energy_kWh"] / df["Energy_kWh"].max())

# Store max energy for runtime use
energy_max = df["Energy_kWh"].max()
joblib.dump(energy_max, os.path.join(MODEL_DIR, "energy_max.pkl"))

# ── drops ────────────────────────────────────────────────────────────────────
# For Energy Stress: exclude direct components (Energy, Friction, Efficiency) to avoid leakage
# For Energy Health: exclude direct components (Efficiency, PowerFactor, Energy) to avoid leakage
always_drop = [
    "Timestamp", "Machine_ID", "Machine_Type", "Section",
    "Machine_State", "Shift", "Degradation_Phase",
    "Target_Energy_Stress", "Target_Energy_Health",
]


def train_best(X_tr, X_te, y_tr, y_te, target_name):
    models = {
        "XGBoost": XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                                 random_state=42, n_jobs=-1, verbosity=0),
        "LightGBM": LGBMRegressor(n_estimators=500, max_depth=12, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                                   random_state=42, n_jobs=-1, verbose=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=500, max_depth=6,
                                                        learning_rate=0.05, subsample=0.8,
                                                        random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=500, max_depth=25,
                                               min_samples_split=3, random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=500, max_depth=25,
                                            min_samples_split=3, random_state=42, n_jobs=-1),
    }
    best_r2, best_model, best_name = -999, None, ""
    rows = []
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        mae = mean_absolute_error(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        r2 = r2_score(y_te, y_pred)
        rows.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
        tag = ""
        if r2 > best_r2:
            best_r2, best_model, best_name = r2, model, name
            tag = " * BEST"
        print(f"  {name:25s}  MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}{tag}")
    return best_model, best_name, pd.DataFrame(rows)


# ═════════════════ 1. ENERGY STRESS INDEX ════════════════════════════════════
print("=" * 60)
print("  TARGET 1: Energy Stress Index")
print("=" * 60)
stress_drop = always_drop.copy()
stress_features = [c for c in df.columns if c not in stress_drop]
print(f"  Features ({len(stress_features)}): {stress_features[:10]}...")
X = df[stress_features]
y = df["Target_Energy_Stress"]

scaler_s = StandardScaler()
X_scaled = pd.DataFrame(scaler_s.fit_transform(X), columns=stress_features)
X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

best_m_s, best_n_s, res_s = train_best(X_tr, X_te, y_tr, y_te, "Energy_Stress")
joblib.dump(best_m_s, os.path.join(MODEL_DIR, "Target_Energy_Stress_best_model.pkl"))
joblib.dump(scaler_s, os.path.join(MODEL_DIR, "scaler_energy_stress.pkl"))
joblib.dump(stress_features, os.path.join(MODEL_DIR, "energy_stress_features.pkl"))
res_s.to_csv(os.path.join(RESULT_DIR, "Target_Energy_Stress_results.csv"), index=False)
print(f"\n  Best: {best_n_s}\n")


# ═════════════════ 2. ENERGY HEALTH SCORE ════════════════════════════════════
print("=" * 60)
print("  TARGET 2: Energy Health Score")
print("=" * 60)
health_drop = always_drop.copy()
health_features = [c for c in df.columns if c not in health_drop]
print(f"  Features ({len(health_features)}): {health_features[:10]}...")
X2 = df[health_features]
y2 = df["Target_Energy_Health"]

scaler_h = StandardScaler()
X2_scaled = pd.DataFrame(scaler_h.fit_transform(X2), columns=health_features)
X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2_scaled, y2, test_size=0.2, random_state=42)

best_m_h, best_n_h, res_h = train_best(X2_tr, X2_te, y2_tr, y2_te, "Energy_Health")
joblib.dump(best_m_h, os.path.join(MODEL_DIR, "Target_Energy_Health_best_model.pkl"))
joblib.dump(scaler_h, os.path.join(MODEL_DIR, "scaler_energy_health.pkl"))
joblib.dump(health_features, os.path.join(MODEL_DIR, "energy_health_features.pkl"))
res_h.to_csv(os.path.join(RESULT_DIR, "Target_Energy_Health_results.csv"), index=False)
print(f"\n  Best: {best_n_h}")

print("\n" + "=" * 60)
print("  ENERGY MODELS TRAINED SUCCESSFULLY")
print("=" * 60)
