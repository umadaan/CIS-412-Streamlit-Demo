# test_local_model.py
import os, pickle, joblib, numpy as np, pandas as pd
import lightgbm as lgb

ART = "artifacts"
# load data_info.pkl
with open(os.path.join(ART, "data_info.pkl"), "rb") as f:
    data_info = pickle.load(f)

# feature_cols expected in data_info
feature_cols = data_info.get("feature_cols") or data_info.get("expected_columns") or data_info.get("feature_order")
if not feature_cols:
    raise SystemExit("feature_cols not found in data_info.pkl. Open the file and check keys: " + ", ".join(data_info.keys()))

# load scaler if present
scaler_path = os.path.join(ART, "lgb_scaler.joblib")
scaler = None
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    print("Loaded scaler:", scaler_path)
else:
    print("No scaler file found at", scaler_path, "- continuing without scaler (tree models may expect raw features)")

# load model: try text booster first then joblib
model = None
txt_path = os.path.join(ART, "lgb_model.txt")
joblib_model_path = os.path.join(ART, "lgb_model.joblib")
if os.path.exists(txt_path):
    model = lgb.Booster(model_file=txt_path)
    model_type = "lgb.Booster (text)"
elif os.path.exists(joblib_model_path):
    model = joblib.load(joblib_model_path)
    model_type = "joblib"
else:
    raise SystemExit("No model found in artifacts. Expected lgb_model.txt or lgb_model.joblib")

print("Loaded model type:", model_type)

# prepare a dummy input using numeric_ranges if available, else zeros
numeric_ranges = data_info.get("numeric_ranges", {})
row = {}
for f in feature_cols:
    if f in numeric_ranges:
        row[f] = numeric_ranges[f].get("default", numeric_ranges[f].get("min", 0))
    else:
        row[f] = 0.0

X = pd.DataFrame([row])
print("Input columns:", X.columns.tolist())

# scale if scaler present and model is lgb.Booster trained on scaled data
X_in = X.values
if scaler is not None:
    try:
        X_in = scaler.transform(X)
    except Exception as e:
        print("Scaler transform failed, using raw features. Error:", e)
        X_in = X.values

# predict
if isinstance(model, lgb.Booster):
    pred = model.predict(X_in)
else:
    pred = model.predict(X_in)
print("Prediction:", pred[0])
print("Direction:", "UP" if pred[0] > 0 else "DOWN")
