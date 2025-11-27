# app.py
# UI-only refactor — keeps all logic unchanged (model loading, feature engineering, indication).
# Replace your existing app.py with this file. Only presentation changed.

import os
import pickle
from pathlib import Path
import joblib
import sys
import inspect

import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.preprocessing import StandardScaler

# ---------- Configuration (edit if needed) ----------
CSV_PATH = r"/Users/utkarshmadaan/Desktop/CIS412_finalproject/AAPL(80-24) Final.csv"
ART_DIR = Path("artifacts")
FEATURE_COLS = ['ma_5','ma_20','rsi_14','log_return','lag_1','lag_2','lag_3','lag_5','lag_10']
# ----------------------------------------------------

# Page config and base styles
st.set_page_config(page_title="AAPL — Next-Day Indicator", layout="wide", initial_sidebar_state="expanded")
# Minimal CSS to tighten spacing and make cards
st.markdown(
    """
    <style>
    .block-container {padding-top: 1rem;}
    .stAlert {margin-top: 0.5rem;}
    .metric-label {font-size:0.85rem; color: #888;}
    .value-card {background-color: #0f1724; padding: 12px; border-radius: 8px; color: #fff;}
    /* header card for recomputed features */
    .feat-header {
        display:flex;
        flex-direction:column;
        gap:4px;
        padding:10px 14px;
        background: linear-gradient(90deg, rgba(15,23,36,0.7), rgba(10,14,20,0.7));
        border-radius:8px;
        margin-bottom:8px;
        border: 1px solid rgba(255,255,255,0.03);
    }
    .feat-title {font-weight:700; font-size:16px; color:#fff;}
    .feat-sub {font-size:13px; color: #9aa3b2;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Helper functions (UNCHANGED) ----------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    roll_up = gain.rolling(period, min_periods=1).mean()
    roll_down = loss.rolling(period, min_periods=1).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100 - 100 / (1 + rs)

def build_features_from_df(df_in):
    df = df_in.copy().reset_index(drop=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    if 'Close' not in df.columns and 'Price' in df.columns:
        df = df.rename(columns={'Price': 'Close'})
    df['return'] = df['Close'].pct_change()
    df['log_return'] = np.log(df['Close']).diff()
    df['rsi_14'] = compute_rsi(df['Close'], period=14)
    df['ma_5'] = df['Close'].rolling(5).mean()
    df['ma_20'] = df['Close'].rolling(20).mean()
    df['lag_1'] = df['return'].shift(1)
    df['lag_2'] = df['return'].shift(2)
    df['lag_3'] = df['return'].shift(3)
    df['lag_5'] = df['return'].shift(5)
    df['lag_10'] = df['return'].shift(10)
    df['next_return'] = df['Close'].pct_change().shift(-1)
    df['target_close'] = df['Close'].shift(-1)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing engineered columns: {missing}")
    df_feat = df[['Date'] + FEATURE_COLS + ['next_return', 'target_close']].dropna().reset_index(drop=True)
    return df_feat

@st.cache_resource
def load_artifacts():
    artifacts = {"scaler": None, "model": None, "data_info": None, "model_type": None}
    scaler_path = ART_DIR / "lgb_scaler.joblib"
    if scaler_path.exists():
        try:
            artifacts['scaler'] = joblib.load(scaler_path)
        except Exception:
            artifacts['scaler'] = None
    model_txt = ART_DIR / "lgb_model.txt"
    model_joblib = ART_DIR / "lgb_model.joblib"
    model_joblib_alt = ART_DIR / "lgb_model.pkl"
    if model_txt.exists():
        try:
            artifacts['model'] = lgb.Booster(model_file=str(model_txt))
            artifacts['model_type'] = 'lgb_booster'
        except Exception:
            artifacts['model'] = None
    elif model_joblib.exists():
        try:
            artifacts['model'] = joblib.load(model_joblib)
            artifacts['model_type'] = type(artifacts['model']).__name__
        except Exception:
            artifacts['model'] = None
    elif model_joblib_alt.exists():
        try:
            artifacts['model'] = joblib.load(model_joblib_alt)
            artifacts['model_type'] = type(artifacts['model']).__name__
        except Exception:
            artifacts['model'] = None
    di = ART_DIR / "data_info.pkl"
    if di.exists():
        try:
            with open(di, "rb") as f:
                artifacts['data_info'] = pickle.load(f)
        except Exception:
            artifacts['data_info'] = None
    return artifacts

def predict_with_model(model, scaler, X_row):
    if isinstance(X_row, pd.DataFrame):
        X_arr = X_row.values
    else:
        X_arr = np.asarray(X_row).reshape(1, -1)
    if scaler is not None:
        try:
            X_scaled = scaler.transform(X_arr)
        except Exception:
            X_scaled = X_arr
    else:
        X_scaled = X_arr
    if isinstance(model, lgb.Booster):
        pred = model.predict(X_scaled)
    else:
        pred = model.predict(X_scaled)
    return float(pred[0])

# ---------- Load artifacts quietly ----------
art = load_artifacts()
scaler = art['scaler']
model = art['model']
data_info = art['data_info']

# ---------- Header (clean; no internals) ----------
st.title("AAPL — Next-Day Return & Price Indicator")
st.markdown(
    "Indicate next-day *return* and implied price from historical feature vectors. "
    "This demo recomputes features from your CSV so you can pick any historical trading day."
)

# Top-level status bar (simple icons, no stack traces)
col_s1, col_s2, col_s3 = st.columns([1,1,2])
with col_s1:
    st.markdown("**Model:**")
    if model is not None:
        st.success("Loaded")
    else:
        st.error("Missing — place model in `artifacts/`")
with col_s2:
    st.markdown("**Scaler:**")
    st.write("Yes" if scaler is not None else "No")
with col_s3:
    st.markdown("**Artifacts info:**")
    if data_info is not None:
        st.write(", ".join(list(data_info.keys())))
    else:
        st.write("data_info not found (optional)")

st.markdown("---")

# ---------- Load CSV (compact block) ----------
csv_col1, csv_col2 = st.columns([3,1])
with csv_col1:
    csv_path_input = st.text_input("CSV path (edit if required)", value=CSV_PATH)
with csv_col2:
    if st.button("Reload CSV"):
        st.experimental_rerun()

csv_file = Path(csv_path_input)
if not csv_file.exists():
    st.error(f"CSV not found at: {csv_file}")
    st.stop()

try:
    df_raw = pd.read_csv(csv_file, parse_dates=['Date'], infer_datetime_format=True)
except Exception as e:
    st.error("Failed to read CSV. Check path and file format.")
    st.stop()

# Attempt rename without noisy logs
if 'Close' not in df_raw.columns and 'Price' in df_raw.columns:
    df_raw = df_raw.rename(columns={'Price': 'Close'})

# Build features (suppress internal details)
try:
    df_feat = build_features_from_df(df_raw)
except Exception as e:
    st.error("Feature creation failed. Ensure CSV has 'Date' and 'Close'/'Price' columns.")
    st.stop()

st.write(f"CSV rows: {len(df_raw)} — usable feature rows after rolling/lagging: {len(df_feat)}")
hist_return_std = float(df_feat['next_return'].std()) if 'next_return' in df_feat.columns else 0.0

# ---------- Main interactive area using tabs ----------
tab_manual, tab_history, tab_info = st.tabs(["Manual Input", "Historical Date", "About & Tips"])

# ---------- Manual Input tab ----------
with tab_manual:
    st.subheader("Manual input — craft a custom feature vector")
    st.markdown("Enter a set of feature values and indicate the next-day return. Useful for experiments.")
    with st.form("manual_form_ui", clear_on_submit=False):
        # arrange inputs in two columns
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            ma_5 = st.number_input("MA5", value=float(df_feat['ma_5'].median()), format="%.6f", step=0.0001)
            rsi_14 = st.number_input("RSI 14", value=float(df_feat['rsi_14'].median()), format="%.3f", min_value=0.0, max_value=100.0)
            lag_1 = st.number_input("lag_1", value=0.0, format="%.6f")
        with c2:
            ma_20 = st.number_input("MA20", value=float(df_feat['ma_20'].median()), format="%.6f", step=0.0001)
            log_return = st.number_input("log_return", value=0.0, format="%.6f")
            lag_2 = st.number_input("lag_2", value=0.0, format="%.6f")
        with c3:
            lag_3 = st.number_input("lag_3", value=0.0, format="%.6f")
            lag_5 = st.number_input("lag_5", value=0.0, format="%.6f")
            lag_10 = st.number_input("lag_10", value=0.0, format="%.6f")

        submit_manual = st.form_submit_button("Indicate (manual)")

    if submit_manual:
        X_row = pd.DataFrame([{
            'ma_5': ma_5, 'ma_20': ma_20, 'rsi_14': rsi_14,
            'log_return': log_return, 'lag_1': lag_1, 'lag_2': lag_2,
            'lag_3': lag_3, 'lag_5': lag_5, 'lag_10': lag_10
        }])
        st.markdown("**Input preview**")
        # Pretty display: Feature / Value with units where meaningful
        units = {
            'ma_5': 'price', 'ma_20': 'price', 'rsi_14': 'index (0-100)',
            'log_return': 'log', 'lag_1': 'return', 'lag_2': 'return',
            'lag_3': 'return', 'lag_5': 'return', 'lag_10': 'return'
        }
        display_df = pd.DataFrame({
            'Feature': X_row.columns,
            'Value': [f"{v:.6f}" if isinstance(v, (int,float,np.floating)) else v for v in X_row.iloc[0].values],
            'Units': [units.get(c, "") for c in X_row.columns]
        })
        # show as a clean two-column table (hide Units if you prefer)
        st.table(display_df[['Feature','Value','Units']].set_index('Feature'))


        if model is None:
            st.error("Model is not loaded. Place model artifact in `artifacts/` and reload.")
        else:
            try:
                pred_val = predict_with_model(model, scaler, X_row)
            except Exception:
                st.error("Indication failed. Check artifacts and scaler compatibility.")
                pred_val = None

            if pred_val is not None:
                # metrics row (direction removed by request)
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Indicated return", f"{pred_val:.6f}", delta=None)
                with m2:
                    st.metric("Indicated return (%)", f"{pred_val*100:.2f}%")

                # movement range
                low = pred_val - hist_return_std
                high = pred_val + hist_return_std
                st.markdown(f"**Estimated movement range (heuristic ±1σ):** {low:.4%} → {high:.4%}")

# ---------- Historical-date tab ----------
with tab_history:
    st.subheader("Historical-date indication (recomputes features from CSV)")
    st.markdown("Pick a historical trading date from your CSV; the app will recompute features for that date and indicate the next trading day.")
    # compact date selector: choose from available dates (shows last N)
    available_dates = df_feat['Date'].dt.date.unique()
    if len(available_dates) > 0:
        default_date = available_dates[-1]
    else:
        default_date = None

    chosen = st.date_input("Pick a date (must be trading day present in CSV)", value=default_date)
    if st.button("Indicate from chosen date"):
        try:
            chosen_ts = pd.Timestamp(chosen)
            mask_exact = df_feat['Date'] == chosen_ts
            if mask_exact.any():
                feat_row = df_feat.loc[mask_exact].iloc[-1]
            else:
                prev_rows = df_feat[df_feat['Date'] <= chosen_ts]
                if len(prev_rows) > 0:
                    feat_row = prev_rows.iloc[-1]
                    st.info(f"No exact row for chosen date; using most recent computed row on {prev_rows['Date'].iloc[-1].date()}")
                else:
                    st.error("No feature data exists for chosen date or earlier. Choose a later date.")
                    feat_row = None
        except Exception:
            st.error("Invalid date chosen.")
            feat_row = None

        if feat_row is not None:
            X_row = pd.DataFrame([feat_row[FEATURE_COLS].to_dict()])

            # ---------- Styled header above recomputed features ----------
            st.markdown(
                """
                <div class="feat-header">
                    <div class="feat-title">Recomputed Feature Vector (used for this indication)</div>
                    <div class="feat-sub">Technical indicators recalculated from CSV for the selected trading date.</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Pretty display for recomputed features (with units)
            units = {
                'ma_5': 'price', 'ma_20': 'price', 'rsi_14': 'index (0-100)',
                'log_return': 'log', 'lag_1': 'return', 'lag_2': 'return',
                'lag_3': 'return', 'lag_5': 'return', 'lag_10': 'return'
            }
            vals = X_row.iloc[0].to_dict()
            display_df = pd.DataFrame({
                'Feature': list(vals.keys()),
                'Value': [f"{vals[k]:.6f}" if pd.notna(vals[k]) and isinstance(vals[k], (int,float,np.floating)) else vals[k] for k in vals.keys()],
                'Units': [units.get(k, "") for k in vals.keys()]
            })
            # nicer visual: big numeric column on the right, units shown
            st.markdown(
                """
                <div style="display:flex; gap:12px; align-items:center;">
                <div style="flex:1">
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.table(display_df[['Feature','Value','Units']].set_index('Feature'))


            if model is None:
                st.error("Model not loaded. Place model in `artifacts/` and reload.")
            else:
                try:
                    pred_val = predict_with_model(model, scaler, X_row)
                except Exception:
                    st.error("Indication failed. Likely artifact/scaler mismatch.")
                    pred_val = None

                if pred_val is not None:
                    # find current_close from raw CSV (robust)
                    current_close = None
                    if 'Date' in df_raw.columns and 'Close' in df_raw.columns:
                        df_local = df_raw.copy()
                        df_local['Date'] = pd.to_datetime(df_local['Date'])
                        mask_exact_raw = df_local['Date'] == chosen_ts
                        if mask_exact_raw.any():
                            current_close = float(df_local.loc[mask_exact_raw, 'Close'].iloc[-1])
                        else:
                            prev_rows_raw = df_local[df_local['Date'] <= chosen_ts]
                            if len(prev_rows_raw) > 0:
                                current_close = float(prev_rows_raw['Close'].iloc[-1])
                                st.info(f"Using close from {prev_rows_raw['Date'].iloc[-1].date()} as latest available.")
                            else:
                                current_close = float(df_local['Close'].iloc[-1])
                                st.warning("No rows on/before chosen date — falling back to latest close.")

                    # show metrics and ranges (direction removed)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Indicated return", f"{pred_val:.6f}", delta=None)
                    with c2:
                        st.metric("Indicated return (%)", f"{pred_val*100:.2f}%")

                    # implied price when close available
                    if current_close is not None:
                        implied = current_close * (1 + pred_val)
                        st.markdown(f"**Current close (CSV):** {current_close:.4f}")
                        st.markdown(f"**Implied next-day close:** {implied:.4f}")

                    # movement range
                    low = pred_val - hist_return_std
                    high = pred_val + hist_return_std
                    st.markdown(f"**Estimated movement range (heuristic ±1σ):** {low:.4%} → {high:.4%}")
                    if current_close is not None:
                        st.markdown(f"**Implied price range:** {current_close*(1+low):.4f} → {current_close*(1+high):.4f}")

                    # actual next-day (if available)
                    if current_close is not None and 'Date' in df_raw.columns and 'Close' in df_raw.columns:
                        df_local = df_raw.copy()
                        df_local['Date'] = pd.to_datetime(df_local['Date'])
                        next_rows = df_local[df_local['Date'] > chosen_ts]
                        if len(next_rows) > 0:
                            actual_next_close = float(next_rows['Close'].iloc[0])
                            actual_ret = (actual_next_close / current_close) - 1.0
                            st.markdown(f"**Actual next-day close (CSV):** {actual_next_close:.4f}")
                            st.markdown(f"**Actual next-day return:** {actual_ret:.6f} ({actual_ret*100:.2f}%)")
                        else:
                            st.info("Actual next-day close not present in CSV (chosen date near end of file).")

# ---------- About & Tips tab (clean) ----------
with tab_info:
    st.subheader("About this demo")
    st.markdown("""
    - This application indicates **next-day return** (not raw price) using historically-recomputed features.
    - The app recomputes features from the CSV to ensure the indication uses the exact same feature construction as your notebook.
    - The movement range is a simple heuristic: indicated return ± historical σ of next-day returns computed from CSV (not a formal confidence interval).
    - To avoid confusing internal messages, this UI intentionally hides internal import/traceback details. If something fails, you'll see friendly errors with guidance.
    """)
    st.markdown("**Artifacts required (place into `artifacts/`):**")
    st.write("- `lgb_model.txt` (preferred) or `lgb_model.joblib`")
    st.write("- `lgb_scaler.joblib` (optional if your model expects scaled inputs)")
    st.write("- `data_info.pkl` (optional metadata)")

st.markdown("---")
