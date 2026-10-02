"""
Diesel Engine Analytics
-----------------------
A simple base project that:
  1. Loads engine data from a CSV (or generates realistic synthetic data)
  2. Computes performance KPIs (power, BSFC, thermal efficiency)
  3. Summarises efficiency by load band
  4. Detects abnormal operating points (Isolation Forest)
  5. Predicts NOx emissions (Random Forest regression)
  6. Saves charts and a processed CSV

Install:  pip install pandas numpy matplotlib scikit-learn
Run:      python diesel_engine_analytics.py                 (synthetic data)
          python diesel_engine_analytics.py --csv mydata.csv (your own data)

Expected CSV columns:
  rpm, torque_nm, fuel_flow_kgph, exhaust_temp_c, coolant_temp_c,
  oil_pressure_bar, nox_ppm, smoke_pct
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

LHV_DIESEL_KJ_PER_KG = 42_500   # lower heating value of diesel
MAX_TORQUE_NM = 800             # rated torque of the demo engine
OUT_DIR = "output"


# ---------------------------------------------------------------- data ----
def generate_data(n=2000, seed=42):
    """Create synthetic diesel engine telemetry (with a few injected faults)."""
    rng = np.random.default_rng(seed)
    rpm = rng.uniform(900, 2200, n)
    load = rng.uniform(10, 100, n)                      # % load
    torque = load / 100 * MAX_TORQUE_NM * rng.normal(1, 0.02, n)

    power_kw = torque * 2 * np.pi * rpm / 60 / 1000
    # brake thermal efficiency rises with load, peaks ~ 40 %
    eff = 0.22 + 0.18 * (1 - np.exp(-load / 35)) - 0.00002 * (rpm - 1600) ** 2 / 100
    eff = np.clip(eff + rng.normal(0, 0.005, n), 0.15, 0.42)
    fuel_kgph = power_kw / (eff * LHV_DIESEL_KJ_PER_KG) * 3600

    exhaust = 180 + 3.2 * load + 0.05 * rpm + rng.normal(0, 8, n)
    coolant = 80 + 0.08 * load + rng.normal(0, 2, n)
    oil_p = 1.5 + 0.0016 * rpm + rng.normal(0, 0.1, n)
    nox = 150 + 9 * load + 0.2 * rpm + 0.4 * (exhaust - 400) + rng.normal(0, 40, n)
    smoke = 2 + 0.08 * load + rng.normal(0, 0.5, n)

    df = pd.DataFrame({
        "rpm": rpm, "torque_nm": torque, "fuel_flow_kgph": fuel_kgph,
        "exhaust_temp_c": exhaust, "coolant_temp_c": coolant,
        "oil_pressure_bar": oil_p, "nox_ppm": nox, "smoke_pct": smoke,
    })

    # inject faults: overheating + low oil pressure + injector problem
    bad = rng.choice(n, 40, replace=False)
    df.loc[bad[:15], "coolant_temp_c"] += 25
    df.loc[bad[15:30], "oil_pressure_bar"] *= 0.45
    df.loc[bad[30:], ["fuel_flow_kgph", "smoke_pct"]] *= 1.5
    return df


def load_data(csv_path=None):
    if csv_path:
        df = pd.read_csv(csv_path)
        required = {"rpm", "torque_nm", "fuel_flow_kgph", "exhaust_temp_c",
                    "coolant_temp_c", "oil_pressure_bar", "nox_ppm", "smoke_pct"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing columns: {sorted(missing)}")
        return df.dropna()
    return generate_data()


# ----------------------------------------------------------------- KPIs ----
def add_kpis(df):
    df = df.copy()
    df["power_kw"] = df["torque_nm"] * 2 * np.pi * df["rpm"] / 60 / 1000
    df["load_pct"] = (df["torque_nm"] / MAX_TORQUE_NM * 100).clip(0, 120)
    df["bsfc_g_kwh"] = df["fuel_flow_kgph"] * 1000 / df["power_kw"]
    fuel_power_kw = df["fuel_flow_kgph"] / 3600 * LHV_DIESEL_KJ_PER_KG
    df["thermal_eff_pct"] = df["power_kw"] / fuel_power_kw * 100
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def efficiency_by_load(df):
    bins = [0, 20, 40, 60, 80, 120]
    labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80%+"]
    df["load_band"] = pd.cut(df["load_pct"], bins=bins, labels=labels)
    return df.groupby("load_band", observed=True).agg(
        samples=("rpm", "count"),
        avg_power_kw=("power_kw", "mean"),
        avg_bsfc=("bsfc_g_kwh", "mean"),
        avg_eff_pct=("thermal_eff_pct", "mean"),
        avg_nox_ppm=("nox_ppm", "mean"),
    ).round(2)


# ------------------------------------------------------ anomaly detection ---
FEATURES = ["rpm", "torque_nm", "fuel_flow_kgph", "exhaust_temp_c",
            "coolant_temp_c", "oil_pressure_bar", "smoke_pct", "bsfc_g_kwh"]


def detect_anomalies(df, contamination=0.03):
    model = IsolationForest(contamination=contamination, random_state=42)
    df = df.copy()
    df["anomaly"] = model.fit_predict(df[FEATURES]) == -1
    df["anomaly_score"] = -model.score_samples(df[FEATURES])
    return df


def diagnose(row):
    """Very simple rule-based reason for a flagged row."""
    reasons = []
    if row["coolant_temp_c"] > 100:
        reasons.append("High coolant temp")
    if row["oil_pressure_bar"] < 2.0:
        reasons.append("Low oil pressure")
    if row["smoke_pct"] > 12 or row["bsfc_g_kwh"] > 330:
        reasons.append("Poor combustion / injector issue")
    return ", ".join(reasons) or "Unusual combination of values"


# --------------------------------------------------------- NOx prediction ---
def train_nox_model(df):
    X = df[["rpm", "load_pct", "exhaust_temp_c", "coolant_temp_c"]]
    y = df["nox_ppm"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=1)
    model = RandomForestRegressor(n_estimators=150, random_state=1)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print(f"NOx model  ->  R2: {r2_score(y_te, pred):.3f} | "
          f"MAE: {mean_absolute_error(y_te, pred):.1f} ppm")
    importance = pd.Series(model.feature_importances_, index=X.columns)
    return model, y_te, pred, importance.sort_values(ascending=False)


# ------------------------------------------------------------------ plots ---
def make_plots(df, y_te, pred, importance):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Diesel Engine Analytics", fontsize=16, fontweight="bold")

    sc = ax[0, 0].scatter(df["load_pct"], df["thermal_eff_pct"],
                          c=df["rpm"], cmap="viridis", s=8)
    ax[0, 0].set(title="Thermal Efficiency vs Load", xlabel="Load (%)",
                 ylabel="Efficiency (%)")
    fig.colorbar(sc, ax=ax[0, 0], label="RPM")

    ok, bad = df[~df["anomaly"]], df[df["anomaly"]]
    ax[0, 1].scatter(ok["coolant_temp_c"], ok["oil_pressure_bar"],
                     s=8, label="Normal", alpha=0.5)
    ax[0, 1].scatter(bad["coolant_temp_c"], bad["oil_pressure_bar"],
                     s=18, color="red", label="Anomaly")
    ax[0, 1].set(title="Anomaly Detection", xlabel="Coolant Temp (°C)",
                 ylabel="Oil Pressure (bar)")
    ax[0, 1].legend()

    ax[1, 0].scatter(y_te, pred, s=8, alpha=0.6)
    lims = [y_te.min(), y_te.max()]
    ax[1, 0].plot(lims, lims, "r--")
    ax[1, 0].set(title="NOx: Actual vs Predicted", xlabel="Actual (ppm)",
                 ylabel="Predicted (ppm)")

    importance.sort_values().plot.barh(ax=ax[1, 1], color="teal")
    ax[1, 1].set(title="NOx Model: Feature Importance")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "diesel_dashboard.png")
    plt.savefig(path, dpi=130)
    plt.close()
    return path


# ------------------------------------------------------------------- main ---
def main():
    parser = argparse.ArgumentParser(description="Diesel Engine Analytics")
    parser.add_argument("--csv", help="Path to engine data CSV (optional)")
    args = parser.parse_args()

    df = add_kpis(load_data(args.csv))
    print(f"Loaded {len(df)} records\n")

    print("=== Overall KPIs ===")
    print(df[["power_kw", "bsfc_g_kwh", "thermal_eff_pct", "nox_ppm"]]
          .describe().loc[["mean", "min", "max"]].round(2), "\n")

    print("=== Performance by Load Band ===")
    print(efficiency_by_load(df), "\n")

    df = detect_anomalies(df)
    flagged = df[df["anomaly"]].copy()
    flagged["diagnosis"] = flagged.apply(diagnose, axis=1)
    print(f"=== Anomalies: {len(flagged)} flagged "
          f"({len(flagged) / len(df):.1%}) ===")
    print(flagged.sort_values("anomaly_score", ascending=False)
          [["rpm", "coolant_temp_c", "oil_pressure_bar", "bsfc_g_kwh", "diagnosis"]]
          .head(8).round(1), "\n")

    _, y_te, pred, importance = train_nox_model(df)

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "processed_engine_data.csv"), index=False)
    flagged.to_csv(os.path.join(OUT_DIR, "anomalies.csv"), index=False)
    chart = make_plots(df, y_te, pred, importance)
    print(f"\nSaved results to '{OUT_DIR}/' (dashboard: {chart})")


if __name__ == "__main__":
    main()
