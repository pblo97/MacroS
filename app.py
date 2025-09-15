
# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st

from macro_core import DataLoad, DataTransform, LayerBuilder
from macro_specs import build_default_specs

st.set_page_config(page_title="Macro Monitor", layout="wide")

st.sidebar.header("Controles")
freq = st.sidebar.selectbox("Frecuencia salida", ["W-FRI", "M"], index=0)
z_window = st.sidebar.slider("Ventana Z-score", min_value=24, max_value=260, value=52, step=4)
date_min = st.sidebar.date_input("Desde", pd.Timestamp.today() - pd.DateOffset(years=10))
date_max = st.sidebar.date_input("Hasta", pd.Timestamp.today())
show_percentiles = st.sidebar.checkbox("Mostrar percentiles", value=True)

@st.cache_data(show_spinner=True, ttl=60*30)
def build_layers(freq: str, z_win: int, dmin: pd.Timestamp, dmax: pd.Timestamp):
    T = DataTransform()
    L = LayerBuilder(loader=DataLoad(), T=T)

    specs = build_default_specs(target_freq=freq, z_win=z_win)
    layers = {}
    for name, spec in specs.items():
        df = L.build(spec)
        layers[name] = df.loc[pd.to_datetime(dmin):pd.to_datetime(dmax)]
    return layers

layers = build_layers(freq, z_window, date_min, date_max)

# --- utilidades ---
def last_value(s: pd.Series):
    s2 = s.dropna()
    return s2.iloc[-1] if len(s2) else np.nan

def pct_rank(s: pd.Series, lookback: int = 260):
    s2 = s.dropna().tail(lookback)
    if s2.empty: return np.nan
    return float((s2.rank(pct=True).iloc[-1] * 100))

def metric_box(col, label: str, value, help_txt=None):
    with col:
        st.metric(label, value=None if pd.isna(value) else f"{value:,.2f}")
        if help_txt: st.caption(help_txt)

st.title("Macro Monitor")

# tarjetas
c1,c2,c3,c4,c5,c6 = st.columns(6)
yc = layers["YieldCurve"]
liq = layers["Liquidity"]
cred = layers["Credit"]
act = layers["Activity"]
inf = layers["Inflation"]

metric_box(c1, "YC 10–2 (pp)", last_value(yc["YC_10_2_pp"]))
metric_box(c2, "NFCI (z)", last_value(liq["NFCI"]))
metric_box(c3, "HY OAS (bps, z)", last_value(cred["HY_OAS_bps"]))
metric_box(c4, "EFFR-EMA13w (z)", last_value(liq["EFFR_chg_13w"]))
metric_box(c5, "INDPRO YoY (z)", last_value(act["INDPRO"]))
metric_box(c6, "CPI YoY (z)", last_value(inf["CPI"]))

st.divider()

# régimen simple
def regime_signal(layers):
    sigs = {}
    sigs["Liquidity"]  = +1 if last_value(layers["Liquidity"]["NFCI"]) < 0 else -1
    sigs["Curve"]      = +1 if last_value(layers["YieldCurve"]["YC_10_2_pp"]) > 0 else -1
    sigs["Credit"]     = +1 if last_value(layers["Credit"]["HY_OAS_bps"]) < 0 else -1  # z<0 = compresión
    sigs["Activity"]   = +1 if last_value(layers["Activity"]["INDPRO"]) > 0 else -1
    sigs["Inflation"]  = +1 if last_value(layers["Inflation"]["CPI"]) <= 0 else -1

    df = pd.DataFrame(sigs, index=["signal"]).T
    df["weight"] = [0.30, 0.25, 0.20, 0.15, 0.10]
    df["score"] = df["signal"] * df["weight"]
    return df

reg = regime_signal(layers)
st.subheader("Régimen (semáforo por capa)")
st.dataframe(reg[["signal","weight","score"]])
total_score = float(reg["score"].sum())
st.markdown(f"**Total score:** `{total_score:+.2f}`")
st.caption("Ejemplo: score > 0 → sesgo riesgo; < 0 → sesgo defensivo. Calibra con backtests.")

def playbook(score: float):
    if score >= 0.2:
        return {
            "Bias": "Riesgo ON (growth/EM/commodities pro-cíclicos)",
            "Equities": "Overweight growth/quality; beta moderada-alta; EM selectivo",
            "Rates": "Neutral a ligera infraponderación de duración",
            "Credit": "IG y HY con sesgo a carry",
            "FX": "Carry positivo; USD neutral/ligeramente débil",
            "Commodities": "Cobre, energía si actividad confirmada"
        }
    elif score <= -0.2:
        return {
            "Bias": "Defensivo (flight-to-quality)",
            "Equities": "Underweight cíclicos; tilt a value defensivo",
            "Rates": "Overweight duración (UST/Bund) si inflación contenida",
            "Credit": "Reducir HY; preferir IG corto",
            "FX": "USD fuerte; evitar EM frágiles",
            "Commodities": "Oro sobre pro-cíclicos"
        }
    else:
        return {
            "Bias": "Neutral/táctico",
            "Equities": "Quality/low-vol; selección bottom-up",
            "Rates": "Barbell (corto + algo de duración)",
            "Credit": "IG intermedio; HY selectivo",
            "FX": "Mixto; rotaciones tácticas",
            "Commodities": "Balanceado"
        }

st.subheader("Playbook sugerido")
for k, v in playbook(total_score).items():
    st.write(f"- **{k}:** {v}")

st.divider()

# Tabs por capa
tabs = st.tabs(["Liquidity", "Yield Curve", "Credit", "Activity", "Inflation"])
maps = {
    "Liquidity": layers["Liquidity"],
    "Yield Curve": layers["YieldCurve"],
    "Credit": layers["Credit"],
    "Activity": layers["Activity"],
    "Inflation": layers["Inflation"],
}
for i, (name, df) in enumerate(maps.items()):
    with tabs[i]:
        st.subheader(name)
        cols = df.columns.tolist()
        default_cols = cols[: min(3, len(cols))]
        pick = st.multiselect(f"Series en {name}", cols, default=default_cols, key=f"pick_{name}")
        if pick:
            st.line_chart(df[pick])
            if show_percentiles:
                st.caption("Percentiles (últimos 5 años para semanal / 60 meses para mensual):")
                lookback = 260 if freq.startswith("W") else 60
                percs = {c: pct_rank(df[c], lookback) for c in pick}
                st.json(percs)

st.divider()
st.subheader("Series Explorer")
all_cols = []
for k, df in layers.items():
    all_cols += [f"{k}:{c}" for c in df.columns]
sel = st.selectbox("Serie", all_cols)
layer_name, col = sel.split(":", 1)
st.line_chart(layers[layer_name][col])
