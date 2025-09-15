
# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st

from macro_core import DataLoad, DataTransform, LayerBuilder, SeriesSpec
from macro_specs import build_default_specs

import os, streamlit as st

# Carga el secret y lo exporta como variable de entorno
if "FRED_API_KEY" in st.secrets:
    os.environ["FRED_API_KEY"] = st.secrets["FRED_API_KEY"]

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

    specs = build_default_specs(target_freq=freq, z_win=z_window)

    # ----- Liquidity con fallback -----
    liq_df = L.build(specs["Liquidity"])
    need_fallback = ("NFCI" not in liq_df.columns) or liq_df["NFCI"].dropna().empty
    if need_fallback:
        st.warning("NFCI vacío; usando ANFCI (Adjusted NFCI) como fallback.")
        liq_alt = specs["Liquidity"]
        liq_alt.series = [
            s if s.alias != "NFCI" else SeriesSpec(
                fred_id="ANFCI", alias="NFCI", units="level", freq_objetivo="W-MON", agg="last"
            )
            for s in liq_alt.series
        ]
        liq_df = L.build(liq_alt)

    layers = {"Liquidity": liq_df}
    # Resto de capas
    for name, spec in specs.items():
        if name == "Liquidity":
            continue
        layers[name] = L.build(spec)

    # Recorte por fechas
    for k in layers:
        layers[k] = layers[k].loc[pd.to_datetime(dmin):pd.to_datetime(dmax)]
    return layers

layers = build_layers(freq, z_window, date_min, date_max)

liq = layers.get("Liquidity")
need_nfci = (liq is None) or (("NFCI" not in liq.columns) and ("NFCI_z" not in liq.columns))
if need_nfci:
    st.warning("Liquidity sin NFCI; descargando NFCI directo (fallback).")
    nfci = DataLoad().get_one("NFCI").asfreq("W-FRI").ffill()
    nfci["NFCI_z"] = (nfci["NFCI"] - nfci["NFCI"].rolling(52).mean()) / nfci["NFCI"].rolling(52).std()
    nfci = nfci.loc[pd.to_datetime(date_min):pd.to_datetime(date_max)]
    layers["Liquidity"] = nfci if liq is None else pd.concat([liq, nfci], axis=1)
liq = layers["Liquidity"]

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
z_val = None
if "NFCI_z" in liq.columns:
    _z = liq["NFCI_z"].dropna()
    if not _z.empty:
        z_val = _z.iloc[-1]

lvl_text = None
if "NFCI" in liq.columns:
    _lvl = liq["NFCI"].dropna()
    if not _lvl.empty:
        lvl_text = f"Nivel: {_lvl.iloc[-1]:+.2f}"
st.write("Cols Liquidity actuales:", list(layers.get("Liquidity", pd.DataFrame()).columns))

def _nfci_view(layers_dict, freq_out, dmin, dmax, z_win=52):
    """Devuelve un DF con columnas NFCI (nivel) y NFCI_z, venga o no en la capa."""
    liq = layers_dict.get("Liquidity")
    df = None

    # 1) intenta tomar de la capa Liquidity lo que exista
    if liq is not None and isinstance(liq, pd.DataFrame):
        cols = [c for c in ["NFCI", "NFCI_level", "NFCI_z"] if c in liq.columns]
        if cols:
            df = liq[cols].copy()

    # 2) si no hay nada útil, baja NFCI directo de FRED
    if df is None or df.dropna(how="all").empty:
        raw = DataLoad().get_one("NFCI")              # nivel
        raw = raw.asfreq("W-FRI").ffill()             # grilla semanal (viernes)
        raw["NFCI_z"] = (raw["NFCI"] - raw["NFCI"].rolling(z_win).mean()) / raw["NFCI"].rolling(z_win).std()
        df = raw

    # 3) recorte por fechas
    df = df.loc[pd.to_datetime(dmin):pd.to_datetime(dmax)]

    # 4) remuestreo SOLO para vista
    if freq_out in ("M", "ME"):
        df = df.resample("ME").last()
    elif freq_out != "W-FRI":
        df = df.resample(freq_out).last()

    # asegura columnas estándar
    if "NFCI" not in df.columns and "NFCI_level" in df.columns:
        df["NFCI"] = df["NFCI_level"]

    return df[["NFCI"]] if "NFCI_z" not in df.columns else df[["NFCI", "NFCI_z"]]

# construye la vista segura
nfci_df = _nfci_view(layers, freq, date_min, date_max, z_window)

# valor z si existe, sino NaN
z_val = float("nan")
if "NFCI_z" in nfci_df.columns:
    _z = nfci_df["NFCI_z"].dropna()
    if not _z.empty:
        z_val = _z.iloc[-1]

# valor de nivel (caption)
lvl_caption = None
if "NFCI" in nfci_df.columns:
    _lvl = nfci_df["NFCI"].dropna()
    if not _lvl.empty:
        lvl_caption = f"NFCI nivel: {_lvl.iloc[-1]:+.2f}"

metric_box(c2, "NFCI (z 52s)", z_val)
if lvl_caption:
    st.caption(lvl_caption)

# si tu capa Liquidity venía vacía, al menos que el tab tenga algo
if "Liquidity" not in layers or layers["Liquidity"].dropna(how="all", axis=1).empty:
    layers["Liquidity"] = nfci_df
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
