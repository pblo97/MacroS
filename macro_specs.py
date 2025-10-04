# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict
from dataclasses import dataclass
import pandas as pd

from macro_core import (
    DataLoad, DataTransform, LayerBuilder,
    LayerSpec, SeriesSpec, TransformStep
)

# Devuelve un diccionario {nombre: LayerSpec} listo para construir
def build_default_specs(target_freq: str = "W-FRI", z_win: int = 52) -> Dict[str, LayerSpec]:
    # --------------------------
    # Pesos internos del pilar Liquidez
    # --------------------------
    W_NFCI     = 0.40  # nivel de condiciones financieras (pro-riesgo si NFCI_z < 0)
    W_EFFR_IMP = 0.30  # impulso de política (anti-riesgo si EFFR_chg_13w_z > 0)
    W_USD_HY   = 0.30  # estrés de mercado (USD↑ + HY OAS↑)

    # ==========================
    # 1) Yield Curve
    # ==========================
    yc_spec = LayerSpec(
        name="YieldCurve",
        target_freq=target_freq,
        default_fill="ffill",
        series=[
            SeriesSpec(fred_id="DGS10", alias="UST10Y", units="pp_from_percent", freq_objetivo="D", agg="last"),
            SeriesSpec(fred_id="DGS2",  alias="UST2Y",  units="pp_from_percent", freq_objetivo="D", agg="last"),
        ],
        derived=[
            TransformStep(fn="spread", arg={"name":"YC_10_2_pp","a":"UST10Y","b":"UST2Y"}),
            TransformStep(fn="expr", arg={
                "name": "YC_10_2_pp_z",
                "expr": f"(YC_10_2_pp - YC_10_2_pp.rolling({z_win}).mean()) / YC_10_2_pp.rolling({z_win}).std()"
            }),
        ],
        # post puede quedar vacío o con funciones permitidas; evitamos 'expr' aquí
        post=[],
    )

    # ==========================
    # 2) Liquidity (rehecha)
    # ==========================
    liq_spec = LayerSpec(
        name="Liquidity",
        target_freq="W-FRI",         # fijo semanal para alinear y evitar look-ahead
        default_fill="ffill",
        series=[
            # Política/overnight (nivel e impulso)
            SeriesSpec(fred_id="EFFR", alias="EFFR", units="pp_from_percent", freq_objetivo="D", agg="last"),
            # Nivel de condiciones financieras (Chicago Fed)
            SeriesSpec(fred_id="NFCI", alias="NFCI", units="level", freq_objetivo="W-FRI", agg="last"),
            # USD broad (si prefieres DXY, cámbialo por tu feed)
            SeriesSpec(fred_id="DTWEXBGS", alias="USD_Broad", units="level", freq_objetivo="D", agg="last"),
            # HY OAS (en % → bps para coherencia)
            SeriesSpec(fred_id="BAMLH0A0HYM2", alias="HY_OAS_bps", units="bp_from_percent", freq_objetivo="D", agg="last"),
        ],
        derived=[
            # Impulso: EFFR vs media 13 semanas (proxy de apretón reciente)
            TransformStep(fn="expr", arg={
                "name": "EFFR_chg_13w",
                "expr": "EFFR - EFFR.rolling(13).mean()"
            }),

            # -------- z-scores internos (todo en derived) ----------
            TransformStep(fn="expr", arg={
                "name": "NFCI_z",
                "expr": f"(NFCI - NFCI.rolling({z_win}).mean()) / NFCI.rolling({z_win}).std()"
            }),
            TransformStep(fn="expr", arg={
                "name": "USD_Broad_z",
                "expr": f"(USD_Broad - USD_Broad.rolling({z_win}).mean()) / USD_Broad.rolling({z_win}).std()"
            }),
            TransformStep(fn="expr", arg={
                "name": "HY_OAS_bps_z",
                "expr": f"(HY_OAS_bps - HY_OAS_bps.rolling({z_win}).mean()) / HY_OAS_bps.rolling({z_win}).std()"
            }),
            TransformStep(fn="expr", arg={
                "name": "EFFR_chg_13w_z",
                "expr": f"(EFFR_chg_13w - EFFR_chg_13w.rolling({z_win}).mean()) / EFFR_chg_13w.rolling({z_win}).std()"
            }),

            # Composite de estrés de mercado (USD + HY)
            TransformStep(fn="expr", arg={
                "name": "LIQ_USD_HY",
                "expr": "0.5 * USD_Broad_z + 0.5 * HY_OAS_bps_z"
            }),

            # Score interno de Liquidez (promedio ponderado)
            TransformStep(fn="expr", arg={
                "name": "Liquidity_score",
                # Nota:
                # - Usamos (-NFCI_z) porque NFCI negativo = laxo = pro-riesgo.
                #   Al negarlo, valores altos indican más "liquidez"/menos estrés.
                # - EFFR_chg_13w_z positivo = apretón => suma estrés (sin signo menos).
                # - LIQ_USD_HY alto = estrés (coherente).
                "expr": f"(({W_NFCI} * (-NFCI_z)) + ({W_EFFR_IMP} * (EFFR_chg_13w_z)) + ({W_USD_HY} * (LIQ_USD_HY))) / ({W_NFCI + W_EFFR_IMP + W_USD_HY})"
            }),

            # Señal discreta tipo semáforo (opcional)
            TransformStep(fn="expr", arg={
                "name": "Liquidity_sig",
                "expr": "(Liquidity_score > 0.5).astype(int) - (Liquidity_score < -0.5).astype(int)"
            }),
        ],
        post=[],  # <- vacío: nada de 'expr' aquí
    )

    # ==========================
    # 3) Credit
    # ==========================
    cred_spec = LayerSpec(
        name="Credit",
        target_freq=target_freq,
        default_fill="ffill",
        series=[
            # BAA10YM ya es spread en pp; si prefieres bps, usa units="bp_from_percent"
            SeriesSpec(fred_id="BAA10YM", alias="BAA10Y_spread_pp", units="pp_from_percent", freq_objetivo="M", agg="last"),
            SeriesSpec(fred_id="BAMLH0A0HYM2", alias="HY_OAS_bps", units="bp_from_percent", freq_objetivo="D", agg="last"),
        ],
        derived=[
            TransformStep(fn="expr", arg={
                "name": "BAA10Y_spread_pp_z",
                "expr": f"(BAA10Y_spread_pp - BAA10Y_spread_pp.rolling({z_win}).mean()) / BAA10Y_spread_pp.rolling({z_win}).std()"
            }),
            TransformStep(fn="expr", arg={
                "name": "HY_OAS_bps_z",
                "expr": f"(HY_OAS_bps - HY_OAS_bps.rolling({z_win}).mean()) / HY_OAS_bps.rolling({z_win}).std()"
            }),
        ],
        post=[],
    )

    # ==========================
    # 4) Activity
    # ==========================
    act_spec = LayerSpec(
        name="Activity",
        target_freq="M",
        default_fill="ffill",
        series=[
            SeriesSpec(fred_id="INDPRO", alias="INDPRO", units="level", freq_objetivo="M", agg="last"),
        ],
        derived=[
            TransformStep(fn="expr", arg={
                "name": "INDPRO_yoy",
                "expr": "INDPRO.pct_change(12)"
            }),
            TransformStep(fn="expr", arg={
                "name": "INDPRO_yoy_z",
                "expr": f"(INDPRO_yoy - INDPRO_yoy.rolling(36).mean()) / INDPRO_yoy.rolling(36).std()"
            }),
        ],
        post=[],
    )

    # ==========================
    # 5) Inflation
    # ==========================
    inf_spec = LayerSpec(
        name="Inflation",
        target_freq="M",
        default_fill="ffill",
        series=[
            SeriesSpec(fred_id="CPIAUCSL", alias="CPI", units="level", freq_objetivo="M", agg="last"),
        ],
        derived=[
            TransformStep(fn="expr", arg={
                "name": "CPI_yoy",
                "expr": "CPI.pct_change(12)"
            }),
            TransformStep(fn="expr", arg={
                "name": "CPI_yoy_z",
                "expr": f"(CPI_yoy - CPI_yoy.rolling(36).mean()) / CPI_yoy.rolling(36).std()"
            }),
        ],
        post=[],
    )

    return {
        "YieldCurve": yc_spec,
        "Liquidity":  liq_spec,      # <- Liquidez ya integra USD + HY OAS y el score
        "Credit":     cred_spec,
        "Activity":   act_spec,
        "Inflation":  inf_spec,
    }
