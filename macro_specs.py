# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict
from dataclasses import dataclass
import pandas as pd

from macro_core import (
    DataLoad, DataTransform, LayerBuilder,
    LayerSpec, SeriesSpec, TransformStep
)

def build_default_specs(target_freq: str = "W-FRI", z_win: int = 52) -> Dict[str, LayerSpec]:
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
        ],
        post=[
            TransformStep(fn="zscore", arg={"cols":["YC_10_2_pp"], "window": z_win}),
        ],
    )

    # ==========================
    # 2) Liquidity (sin .rolling() en expr)
    # ==========================
    liq_spec = LayerSpec(
        name="Liquidity",
        target_freq="W-FRI",
        default_fill="ffill",
        series=[
            SeriesSpec(fred_id="EFFR",     alias="EFFR",      units="pp_from_percent", freq_objetivo="D",     agg="last"),
            SeriesSpec(fred_id="NFCI",     alias="NFCI",      units="level",           freq_objetivo="W-FRI", agg="last"),
            SeriesSpec(fred_id="DTWEXBGS", alias="USD_Broad", units="level",           freq_objetivo="D",     agg="last"),
            SeriesSpec(fred_id="BAMLH0A0HYM2", alias="HY_OAS_bps", units="bp_from_percent", freq_objetivo="D", agg="last"),
        ],
        derived=[
            # Proxy simple de "impulso" sin .rolling(): cambio 13 semanas.
            # OJO: si tu core NO permite .shift() dentro de expr, elimina esta línea.
            TransformStep(fn="expr", arg={
                "name": "EFFR_chg_13w",
                "expr": "EFFR - EFFR"  # placeholder neutro si .shift no está permitido
            }),
        ],
        post=[
            # Estandariza TODO aquí (internamente usa rolling permitido por 'zscore')
            TransformStep(fn="zscore", arg={"cols": ["EFFR","NFCI","USD_Broad","HY_OAS_bps"], "window": z_win}),
            # Nota: dejamos el composite y el score para fuera del layer,
            # porque 'expr' en post no está permitido por tu macro_core.
        ],
    )
    # Si en tu macro_core SÍ existe un step 'diff' o 'lag', puedes reemplazar el placeholder por:
    #   TransformStep(fn="lag",  arg={"name":"EFFR_lag_13w","col":"EFFR","periods":13})
    #   TransformStep(fn="expr", arg={"name":"EFFR_chg_13w","expr":"EFFR - EFFR_lag_13w"})
    # y luego incluir "EFFR_chg_13w" en el zscore de post.

    # ==========================
    # 3) Credit
    # ==========================
    cred_spec = LayerSpec(
        name="Credit",
        target_freq=target_freq,
        default_fill="ffill",
        series=[
            SeriesSpec(fred_id="BAA10YM", alias="BAA10Y_spread_pp", units="pp_from_percent", freq_objetivo="M", agg="last"),
            SeriesSpec(fred_id="BAMLH0A0HYM2", alias="HY_OAS_bps", units="bp_from_percent", freq_objetivo="D", agg="last"),
        ],
        post=[
            TransformStep(fn="zscore", arg={"cols":"all", "window": z_win}),
        ],
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
        post=[
            TransformStep(fn="yoy", arg={"cols":"all"}),
            TransformStep(fn="zscore", arg={"cols":"all","window":36}),
        ],
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
        post=[
            TransformStep(fn="yoy", arg={"cols":"all"}),
            TransformStep(fn="zscore", arg={"cols":"all","window":36}),
        ],
    )

    return {
        "YieldCurve": yc_spec,
        "Liquidity":  liq_spec,
        "Credit":     cred_spec,
        "Activity":   act_spec,
        "Inflation":  inf_spec,
    }
