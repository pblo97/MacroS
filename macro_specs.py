from __future__ import annotations
from typing import Dict
from macro_core import LayerSpec, SeriesSpec, TransformStep

def build_default_specs(target_freq: str = "W-FRI", z_win: int = 52) -> Dict[str, LayerSpec]:
    # --- Liquidity: fijo semanal y dos columnas (nivel y z) ---
    liq_spec = LayerSpec(
        name="Liquidity",
        target_freq="W-FRI",                 # fijo semanal para evitar NaNs por resample
        default_fill="ffill",
        series=[
            SeriesSpec(fred_id="EFFR", alias="EFFR", units="pp_from_percent", freq_objetivo="D",     agg="last"),
            SeriesSpec(fred_id="NFCI", alias="NFCI_level",            units="level",          freq_objetivo="W-FRI", agg="last"),
        ],
        derived=[
            # z-score sin pisar el nivel; usa la ventana configurada
            TransformStep(fn="expr", arg={
                "name": "NFCI_z",
                "expr": f"(NFCI_level - NFCI_level.rolling({z_win}).mean()) / NFCI_level.rolling({z_win}).std()"
            }),
            TransformStep(fn="expr", arg={"name":"EFFR_chg_13w", "expr":"EFFR - EFFR.rolling(13).mean()"}),
        ],
        post=[],  # no hagas post sobre NFCI para no perder la columna
    )

    # --- el resto igual que tenías, pero usa 'ME' si quieres fin de mes ---
    yc_spec = LayerSpec(
        name="YieldCurve", target_freq=target_freq, default_fill="ffill",
        series=[
            SeriesSpec(fred_id="DGS10", alias="UST10Y", units="pp_from_percent", freq_objetivo="D",  agg="last"),
            SeriesSpec(fred_id="DGS2",  alias="UST2Y",  units="pp_from_percent", freq_objetivo="D",  agg="last"),
        ],
        derived=[TransformStep(fn="spread", arg={"name":"YC_10_2_pp","a":"UST10Y","b":"UST2Y"})],
        post=[TransformStep(fn="zscore", arg={"cols":["YC_10_2_pp"], "window": z_win})],
    )

    cred_spec = LayerSpec(
        name="Credit", target_freq=target_freq, default_fill="ffill",
        series=[
            SeriesSpec(fred_id="BAA10YM",      alias="BAA10Y_spread_pp", units="pp_from_percent", freq_objetivo="ME", agg="last"),
            SeriesSpec(fred_id="BAMLH0A0HYM2", alias="HY_OAS_bps",       units="bp_from_percent", freq_objetivo="D",  agg="last"),
        ],
        post=[TransformStep(fn="zscore", arg={"cols":"all", "window": z_win})],
    )

    act_spec = LayerSpec(
        name="Activity", target_freq="ME", default_fill="ffill",
        series=[SeriesSpec(fred_id="INDPRO", alias="INDPRO", units="level", freq_objetivo="ME", agg="last")],
        post=[TransformStep(fn="yoy", arg={"cols":"all"}), TransformStep(fn="zscore", arg={"cols":"all","window":36})],
    )

    inf_spec = LayerSpec(
        name="Inflation", target_freq="ME", default_fill="ffill",
        series=[SeriesSpec(fred_id="CPIAUCSL", alias="CPI", units="level", freq_objetivo="ME", agg="last")],
        post=[TransformStep(fn="yoy", arg={"cols":"all"}), TransformStep(fn="zscore", arg={"cols":"all","window":36})],
    )

    return {
        "Liquidity":  liq_spec,
        "YieldCurve": yc_spec,
        "Credit":     cred_spec,
        "Activity":   act_spec,
        "Inflation":  inf_spec,
    }