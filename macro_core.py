
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal, Union

import numpy as np
import pandas as pd
import requests

Number = Union[int, float, np.number]

# =============================================================
# DataLoad: FRED -> DataFrame crudo (índice datetime, col con id)
# =============================================================

class DataLoad:
    def __init__(self, api_key: Optional[str] = None, timeout: int = 20, retries: int = 3, backoff: float = 1.7):
        self.api_key = api_key or os.getenv("FRED_API_KEY", "02ea49012ba021ea89f1110c48de7380")
        self.timeout = int(timeout)
        self.retries = int(retries)
        self.backoff = float(backoff)
        self.url = "https://api.stlouisfed.org/fred/series/observations"

    def request_observations(self, series_id: str, **extra_params) -> Dict[str, Any]:
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }
        params.update({k: v for k, v in extra_params.items() if v is not None})

        last_err = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = requests.get(self.url, params=params, timeout=self.timeout)
                if 200 <= resp.status_code < 300:
                    return resp.json()
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except Exception as e:
                last_err = str(e)
            time.sleep(self.backoff * attempt)

        raise RuntimeError(f"Fallo API FRED [{series_id}]: {last_err}")

    def _to_frame(self, payload: dict, series_id: str) -> pd.DataFrame:
        obs = payload.get("observations", [])
        df = pd.DataFrame(obs)
        if df.empty:
            return pd.DataFrame(columns=[series_id]).set_index(pd.DatetimeIndex([], name="date"))
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.set_index("date")[["value"]].rename(columns={"value": series_id}).sort_index()

    def get_one(
        self,
        cfg_or_id: Union[dict, str],
        *,
        observation_start: Optional[str] = None,
        observation_end: Optional[str] = None
    ) -> pd.DataFrame:
        if isinstance(cfg_or_id, dict):
            series_id = cfg_or_id["id"]
        else:
            series_id = str(cfg_or_id)
        raw = self.request_observations(
            series_id,
            observation_start=observation_start,
            observation_end=observation_end,
        )
        df = self._to_frame(raw, series_id)
        return df


# =============================================================
# Especificaciones (Steps / Series / Capa)
# =============================================================

@dataclass
class TransformStep:
    fn: str
    arg: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SeriesSpec:
    fred_id: str
    alias: Optional[str] = None
    units: Optional[str] = None             # normalize_units (pp, %, bp, rebase, log, scale:)
    freq_objetivo: str = "M"
    agg: Literal["last","mean"] = "last"
    fill: Optional[Literal["ffill","bfill","both"]] = "ffill"
    steps: List[TransformStep] = field(default_factory=list)

@dataclass
class LayerSpec:
    name: str
    series: List[SeriesSpec]
    target_freq: str = "M"
    resample_agg: Literal["last","mean"] = "last"
    how_merge: Literal["outer","inner","asof"] = "outer"
    default_fill: Optional[Literal["ffill","bfill","both"]] = "ffill"
    derived: List[TransformStep] = field(default_factory=list)
    post: List[TransformStep] = field(default_factory=list)


# =============================================================
# DataTransform: transformaciones sobre Series (índice datetime)
# =============================================================

class DataTransform:
    # --- Limpieza ---
    def maybe_fill(self, s: pd.Series, fill: Optional[Literal["ffill","bfill","both"]] = None) -> pd.Series:
        if fill is None:
            return s
        if fill == "ffill":
            return s.ffill()
        if fill == "bfill":
            return s.bfill()
        if fill == "both":
            return s.ffill().bfill()
        raise ValueError(f"Fill no soportado: {fill}")

    def dropna(self, s: pd.Series) -> pd.Series:
        return s.dropna()

    def clip(self, s: pd.Series, clip_lower: Optional[Number] = None, clip_upper: Optional[Number] = None) -> pd.Series:
        return s.clip(lower=clip_lower, upper=clip_upper)

    def winsorize(self, s: pd.Series, percentile_low: float = 0.01, percentile_up: float = 0.99) -> pd.Series:
        if s.empty:
            return s
        lo = s.quantile(percentile_low)
        hi = s.quantile(percentile_up)
        return s.clip(lower=lo, upper=hi)

    def normalize_units(self, s: pd.Series, units: str) -> pd.Series:
        if s is None or s.empty:
            return s
        key = (units or "").strip().lower()

        # Porcentaje / puntos / bps
        if key == "percent_from_decimal":
            return s * 100.0
        if key == "decimal_from_percent":
            return s / 100.0
        if key == "pp_from_decimal":
            return s * 100.0
        if key == "pp_from_percent":
            return s
        if key == "bp_from_percent":
            return s * 100.0
        if key == "bp_from_decimal":
            return s * 10000.0
        if key == "percent_from_bp":
            return s / 100.0
        if key == "decimal_from_bp":
            return s / 10000.0

        # Escalado genérico
        if key.startswith("scale:"):
            factor = float(key.split("scale:", 1)[1])
            return s * factor

        # Rebase
        if key.startswith("rebase:"):
            payload = key.split("rebase:", 1)[1]
            if "@" in payload:
                base_str, date_str = payload.split("@", 1)
                base_val = float(base_str)
                ref_date = pd.to_datetime(date_str)
                ref_val = s.loc[ref_date]
            else:
                base_val = float(payload)
                ref_val = s.dropna().iloc[0]
            return (s / ref_val) * base_val

        if key == "log":
            return np.log(s.where(s > 0))
        if key in {"level", "identity", "none", ""}:
            return s

        raise ValueError(f"Modo de normalize_units no soportado: '{units}'")

    def log(self, s: pd.Series) -> pd.Series:
        return np.log(s.where(s > 0))

    # --- Variaciones ---
    def diff(self, s: pd.Series, periods: int = 1) -> pd.Series:
        return s.diff(periods)

    def pct_change(self, s: pd.Series, periods: int = 1) -> pd.Series:
        return s.pct_change(periods=periods)

    def mom(self, s: pd.Series) -> pd.Series:
        return s.pct_change(1)

    def yoy(self, s: pd.Series) -> pd.Series:
        return s.pct_change(12)

    def qoq_annualized(self, s: pd.Series, compounded: bool = True) -> pd.Series:
        qoq = s.pct_change(3)
        return (1.0 + qoq) ** 4 - 1.0 if compounded else qoq * 4.0

    def rolling_pct_change(self, s: pd.Series, window: int) -> pd.Series:
        return s / s.shift(window) - 1.0

    # --- Tendencia / Suavizado ---
    def rolling_mean(self, s: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
        return s.rolling(window, min_periods=min_periods or window).mean()

    def rolling_std(self, s: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
        return s.rolling(window, min_periods=min_periods or window).std()

    def ema(self, s: pd.Series, span: int, adjust: bool = False) -> pd.Series:
        return s.ewm(span=span, adjust=adjust).mean()

    def slope(
        self,
        s: pd.Series,
        window: int,
        method: Literal["diff", "ols"] = "ols",
        min_periods: Optional[int] = None,
        as_annualized: bool = False,
        periods_per_year: Optional[int] = None,
    ) -> pd.Series:
        mp = min_periods or window
        if method == "diff":
            out = s - s.shift(window)
        elif method == "ols":
            x = np.arange(window, dtype=float)
            x = (x - x.mean())
            var_x = float((x**2).sum())
            def _beta(y: np.ndarray) -> float:
                if np.isnan(y).any():
                    return np.nan
                y = y - y.mean()
                cov_xy = float((x * y).sum())
                return cov_xy / var_x if var_x != 0 else np.nan
            out = s.rolling(window, min_periods=mp).apply(_beta, raw=True)
        else:
            raise ValueError("method debe ser 'diff' u 'ols'")
        if as_annualized and periods_per_year:
            out = out * periods_per_year
        return out

    # --- Estandarización ---
    def zscore(self, s: pd.Series, window: Optional[int] = None) -> pd.Series:
        if window is None:
            return (s - s.mean()) / s.std()
        roll = s.rolling(window)
        return (s - roll.mean()) / roll.std()

    def percentile_rank(self, s: pd.Series, window: int) -> pd.Series:
        return s.rolling(window, min_periods=window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

    def rebase_index(self, s: pd.Series, base: float = 100.0, ref: Optional[pd.Timestamp] = None) -> pd.Series:
        if s.empty:
            return s
        ref_val = s.loc[ref] if ref is not None else s.dropna().iloc[0]
        return (s / ref_val) * base

    # --- Composición intra-serie ---
    def delta_vs_min(self, s: pd.Series, window: int) -> pd.Series:
        return s - s.rolling(window).min()

    def delta_vs_max(self, s: pd.Series, window: int) -> pd.Series:
        return s - s.rolling(window).max()

    def cumulative_sum(self, s: pd.Series) -> pd.Series:
        return s.cumsum()

    def annualize_from_periods(self, s_returns: pd.Series, periods_per_year: int, compounded: bool = True) -> pd.Series:
        return (1.0 + s_returns) ** periods_per_year - 1.0 if compounded else s_returns * periods_per_year

    # --- Macro específicas ---
    def ma_weeks_to_month(self, s_weekly_ma: pd.Series) -> pd.Series:
        return s_weekly_ma.resample("M").last()

    def annualized_growth(self, s: pd.Series, window: int, periods_per_year: int, compounded: bool = True) -> pd.Series:
        ratio = s / s.shift(window)
        return (ratio ** (periods_per_year / window) - 1.0) if compounded else (ratio - 1.0) * (periods_per_year / window)

    def flag_threshold(self, s: pd.Series, threshold: Number, op: Literal["ge","gt","le","lt","eq"] = "ge") -> pd.Series:
        ops = {
            "ge": s >= threshold,
            "gt": s > threshold,
            "le": s <= threshold,
            "lt": s < threshold,
            "eq": s == threshold,
        }
        if op not in ops:
            raise ValueError("op debe ser uno de {'ge','gt','le','lt','eq'}")
        return ops[op].astype(int)

    def flag_persistence(self, cond_series: pd.Series, n_periods: int) -> pd.Series:
        x = cond_series.astype(int)
        return (x.rolling(n_periods).sum() >= n_periods).astype(int)


# =============================================================
# LayerBuilder
# =============================================================

class LayerBuilder:
    def __init__(self, loader: DataLoad, T: DataTransform):
        self.loader = loader
        self.T = T
        self.step_registry: Dict[str, Any] = {
            # Limpieza / unidades
            "maybe_fill": lambda s, **k: self.T.maybe_fill(s, k.get("fill")),
            "dropna":     lambda s, **k: self.T.dropna(s),
            "clip":       lambda s, **k: self.T.clip(s, k.get("clip_lower"), k.get("clip_upper")),
            "winsorize":  lambda s, **k: self.T.winsorize(s, k.get("percentile_low", 0.01), k.get("percentile_up", 0.99)),
            "normalize_units": lambda s, **k: self.T.normalize_units(s, k["units"]),

            # Variaciones
            "diff":        lambda s, **k: self.T.diff(s, k.get("periods", 1)),
            "pct_change":  lambda s, **k: self.T.pct_change(s, k.get("periods", 1)),
            "mom":         lambda s, **k: self.T.mom(s),
            "yoy":         lambda s, **k: self.T.yoy(s),
            "qoq_annualized": lambda s, **k: self.T.qoq_annualized(s, k.get("compounded", True)),
            "rolling_pct_change": lambda s, **k: self.T.rolling_pct_change(s, k["window"]),

            # Tendencia
            "rolling_mean": lambda s, **k: self.T.rolling_mean(s, k["window"], k.get("min_periods")),
            "rolling_std":  lambda s, **k: self.T.rolling_std(s, k["window"], k.get("min_periods")),
            "ema":          lambda s, **k: self.T.ema(s, k["span"], k.get("adjust", False)),
            "slope":        lambda s, **k: self.T.slope(
                                s, k["window"], k.get("method", "ols"), k.get("min_periods"),
                                k.get("as_annualized", False), k.get("periods_per_year"),
                            ),

            # Estandarización
            "zscore":          lambda s, **k: self.T.zscore(s, k.get("window")),
            "percentile_rank": lambda s, **k: self.T.percentile_rank(s, k["window"]),

            # Composición
            "delta_vs_min": lambda s, **k: self.T.delta_vs_min(s, k["window"]),
            "delta_vs_max": lambda s, **k: self.T.delta_vs_max(s, k["window"]),
            "cumsum":       lambda s, **k: self.T.cumulative_sum(s),

            # Macro
            "annualize_from_periods": lambda s, **k: self.T.annualize_from_periods(s, k["periods_per_year"], k.get("compounded", True)),
            "annualized_growth":      lambda s, **k: self.T.annualized_growth(s, k["window"], k["periods_per_year"], k.get("compounded", True)),
        }

    # --- helpers internos ---
    def _resample_series(self, s: pd.Series, target_freq: Optional[str], agg: Literal["last","mean"]) -> pd.Series:
        if target_freq is None:
            return s
        if agg == "last":
            return s.resample(target_freq).last()
        elif agg == "mean":
            return s.resample(target_freq).mean()
        raise ValueError(f"resample_agg no soportado: {agg}")

    def _apply_steps(self, s: pd.Series, steps: List[TransformStep]) -> pd.Series:
        out = s
        for st in steps or []:
            if st.fn not in self.step_registry:
                raise ValueError(f"Paso '{st.fn}' no está registrado")
            out = self.step_registry[st.fn](out, **(st.arg or {}))
        return out

    def _apply_post(self, df: pd.DataFrame, post_steps: List[TransformStep]) -> pd.DataFrame:
        if not post_steps:
            return df
        out = df.copy()
        for st in post_steps:
            fn = st.fn
            args = dict(st.arg or {})
            cols = args.pop("cols", "all")
            if cols == "all":
                cols_to_apply = list(out.columns)
            else:
                cols_to_apply = [c for c in cols if c in out.columns]
            if fn == "expr":
                raise ValueError("Usa 'expr' en 'derived', no en 'post'")
            for c in cols_to_apply:
                s = out[c]
                s2 = self._apply_steps(s, [TransformStep(fn=fn, arg=args)])
                out[c] = s2
        return out

    def _compute_derived(self, df: pd.DataFrame, derived_steps: List[TransformStep]) -> pd.DataFrame:
        out = df.copy()
        for st in derived_steps or []:
            fn = st.fn
            a = (st.arg or {})
            if fn == "expr":
                name = a["name"]
                expr = a["expr"]  # expresión tipo "A - B" con nombres de columnas
                res = out.eval(expr, engine="python")
                out[name] = res
            elif fn in {"spread", "diff"}:
                name = a["name"]; A = a["a"]; B = a["b"]
                out[name] = out[A] - out[B]
            elif fn in {"ratio", "div"}:
                name = a["name"]; A = a["a"]; B = a["b"]
                out[name] = out[A] / out[B]
            elif fn in {"sum"}:
                name = a["name"]; A = a["a"]; B = a["b"]
                out[name] = out[A] + out[B]
            elif fn in {"prod"}:
                name = a["name"]; A = a["a"]; B = a["b"]
                out[name] = out[A] * out[B]
            else:
                raise ValueError(f"Derived '{fn}' no soportado")
        return out

    def build(self, spec: LayerSpec) -> pd.DataFrame:
        frames = []
        # 1) Serie a serie
        for s in spec.series:
            df_raw = self.loader.get_one(s.fred_id)
            ser = df_raw[s.fred_id]

            col_name = s.alias or s.fred_id

            if s.units:
                ser = self.T.normalize_units(ser, s.units)

            target_freq = s.freq_objetivo or spec.target_freq
            ser = self._resample_series(ser, target_freq, s.agg)

            fill_mode = s.fill if s.fill is not None else spec.default_fill
            ser = self.T.maybe_fill(ser, fill_mode)

            if s.steps:
                ser = self._apply_steps(ser, s.steps)

            frames.append(ser.to_frame(col_name))

        # 2) Merge
        if not frames:
            return pd.DataFrame()
        if spec.how_merge == "outer":
            df = pd.concat(frames, axis=1, join="outer")
        elif spec.how_merge == "inner":
            df = pd.concat(frames, axis=1, join="inner")
        elif spec.how_merge == "asof":
            df = pd.concat(frames, axis=1, join="outer").sort_index().ffill()
        else:
            raise ValueError(f"how_merge no soportado: {spec.how_merge}")

        # 3) Derivados
        if spec.derived:
            df = self._compute_derived(df, spec.derived)

        # 4) Post
        if spec.post:
            df = self._apply_post(df, spec.post)

        # 5) Resample final
        if spec.target_freq:
            df = df.resample(spec.target_freq).last()

        return df

    def build_all(self, specs: List[LayerSpec]) -> Dict[str, pd.DataFrame]:
        out = {}
        for sp in specs:
            out[sp.name] = self.build(sp)
        return out
