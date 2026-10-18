"""Lee–Carter 随机死亡率模型（Lee & Carter 1992）。

    log m(x,t) = a(x) + b(x)·k(t) + ε(x,t)

- a(x)：各年龄死亡率的平均水平（对数）
- k(t)：整体死亡水平随时间的指数，下降 = 死亡率改善
- b(x)：各年龄对 k(t) 的敏感度（哪些年龄改善得快）
k(t) 用带漂移的随机游走预测：k(t+1) = k(t) + d + e(t)，e ~ N(0, σ²)。
蒙特卡洛模拟同时考虑 e(t) 的随机性和漂移 d 的估计误差，得到预测区间。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from . import lifetable


@dataclass
class LeeCarter:
    ax: pd.Series
    bx: pd.Series
    kt: pd.Series
    explained: float
    drift: float
    sigma: float
    drift_se: float
    jump_off: pd.Series  # 预测起点的 log m(x)

    @property
    def ages(self) -> np.ndarray:
        return self.ax.index.to_numpy()

    @property
    def last_year(self) -> int:
        return int(self.kt.index[-1])


def fit(log_m: pd.DataFrame, deaths: pd.DataFrame | None = None, exposure: pd.DataFrame | None = None, jump_off: str = "observed") -> LeeCarter:
    """log_m：age × year 的对数死亡率矩阵。

    deaths/exposure 给出时做 Lee–Carter 第二步：逐年重估 k(t)，使模型死亡总数 = 实际死亡总数。
    jump_off="observed" 采用最后一年的实际 log m 作为预测起点（Lee & Miller 2001 的修正，减小起点偏差）；
    "fitted" 用模型拟合值。
    """
    ax = log_m.mean(axis=1)
    centered = log_m.sub(ax, axis=0).to_numpy()
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    bx = u[:, 0] / u[:, 0].sum()
    kt = s[0] * vt[0] * u[:, 0].sum()
    explained = float(s[0] ** 2 / (s**2).sum())
    bx = pd.Series(bx, index=log_m.index)
    kt = pd.Series(kt, index=log_m.columns)

    if deaths is not None and exposure is not None:
        new_k = {}
        for t in log_m.columns:
            d_obs = deaths[t].loc[log_m.index].sum()
            e = exposure[t].loc[log_m.index]
            new_k[t] = brentq(lambda k: (e * np.exp(ax + bx * k)).sum() - d_obs, kt[t] - 50, kt[t] + 50)
        kt = pd.Series(new_k)
        shift = kt.mean()
        kt -= shift
        ax = ax + bx * shift

    diffs = kt.diff().dropna()
    n = len(kt) - 1
    drift = float((kt.iloc[-1] - kt.iloc[0]) / n)
    sigma = float(np.sqrt(((diffs - drift) ** 2).sum() / (n - 1)))
    start = log_m.iloc[:, -1] if jump_off == "observed" else ax + bx * kt.iloc[-1]
    return LeeCarter(ax=ax, bx=bx, kt=kt, explained=explained, drift=drift, sigma=sigma, drift_se=sigma / np.sqrt(n), jump_off=start)


def forecast_kt(model: LeeCarter, horizon: int, n_sims: int = 0, seed: int = 0) -> np.ndarray:
    """返回 k 相对起点的增量路径。n_sims=0 返回中位路径 (horizon,)，否则 (n_sims, horizon)。"""
    h = np.arange(1, horizon + 1)
    if n_sims == 0:
        return model.drift * h
    rng = np.random.default_rng(seed)
    drifts = rng.normal(model.drift, model.drift_se, size=(n_sims, 1))
    shocks = rng.normal(0.0, model.sigma, size=(n_sims, horizon)).cumsum(axis=1)
    return drifts * h + shocks


def forecast_log_m(model: LeeCarter, horizon: int, n_sims: int = 0, seed: int = 0) -> np.ndarray:
    """log m 预测。n_sims=0 → (horizon, ages)；否则 (n_sims, horizon, ages)。"""
    dk = forecast_kt(model, horizon, n_sims, seed)
    base = model.jump_off.to_numpy()
    b = model.bx.to_numpy()
    if n_sims == 0:
        return base[None, :] + dk[:, None] * b[None, :]
    return base[None, None, :] + dk[:, :, None] * b[None, None, :]


def close_and_e0(log_m_row: np.ndarray, ages: np.ndarray, sex: str, close_from: int = 90, fit_ages: range = range(80, 90)) -> float:
    m = pd.Series(np.exp(log_m_row), index=ages)
    m = lifetable.kannisto_close(m, start_age=close_from, fit_ages=fit_ages)
    return lifetable.e0(m, sex)


def e0_paths(model: LeeCarter, sex: str, horizon: int, n_sims: int = 0, seed: int = 0, **kwargs) -> np.ndarray:
    lm = forecast_log_m(model, horizon, n_sims, seed)
    if n_sims == 0:
        return np.array([close_and_e0(lm[h], model.ages, sex, **kwargs) for h in range(horizon)])
    return np.array([[close_and_e0(lm[i, h], model.ages, sex, **kwargs) for h in range(horizon)] for i in range(n_sims)])
