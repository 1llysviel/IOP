"""生育模型：年龄曲线（Hadwiger）、进度效应（Bongaarts–Feeney）、TFR 贝叶斯 AR(1) 预测。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq, curve_fit

AGES = np.arange(15, 50)


# ---------- 年龄曲线 ----------

def hadwiger(x: np.ndarray, tfr: float, h: float, t: float) -> np.ndarray:
    """Hadwiger (1940) 生育函数。tfr 控制水平（曲线下面积），t 控制位置（≈峰值年龄附近），h 控制集中度。"""
    x = np.asarray(x, dtype=float)
    return tfr * h / (t * np.sqrt(np.pi)) * (t / x) ** 1.5 * np.exp(-(h**2) * (t / x + x / t - 2.0))


def fit_hadwiger(ages: np.ndarray, asfr: np.ndarray) -> dict[str, float]:
    """最小二乘拟合；ages 用年龄组中点（单岁取 x+0.5）。"""
    p0 = [asfr.sum() * np.mean(np.diff(ages)) if len(ages) > 1 else 1.5, 3.0, 27.0]
    params, _ = curve_fit(hadwiger, ages, asfr, p0=p0, bounds=([0.1, 0.5, 18], [8, 10, 40]), maxfev=20000)
    fitted = hadwiger(ages, *params)
    ss_res = float(((asfr - fitted) ** 2).sum())
    ss_tot = float(((asfr - asfr.mean()) ** 2).sum())
    return {"tfr": params[0], "h": params[1], "t": params[2], "r2": 1 - ss_res / ss_tot}


def mean_age(ages: np.ndarray, asfr: np.ndarray) -> float:
    """平均生育年龄 MAC = Σ(x·f) / Σf，ages 为区间中点。"""
    return float((ages * asfr).sum() / asfr.sum())


def schedule(tfr: float, mac: float, h: float, ages: np.ndarray = AGES) -> pd.Series:
    """给定 TFR 与目标 MAC，保持形状参数 h，反解 Hadwiger 的 t，生成单岁 ASFR（和 = tfr）。"""
    mids = ages + 0.5

    def gap(t: float) -> float:
        return mean_age(mids, hadwiger(mids, 1.0, h, t)) - mac

    t = brentq(gap, 18, 40)
    f = hadwiger(mids, 1.0, h, t)
    return pd.Series(tfr * f / f.sum(), index=ages)


# ---------- 进度效应 ----------

def tempo_adjusted_tfr(tfr: pd.Series, mac: pd.Series) -> pd.Series:
    """Bongaarts & Feeney (1998)：TFR* = TFR / (1 − r)，r 为平均生育年龄的年变化（中心差分）。

    推迟生育时（r > 0），当期 TFR 被“压低”，TFR* 估计没有推迟时的生育水平（数量效应）。
    注意：严格做法应分孩次计算再相加，这里因缺少逐年分孩次数据只做合计版本。
    """
    r = (mac.shift(-1) - mac.shift(1)) / 2.0
    return tfr / (1.0 - r)


# ---------- MAC 趋势 ----------

def logistic(t: np.ndarray, lower: float, upper: float, mid: float, scale: float) -> np.ndarray:
    return lower + (upper - lower) / (1.0 + np.exp(-(t - mid) / scale))


def fit_mac_trend(mac: pd.Series, upper: float) -> dict[str, float]:
    """在给定上限 upper（情景参数）下拟合 logistic 的下限、中点与尺度。

    只用中国自身数据时上限无法识别（拟合会顶到边界），所以上限由情景给定，
    参照东亚低生育社会：日本 MAC ≈ 32，韩国 ≈ 33。
    """
    t = mac.index.to_numpy(dtype=float)

    def f(t, lower, mid, scale):
        return logistic(t, lower, upper, mid, scale)

    params, _ = curve_fit(f, t, mac.to_numpy(), p0=[mac.min() - 0.5, t.mean() + 10, 10.0],
                          bounds=([15, 1980, 2], [mac.min(), 2100, 60]), maxfev=20000)
    return {"lower": float(params[0]), "upper": upper, "mid": float(params[1]), "scale": float(params[2])}


def project_mac(trend: dict[str, float], years: np.ndarray) -> pd.Series:
    return pd.Series(logistic(np.asarray(years, dtype=float), trend["lower"], trend["upper"], trend["mid"], trend["scale"]), index=years)


# ---------- TFR 贝叶斯 AR(1) ----------

@dataclass
class TfrPrior:
    """log TFR 的长期均值 μ 的先验：TFR 中位数 median，对数尺度标准差 log_sd；ρ 在 [rho_lo, rho_hi] 均匀。"""

    median: float = 1.3
    log_sd: float = 0.25
    rho_lo: float = 0.5
    rho_hi: float = 1.0


@dataclass
class TfrPosterior:
    mu_grid: np.ndarray
    rho_grid: np.ndarray
    weights: np.ndarray  # (len(mu), len(rho))，归一化
    ssr: np.ndarray
    n: int
    last: float  # 最后观测值的 log TFR
    last_year: int
    prior: TfrPrior

    def marginal_mu(self) -> pd.Series:
        return pd.Series(self.weights.sum(axis=1), index=np.exp(self.mu_grid))

    def marginal_rho(self) -> pd.Series:
        return pd.Series(self.weights.sum(axis=0), index=self.rho_grid)

    def sample(self, n: int, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        flat = rng.choice(self.weights.size, size=n, p=self.weights.ravel())
        i, j = np.unravel_index(flat, self.weights.shape)
        sigma2 = (self.ssr[i, j] / 2.0) / rng.gamma(self.n / 2.0, size=n)  # 逆伽马
        return pd.DataFrame({"mu": self.mu_grid[i], "rho": self.rho_grid[j], "sigma": np.sqrt(sigma2)})


def fit_tfr_ar1(tfr: pd.Series, prior: TfrPrior = TfrPrior(), n_mu: int = 241, n_rho: int = 201) -> TfrPosterior:
    """模型：y_t − μ = ρ (y_{t−1} − μ) + ε_t，y = log TFR，ε ~ N(0, σ²)。

    σ² 取 Jeffreys 先验 p(σ²) ∝ 1/σ² 并解析积分掉，(μ, ρ) 的后验在网格上精确计算：
        p(μ, ρ | y) ∝ p(μ) p(ρ) · SSR(μ, ρ)^(−n/2)
    这与联合国 BayesTFR “第三阶段”模型结构相同（他们用跨国分层先验，这里用单国 + 显式先验）。
    """
    y = np.log(tfr.to_numpy(dtype=float))
    y_prev, y_cur = y[:-1], y[1:]
    n = len(y_cur)
    mu0 = np.log(prior.median)
    mu_grid = np.linspace(mu0 - 4 * prior.log_sd, mu0 + 4 * prior.log_sd, n_mu)
    rho_grid = np.linspace(prior.rho_lo, prior.rho_hi, n_rho)
    M, R = np.meshgrid(mu_grid, rho_grid, indexing="ij")
    resid = (y_cur[None, None, :] - M[..., None]) - R[..., None] * (y_prev[None, None, :] - M[..., None])
    ssr = (resid**2).sum(axis=-1)
    log_post = -0.5 * ((M - mu0) / prior.log_sd) ** 2 - (n / 2.0) * np.log(ssr)
    w = np.exp(log_post - log_post.max())
    w /= w.sum()
    return TfrPosterior(mu_grid, rho_grid, w, ssr, n, float(y[-1]), int(tfr.index[-1]), prior)


def simulate_tfr(post: TfrPosterior, horizon: int, n_sims: int = 2000, seed: int = 0, floor: float = 0.5) -> pd.DataFrame:
    """从后验抽参数并向前模拟，返回 (n_sims, horizon) 的 TFR 路径，列为年份。"""
    rng = np.random.default_rng(seed + 1)
    params = post.sample(n_sims, seed)
    y = np.full(n_sims, post.last)
    out = np.empty((n_sims, horizon))
    mu, rho, sigma = params["mu"].to_numpy(), params["rho"].to_numpy(), params["sigma"].to_numpy()
    for h in range(horizon):
        y = mu + rho * (y - mu) + rng.normal(0.0, sigma)
        y = np.maximum(y, np.log(floor))
        out[:, h] = y
    years = np.arange(post.last_year + 1, post.last_year + 1 + horizon)
    return pd.DataFrame(np.exp(out), columns=years)


def fan(paths: pd.DataFrame, levels: tuple[float, ...] = (2.5, 10, 50, 90, 97.5)) -> pd.DataFrame:
    return pd.DataFrame({f"p{lv:g}": np.percentile(paths.to_numpy(), lv, axis=0) for lv in levels}, index=paths.columns)
