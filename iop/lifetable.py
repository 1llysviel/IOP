"""单岁生命表与 Kannisto 高龄平滑。

生命表各列（标准人口学记号）：
    mx  中心死亡率            qx  x 岁到 x+1 岁的死亡概率
    ax  死者在区间内平均存活年数  lx  活到 x 岁的人数（基数 1）
    dx  区间内死亡人数          Lx  区间内存活人年数
    Tx  x 岁以后的总人年数      ex  x 岁时的预期余寿
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def infant_ax(m0: float, sex: str) -> float:
    """0 岁 a0：Andreev & Kingkade (2015)，HMD 方法协议 v6 采用。"""
    if sex == "male":
        if m0 < 0.02300:
            return 0.14929 - 1.99545 * m0
        if m0 < 0.08307:
            return 0.02832 + 3.26201 * m0
        return 0.29915
    if m0 < 0.01724:
        return 0.14903 - 2.05527 * m0
    if m0 < 0.06891:
        return 0.04667 + 3.88089 * m0
    return 0.31411


def life_table(mx: pd.Series | np.ndarray, sex: str) -> pd.DataFrame:
    """由单岁 mx 构造生命表；最后一个年龄为开放组。"""
    mx = pd.Series(mx, dtype=float)
    ages = mx.index.to_numpy()
    m = mx.to_numpy()
    n = len(m)
    ax = np.full(n, 0.5)
    ax[0] = infant_ax(m[0], sex)
    qx = m / (1.0 + (1.0 - ax) * m)
    qx = np.clip(qx, 0.0, 1.0)
    qx[-1] = 1.0
    lx = np.ones(n)
    for i in range(1, n):
        lx[i] = lx[i - 1] * (1.0 - qx[i - 1])
    dx = lx * qx
    Lx = lx - (1.0 - ax) * dx
    Lx[-1] = lx[-1] / m[-1]
    ax[-1] = 1.0 / m[-1]
    Tx = Lx[::-1].cumsum()[::-1]
    ex = Tx / lx
    return pd.DataFrame({"mx": m, "qx": qx, "ax": ax, "lx": lx, "dx": dx, "Lx": Lx, "Tx": Tx, "ex": ex}, index=pd.Index(ages, name="age"))


def e0(mx: pd.Series | np.ndarray, sex: str) -> float:
    return float(life_table(mx, sex)["ex"].iloc[0])


def fit_kannisto(mx: pd.Series, fit_ages: range = range(80, 95), deaths: pd.Series | None = None, exposure: pd.Series | None = None) -> tuple[float, float]:
    """拟合 Kannisto 模型 μ(x) = a·e^{b(x-80)} / (1 + a·e^{b(x-80)})。

    有死亡数和暴露时用泊松极大似然；否则在 logit(m) 上做最小二乘。返回 (a, b)。
    """
    x = np.asarray(list(fit_ages), dtype=float) - 80.0
    m = mx.loc[list(fit_ages)].to_numpy(dtype=float)
    y = np.log(m / (1.0 - np.clip(m, None, 0.999)))
    b0, loga0 = np.polyfit(x, y, 1)
    if deaths is None or exposure is None:
        return float(np.exp(loga0)), float(b0)

    d = deaths.loc[list(fit_ages)].to_numpy(dtype=float)
    e = exposure.loc[list(fit_ages)].to_numpy(dtype=float)

    def nll(theta: np.ndarray) -> float:
        a, b = np.exp(theta[0]), theta[1]
        mu = a * np.exp(b * x) / (1.0 + a * np.exp(b * x))
        return float(np.sum(e * mu - d * np.log(mu)))

    res = minimize(nll, x0=[loga0, b0], method="Nelder-Mead")
    return float(np.exp(res.x[0])), float(res.x[1])


def kannisto(ages: np.ndarray, a: float, b: float) -> np.ndarray:
    z = a * np.exp(b * (np.asarray(ages, dtype=float) - 80.0))
    return z / (1.0 + z)


def kannisto_close(mx: pd.Series, start_age: int = 90, max_age: int = 110, fit_ages: range = range(80, 95), **kwargs) -> pd.Series:
    """用 Kannisto 曲线替换 start_age 以上的 mx，并延伸到 max_age（开放组）。

    高龄段普查数据常见年龄堆积、年龄夸报与死亡漏报，直接用会让 mx 在 95 岁后反常下降。
    """
    a, b = fit_kannisto(mx, fit_ages, **kwargs)
    young = mx.loc[: start_age - 1]
    old_ages = np.arange(start_age, max_age + 1)
    out = pd.concat([young, pd.Series(kannisto(old_ages, a, b), index=old_ages)])
    out.index.name = "age"
    return out


def survival_ratios(lt: pd.DataFrame, open_age: int) -> pd.Series:
    """队列要素预测用的一年期存活比 S(x) = L(x+1)/L(x)，开放组 open_age+ 合并处理。

    返回 index 为 x（从 x 岁到 x+1 岁）的 Series；index=-1 表示出生到 0 岁（L0 / l0）。
    """
    L = lt["Lx"]
    L_open = L.loc[open_age:].sum()
    T_prev = L.loc[open_age - 1:].sum()
    s = {-1: L.iloc[0] / lt["lx"].iloc[0]}
    for x in range(0, open_age - 1):
        s[x] = L.loc[x + 1] / L.loc[x]
    s[open_age - 1] = L_open / T_prev  # (open-1 岁与开放组) 共同进入开放组
    return pd.Series(s).sort_index()
