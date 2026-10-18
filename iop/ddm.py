"""死亡分布法（Death Distribution Methods）：广义增长平衡法 GGB（Hill 1987）。

思想：对 x 岁以上的人群，有恒等式
    进入率 b(x+) − 增长率 r(x+) = 死亡率 d_true(x+)
若普查死亡登记完整度为 c（登记数 = c × 真实数），两次普查覆盖率之比为 e^{kt}，则
    b(x+) − r(x+) = k + (1/c) · d_obs(x+)
对各年龄 x 做直线回归：斜率的倒数就是死亡登记完整度 c。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import census

GROUP_STARTS = np.arange(0, 100, 5)  # 0,5,...,95；100+ 为开放组


def _five_year(single: pd.Series) -> pd.Series:
    s = single.copy()
    groups = np.where(s.index >= 100, 100, (s.index // 5) * 5)
    return s.groupby(groups).sum()


def population_5y(year: int, sex: str) -> pd.Series:
    if year == 2020:
        g = census.population_groups(2020)
        return pd.Series(g[sex].to_numpy(), index=g["start"].to_numpy())
    return _five_year(census.population(year)[sex])


def deaths_5y(year: int, sex: str) -> pd.Series:
    d = census.deaths_groups(year)
    s = pd.Series(d[sex].to_numpy(), index=d["start"].to_numpy())
    return _five_year(s)


def ggb(year1: int, year2: int, sex: str, fit_ages: range = range(15, 65, 5)) -> dict[str, object]:
    t = float(year2 - year1)
    n1, n2 = population_5y(year1, sex), population_5y(year2, sex)
    deaths_annual = (deaths_5y(year1, sex) + deaths_5y(year2, sex)) / 2.0

    ages = n1.index.to_numpy()
    cum1 = n1[::-1].cumsum()[::-1]
    cum2 = n2[::-1].cumsum()[::-1]
    dcum = deaths_annual[::-1].cumsum()[::-1]

    rows = []
    for x in ages[1:]:
        if x >= 100:
            continue
        pyl = t * np.sqrt(cum1.loc[x] * cum2.loc[x])
        entrants = t * np.sqrt(n1.loc[x - 5] * n2.loc[x]) / 5.0
        rows.append(
            {
                "age": x,
                "b": entrants / pyl,
                "r": np.log(cum2.loc[x] / cum1.loc[x]) / t,
                "d": dcum.loc[x] * t / pyl,
            }
        )
    table = pd.DataFrame(rows).set_index("age")
    table["b_minus_r"] = table["b"] - table["r"]
    sel = table.loc[list(fit_ages)]
    slope, intercept = np.polyfit(sel["d"], sel["b_minus_r"], 1)
    return {
        "table": table,
        "slope": float(slope),
        "intercept": float(intercept),
        "completeness": float(1.0 / slope),
        "relative_coverage": float(np.exp(intercept * t)),
        "fit_ages": fit_ages,
    }
