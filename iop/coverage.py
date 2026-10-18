"""普查漏报校正：死亡与出生的水平用官方汇总指标校准，年龄模式保留普查。

为什么需要：
- 普查记录的是“普查前 12 个月家中发生的死亡/出生”，独居老人去世、婴儿出生后很快死亡等情况容易漏报。
- 2020 年普查死亡 797 万，而统计局年度死亡约 998 万；婴儿死亡率普查约 1.7‰，卫健委监测为 5.4‰。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from . import census, lifetable, official

SEXES = ("male", "female")


def _q(m: np.ndarray | float, a: float = 0.5) -> np.ndarray | float:
    return m / (1.0 + (1.0 - a) * m)


def adjusted_mortality(year: int, target: str = "e0", close_from: int = 90, max_age: int = 110) -> dict[str, object]:
    """返回 {'mx': {sex: Series(0..max_age)}, 'factors': {...}, 'implied_deaths': float, 'raw': DataFrame}。

    步骤：
    1. 0 岁：两性共用一个系数 f0，使按出生性别加权的 q0 = 官方婴儿死亡率；
    2. 1-4 岁：系数 f1，使 1 − (1−q0)·(1−4q1) = 官方 5 岁以下死亡率；
    3. 5 岁以上：系数 fa，两种口径——
       target="deaths"：两性共用 fa，使全年龄死亡总数 = 官方粗死亡率 × 普查人口；
       target="e0"    ：分性别 fa，使校正后 e0 = 官方平均预期寿命；
    4. 90 岁以上：用 Kannisto 曲线（在 80-94 岁上拟合）替换并延伸到 110 岁。
    """
    raw = census.mortality_rates(year)
    pop = census.population(year)
    births = census.births_by_sex(year)
    w = {s: births[s] / (births["male"] + births["female"]) for s in SEXES}

    def q0_both(f: float) -> float:
        return sum(w[s] * _q(f * raw.loc[0, s], lifetable.infant_ax(f * raw.loc[0, s], s)) for s in SEXES)

    f0 = brentq(lambda f: q0_both(f) - official.INFANT_MORTALITY[year], 0.1, 50)
    q0 = official.INFANT_MORTALITY[year]

    def q1_4(f: float) -> float:
        surv = sum(w[s] * np.prod(1.0 - _q(f * raw.loc[1:4, s].to_numpy())) for s in SEXES)
        return 1.0 - surv

    target_4q1 = 1.0 - (1.0 - official.UNDER5_MORTALITY[year]) / (1.0 - q0)
    f1 = brentq(lambda f: q1_4(f) - target_4q1, 0.1, 50)

    def build(s: str, fa: float) -> pd.Series:
        m = raw[s].copy()
        m.loc[0] *= f0
        m.loc[1:4] *= f1
        m.loc[5:] *= fa
        return lifetable.kannisto_close(m, start_age=close_from, max_age=max_age)

    if target == "deaths":
        target_deaths = official.CRUDE_DEATH_RATE[year] * pop.to_numpy().sum()
        child = sum((raw.loc[0, s] * f0 * pop.loc[0, s]) + (raw.loc[1:4, s] * f1 * pop.loc[1:4, s]).sum() for s in SEXES)
        adult_obs = sum((raw.loc[5:, s] * pop.loc[5:, s]).sum() for s in SEXES)
        fa = {s: (target_deaths - child) / adult_obs for s in SEXES}
    elif target == "e0":
        fa = {s: brentq(lambda f: lifetable.e0(build(s, f), s) - official.LIFE_EXPECTANCY[year][s], 0.3, 5.0) for s in SEXES}
    else:
        raise ValueError(target)

    mx = {s: build(s, fa[s]) for s in SEXES}
    implied = sum((mx[s].loc[: census.MAX_AGE - 1] * pop.loc[: census.MAX_AGE - 1, s]).sum() for s in SEXES)
    implied += sum(mx[s].loc[census.MAX_AGE] * pop.loc[census.MAX_AGE, s] for s in SEXES)
    factors = {"infant": f0, "child_1_4": f1, **{f"adult_{s}": fa[s] for s in SEXES}}
    return {"mx": mx, "factors": factors, "implied_deaths": float(implied), "raw": raw}


def implied_births(year: int) -> float:
    """普查年龄别生育率 × 普查全量育龄妇女数。"""
    f = census.fertility(year)
    women = census.population(year).loc[15:49, "female"]
    return float((f["asfr"] * women).sum())


def fertility_coverage(year: int) -> dict[str, float]:
    """出生水平校正系数 = 官方出生数（粗出生率 × 人口）/ 普查口径出生数。"""
    official_births = official.CRUDE_BIRTH_RATE[year] * census.population(year).to_numpy().sum()
    implied = implied_births(year)
    factor = official_births / implied
    return {
        "census_births": census.births_total(year),
        "implied_births": implied,
        "official_births": official_births,
        "factor": factor,
        "tfr_census": census.tfr(year),
        "tfr_adjusted": census.tfr(year) * factor,
    }


def cohort_consistency() -> pd.DataFrame:
    """同一出生队列在相邻两次普查中的规模对比（5 岁组），检查低龄人口漏报。

    例：2000 年普查 0-4 岁 → 2010 年普查 10-14 岁。十年间死亡很少、国际迁移可忽略，
    若后一次普查人数明显更多，说明前一次漏登了儿童（常见于计划生育时期的“瞒报”）。
    """
    from .ddm import population_5y

    rows = []
    for y1, y2 in ((2000, 2010), (2010, 2020)):
        p1 = population_5y(y1, "male") + population_5y(y1, "female")
        p2 = population_5y(y2, "male") + population_5y(y2, "female")
        for a in range(0, 30, 5):
            rows.append(
                {
                    "cohort_born": f"{y1 - a - 5 + 1}-{y1 - a}",
                    "census1": y1,
                    "age1": f"{a}-{a + 4}",
                    "pop1": p1.loc[a],
                    "census2": y2,
                    "age2": f"{a + 10}-{a + 14}",
                    "pop2": p2.loc[a + 10],
                    "ratio": p2.loc[a + 10] / p1.loc[a],
                }
            )
    return pd.DataFrame(rows)
