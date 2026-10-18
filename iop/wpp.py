"""UN World Population Prospects 2024（中国，LocID=156）读取。原始单位为千人，这里统一换算为“人”。

注意：WPP 的历史值（1950-2023）本身是联合国综合普查、调查与模型得到的估计，不是原始登记数。
本项目把它当作“长时间序列 + 外部对照基准”，主数据仍以普查为准。
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import EXTERNAL_DIR

FILES = {
    "indicators": "WPP2024_Demographic_Indicators_Medium_CHN.csv",
    "deaths": "WPP2024_DeathsBySingleAgeSex_Medium_1950-2023_CHN.csv",
    "population": "WPP2024_Population1JanuaryBySingleAgeSex_Medium_1950-2023_CHN.csv",
    "asfr5": "WPP2024_Fertility_by_Age5_CHN.csv",
}
THOUSAND = 1000.0
MAX_AGE = 100


@lru_cache(maxsize=None)
def _read(key: str) -> pd.DataFrame:
    return pd.read_csv(EXTERNAL_DIR / FILES[key], encoding="utf-8-sig", low_memory=False)


def indicators() -> pd.DataFrame:
    """年度汇总指标（中方案 1950-2100），index=year。人数列已换算为人。"""
    df = _read("indicators").set_index("Time")
    df.index.name = "year"
    count_cols = ["TPopulation1Jan", "TPopulation1July", "Births", "Deaths", "NetMigrations"]
    df[count_cols] = df[count_cols] * THOUSAND
    return df


def _by_age_sex(key: str, male: str, female: str) -> pd.DataFrame:
    df = _read(key)
    df = df[df["Variant"] == "Medium"]
    age = df["AgeGrpStart"].clip(upper=MAX_AGE)
    out = (
        pd.DataFrame({"year": df["Time"], "age": age, "male": df[male] * THOUSAND, "female": df[female] * THOUSAND})
        .groupby(["year", "age"])
        .sum()
    )
    return out


@lru_cache(maxsize=None)
def deaths() -> pd.DataFrame:
    """日历年死亡数，MultiIndex (year, age)，列 male/female。"""
    return _by_age_sex("deaths", "DeathMale", "DeathFemale")


@lru_cache(maxsize=None)
def population_jan1() -> pd.DataFrame:
    """1 月 1 日人口，MultiIndex (year, age)，列 male/female。"""
    return _by_age_sex("population", "PopMale", "PopFemale")


@lru_cache(maxsize=None)
def exposure() -> pd.DataFrame:
    """年中暴露人口 ≈ (1 月 1 日人口 + 次年 1 月 1 日人口) / 2；最后一年用 7 月 1 日总人口比例近似。"""
    pop = population_jan1()
    years = pop.index.get_level_values("year").unique()
    frames = []
    for y in years:
        cur = pop.loc[y]
        nxt = pop.loc[y + 1] if y + 1 in years else None
        if nxt is None:
            ratio = indicators().loc[y, "TPopulation1July"] / indicators().loc[y, "TPopulation1Jan"]
            mid = cur * ratio
        else:
            mid = (cur + nxt) / 2.0
        frames.append(mid.assign(year=y).set_index("year", append=True).swaplevel())
    return pd.concat(frames)


@lru_cache(maxsize=None)
def mortality_rates() -> dict[str, pd.DataFrame]:
    """单岁中心死亡率 m(x,t) = D / E。返回 {'male','female'}，每个为 age × year 的矩阵。"""
    d, e = deaths(), exposure()
    out = {}
    for sex in ("male", "female"):
        m = (d[sex] / e[sex]).unstack("year")
        out[sex] = m.sort_index()
    return out


def asfr5(variant: str = "Medium") -> pd.DataFrame:
    """5 岁组年龄别生育率（每名妇女），index=year，列为组起始年龄 15,20,...,45（10-14、50-54 并入两端）。"""
    df = _read("asfr5")
    df = df[df["Variant"] == variant]
    wide = df.pivot_table(index="Time", columns="AgeGrpStart", values="ASFR", aggfunc="first") / THOUSAND
    wide.index.name = "year"
    return wide


def tfr_from_asfr5(variant: str = "Medium") -> pd.Series:
    return asfr5(variant).sum(axis=1) * 5.0


def mean_age_childbearing_from_asfr5(variant: str = "Medium") -> pd.Series:
    a = asfr5(variant)
    mids = np.asarray(a.columns, dtype=float) + 2.5
    return (a * mids).sum(axis=1) / a.sum(axis=1)
