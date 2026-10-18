"""三次全国人口普查读取，统一为“单岁 × 性别 × 人”。

口径说明（重要）：
- 人口：2000、2010 年为单岁全量；2020 年只有 5 岁组，用 graduation.split_groups 拆单岁。
- 死亡：2000 年为单岁（附“平均人口”）；2010、2020 年为全国行的 0 / 1-4 / 5 岁组。
- 出生：2010、2020 年为短表全量；2000 年仅有长表（约 10% 抽样），需按抽样比放大。
- 生育（*-B0603、五-l0606）：均为长表抽样。生育率可直接用，妇女数与出生数需放大。
所有输出单位为“人”；开放年龄组统一为 100 岁及以上，记在 age=100。
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd

from . import DATASET_DIR
from .graduation import interpolate_log_rates, split_groups

YEARS = (2000, 2010, 2020)
MAX_AGE = 100
FERTILE_AGES = range(15, 50)

POP_FILES = {2000: "五-t0201.xls", 2010: "六-A0201.xls", 2020: "七-A0202.xls"}
DEATH_FILES = {2000: "五-t0604.xls", 2010: "六-A0601.xls", 2020: "七-A0601.xls"}
BIRTH_FILES = {2000: "五-l0601.xls", 2010: "六-A0112.xls", 2020: "七-A0110.xls"}
FERTILITY_FILES = {2000: ("五-l0606.xls", "长表6-6"), 2010: ("六-B0603.xls", "Sheet1"), 2020: ("七-B0603.xls", "Sheet1")}


def _clean(value: object) -> str:
    return re.sub(r"\s+", "", str(value)) if pd.notna(value) else ""


def parse_age_label(label: str) -> tuple[int, int | None] | None:
    """'0岁'/'0' → (0, 1)；'1-4岁' → (1, 4)；'100岁及以上' → (100, None)；其他 → None。"""
    label = _clean(label)
    if m := re.fullmatch(r"(\d+)岁及以上", label):
        return int(m.group(1)), None
    if m := re.fullmatch(r"(\d+)-(\d+)岁", label):
        start, end = int(m.group(1)), int(m.group(2))
        return start, end - start + 1
    if m := re.fullmatch(r"(\d+)岁?", label):
        return int(m.group(1)), 1
    return None


def _num(value: object) -> float:
    return float(pd.to_numeric(value, errors="coerce"))


@lru_cache(maxsize=None)
def _read(filename: str, sheet: str | int = 0) -> pd.DataFrame:
    return pd.read_excel(DATASET_DIR / filename, sheet_name=sheet, header=None)


def _age_rows(df: pd.DataFrame, cols: dict[str, int], single_only: bool) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        parsed = parse_age_label(row.iloc[0])
        if parsed is None:
            continue
        start, width = parsed
        if single_only and width not in (1, None):
            continue
        rows.append({"start": start, "width": width, **{k: _num(row.iloc[i]) for k, i in cols.items()}})
    return pd.DataFrame(rows)


def population_groups(year: int) -> pd.DataFrame:
    """普查原始年龄行（单岁或 5 岁组，取决于年份）。列：start, width, male, female。"""
    df = _read(POP_FILES[year])
    single_only = year != 2020
    out = _age_rows(df, {"male": 2, "female": 3}, single_only=single_only)
    return out.drop_duplicates("start").sort_values("start").reset_index(drop=True)


@lru_cache(maxsize=None)
def population(year: int) -> pd.DataFrame:
    """单岁人口，index=age(0..100)，列 male/female。2020 年由 5 岁组拆分。"""
    groups = population_groups(year)
    if year != 2020:
        out = groups.set_index("start")[["male", "female"]]
    else:
        # 0-4 岁组内出生数下降很快，样条会高估 0 岁；用“普查前 12 个月出生 − 婴儿死亡”锚定 0 岁，
        # 其余 1-99 岁再用累计样条拆分。
        births = births_by_sex(year)
        infant_deaths = deaths_groups(year).set_index("start").loc[0]
        cols = {}
        for sex in ("male", "female"):
            age0 = births[sex] - infant_deaths[sex]
            rest = groups[sex].to_numpy().copy()
            rest[0] -= age0
            starts = groups["start"].to_numpy().copy()
            starts[0] = 1
            single = split_groups(starts, rest, open_start=MAX_AGE)
            single.loc[0] = age0
            cols[sex] = single.sort_index()
        out = pd.DataFrame(cols)
    out.index.name = "age"
    return out.loc[0:MAX_AGE]


def deaths_groups(year: int) -> pd.DataFrame:
    """死亡人口（普查前 12 个月）。列：start, width, male, female；2000 年另含 exposure_male/female。"""
    df = _read(DEATH_FILES[year])
    if year == 2000:
        out = _age_rows(df, {"exposure_male": 2, "exposure_female": 3, "male": 5, "female": 6}, single_only=True)
        return out.sort_values("start").reset_index(drop=True)

    header = df.iloc[3].ffill()
    sex_row = df.iloc[4]
    national = df[df.iloc[:, 0].map(_clean) == "全国"].iloc[0]
    rows = []
    for j in range(4, df.shape[1]):
        parsed = parse_age_label(header.iloc[j])
        sex = _clean(sex_row.iloc[j])
        if parsed is None or sex not in ("男", "女"):
            continue
        start, width = parsed
        rows.append({"start": start, "width": width, "sex": "male" if sex == "男" else "female", "deaths": _num(national.iloc[j])})
    tidy = pd.DataFrame(rows)
    tidy["width"] = tidy["width"].fillna(0).astype(int)  # 0 暂代开放组，透视后再还原
    wide = tidy.pivot(index=["start", "width"], columns="sex", values="deaths").reset_index()
    wide.columns.name = None
    wide["width"] = wide["width"].replace(0, np.nan)
    return wide.sort_values("start").reset_index(drop=True)


@lru_cache(maxsize=None)
def mortality_rates(year: int) -> pd.DataFrame:
    """单岁中心死亡率 m(x)，index=age(0..100)，列 male/female。未做高龄平滑（见 lifetable.kannisto_close）。

    2000 年直接用表中的“平均人口”作暴露；2010/2020 年按组计算 m = D / (P + D/2)，
    其中 P 为普查时点人口（近似期末人口，加半个死亡数近似期中人口），再对 log m 插值到单岁。
    """
    d = deaths_groups(year)
    if year == 2000:
        out = pd.DataFrame(
            {sex: d[sex].to_numpy() / d[f"exposure_{sex}"].to_numpy() for sex in ("male", "female")}, index=d["start"]
        )
        out.index.name = "age"
        return out.loc[0:MAX_AGE]

    pop = population(year)
    widths = d["width"].fillna(1).astype(int).to_numpy()
    out = {}
    for sex in ("male", "female"):
        exposure = []
        for start, width in zip(d["start"], d["width"]):
            ages = [MAX_AGE] if pd.isna(width) else list(range(int(start), int(start + width)))
            exposure.append(pop.loc[ages, sex].sum())
        rates = d[sex].to_numpy() / (np.array(exposure) + d[sex].to_numpy() / 2.0)
        out[sex] = interpolate_log_rates(d["start"].to_numpy(), widths, rates, MAX_AGE)
        out[sex].loc[MAX_AGE] = rates[-1]
    df = pd.DataFrame(out)
    df.index.name = "age"
    return df


@lru_cache(maxsize=None)
def fertility(year: int) -> pd.DataFrame:
    """育龄妇女分年龄、孩次生育（长表抽样）。index=age(15..49)。

    列：women, births, births1/2/3（抽样人数，未放大），asfr, asfr1/2/3（每名妇女，已除以 1000）。
    """
    filename, sheet = FERTILITY_FILES[year]
    df = _read(filename, sheet)
    cols = {"women": 1, "births": 2, "asfr": 3, "births1": 4, "asfr1": 5, "births2": 6, "asfr2": 7, "births3": 8, "asfr3": 9}
    out = _age_rows(df, cols, single_only=True)
    out = out[out["start"].isin(FERTILE_AGES)].set_index("start").drop(columns="width").fillna(0.0)
    out.index.name = "age"
    for c in ("asfr", "asfr1", "asfr2", "asfr3"):
        out[c] = out[c] / 1000.0
    return out


def sample_fraction(year: int) -> float:
    """长表抽样比 = 长表育龄妇女数 / 普查全量 15-49 岁女性人口。"""
    women_sample = fertility(year)["women"].sum()
    women_full = population(year).loc[15:49, "female"].sum()
    return float(women_sample / women_full)


def births_by_sex(year: int) -> dict[str, float]:
    """普查前 12 个月出生人数（全量口径，人），按性别。2000 年由长表按抽样比放大。"""
    df = _read(BIRTH_FILES[year])
    first = df[df.iloc[:, 1].map(lambda v: pd.notna(pd.to_numeric(v, errors="coerce")))].iloc[0]
    scale = 1.0 / sample_fraction(2000) if year == 2000 else 1.0
    return {"total": _num(first.iloc[1]) * scale, "male": _num(first.iloc[2]) * scale, "female": _num(first.iloc[3]) * scale}


def births_total(year: int) -> float:
    return births_by_sex(year)["total"]


def tfr(year: int) -> float:
    return float(fertility(year)["asfr"].sum())


def summary() -> pd.DataFrame:
    rows = []
    for year in YEARS:
        pop = population(year)
        rows.append(
            {
                "year": year,
                "population": pop.to_numpy().sum(),
                "births": births_total(year),
                "deaths": deaths_groups(year)[["male", "female"]].to_numpy().sum(),
                "tfr": tfr(year),
                "sample_fraction": sample_fraction(year),
            }
        )
    return pd.DataFrame(rows).set_index("year")
