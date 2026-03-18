from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .demography import strip_spaces


AGE_GROUPS = ["15-19岁", "20-24岁", "25-29岁", "30-34岁", "35-39岁", "40-44岁", "45-49岁"]
SHIFT = 2


def load_2000_population_and_mortality(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0, header=[3, 4]).dropna(how="all")
    age_col = df.columns[0]
    df["年龄组"] = df[age_col].map(strip_spaces)
    df = df[df["年龄组"].isin(AGE_GROUPS)].copy()

    df["女性人口"] = pd.to_numeric(df.iloc[:, 3], errors="coerce")
    df["女性死亡"] = pd.to_numeric(df.iloc[:, 6], errors="coerce")
    df["女性死亡率"] = df["女性死亡"] / df["女性人口"]
    out = df[["年龄组", "女性人口", "女性死亡", "女性死亡率"]].copy()
    out.columns = ["年龄组", "女性人口", "女性死亡", "女性死亡率"]
    return out


def load_national_female_deaths(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0, header=[3, 4]).dropna(how="all")
    region_col = df.columns[0]
    nationwide = df[df[region_col].map(strip_spaces) == "全国"]
    if nationwide.empty:
        raise ValueError(f"未在 {path.name} 中找到“全国”行。")
    row = nationwide.iloc[0]

    out = []
    for age in AGE_GROUPS:
        female_death = pd.to_numeric(pd.Series([row[(age, "女")]]), errors="coerce").iloc[0]
        out.append({"年龄组": age, "女性死亡": female_death})
    return pd.DataFrame(out)


def estimate_population(female_deaths: pd.DataFrame, base_rates: pd.DataFrame) -> pd.DataFrame:
    merged = female_deaths.merge(base_rates[["年龄组", "女性死亡率"]], on="年龄组", how="left")
    merged["女性人口估计"] = merged["女性死亡"] / merged["女性死亡率"]
    return merged[["年龄组", "女性死亡", "女性人口估计"]]


def cohort_survival(base_pop: pd.Series, next_pop: pd.Series) -> pd.DataFrame:
    rows = []
    for i in range(len(AGE_GROUPS) - SHIFT):
        age_from = AGE_GROUPS[i]
        age_to = AGE_GROUPS[i + SHIFT]
        rows.append({"起始年龄组": age_from, "10年后年龄组": age_to, "存活率": next_pop.loc[age_to] / base_pop.loc[age_from]})
    return pd.DataFrame(rows)


def plot_asp_dashboard(
    pop_2000: pd.Series, pop_2010: pd.Series, pop_2020: pd.Series, s0010: pd.DataFrame, s1020: pd.DataFrame
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(AGE_GROUPS, [pop_2000[a] for a in AGE_GROUPS], label="2000 年（原始）", linewidth=2)
    plt.plot(AGE_GROUPS, [pop_2010[a] for a in AGE_GROUPS], label="2010 年（反推估计）", linewidth=2, linestyle="--")
    plt.plot(AGE_GROUPS, [pop_2020[a] for a in AGE_GROUPS], label="2020 年（反推估计）", linewidth=2, linestyle="-.")
    plt.title("15-49岁女性人口（队列推移前置估计）")
    plt.xlabel("年龄组")
    plt.ylabel("女性人口")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(s0010["起始年龄组"], s0010["存活率"], marker="o", label="2000→2010")
    plt.plot(s1020["起始年龄组"], s1020["存活率"], marker="o", label="2010→2020", linestyle="--")
    plt.axhline(s0010["存活率"].mean(), color="tab:blue", alpha=0.2, linestyle=":")
    plt.axhline(s1020["存活率"].mean(), color="tab:orange", alpha=0.2, linestyle=":")
    plt.title("15-49岁女性10年队列推移存活率")
    plt.xlabel("起始年龄组")
    plt.ylabel("存活率")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()
