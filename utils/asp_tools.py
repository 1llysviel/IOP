from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .demography import strip_spaces


AGE_GROUPS = ["15-19岁", "20-24岁", "25-29岁", "30-34岁", "35-39岁", "40-44岁", "45-49岁"]
SHIFT = 2
DEFAULT_ASP_FILES = {
    2000: "五-t0604.xls",
    2010: "六-A0601.xls",
    2020: "七-A0601.xls",
}


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


def load_asp_history(base_dir: Path) -> dict[str, object]:
    f2000 = base_dir / "Dataset" / DEFAULT_ASP_FILES[2000]
    f2010 = base_dir / "Dataset" / DEFAULT_ASP_FILES[2010]
    f2020 = base_dir / "Dataset" / DEFAULT_ASP_FILES[2020]

    base_2000 = load_2000_population_and_mortality(f2000)
    d2010 = load_national_female_deaths(f2010)
    d2020 = load_national_female_deaths(f2020)

    p2010 = estimate_population(d2010, base_2000)
    p2020 = estimate_population(d2020, base_2000)

    pop_2000 = base_2000.set_index("年龄组")["女性人口"]
    pop_2010 = p2010.set_index("年龄组")["女性人口估计"]
    pop_2020 = p2020.set_index("年龄组")["女性人口估计"]

    s0010 = cohort_survival(pop_2000, pop_2010)
    s1020 = cohort_survival(pop_2010, pop_2020)

    return {
        "base_2000": base_2000,
        "deaths_2010": d2010,
        "deaths_2020": d2020,
        "populations": {2000: pop_2000, 2010: pop_2010, 2020: pop_2020},
        "survival": {(2000, 2010): s0010, (2010, 2020): s1020},
    }


def print_asp_summary(history: dict[str, object], forecast_result: dict[str, object] | None = None) -> None:
    s0010 = history["survival"][(2000, 2010)]
    s1020 = history["survival"][(2010, 2020)]

    print("说明：2010/2020 年女性人口由“女性死亡人数 / 2000年同年龄女性死亡率”反推得到。")
    print("2000→2010 存活率（15-19→25-29 ... 35-39→45-49）：")
    print(s0010.to_string(index=False))
    print(f"平均存活率: {s0010['存活率'].mean():.4f}")
    print()
    print("2010→2020 存活率（15-19→25-29 ... 35-39→45-49）：")
    print(s1020.to_string(index=False))
    print(f"平均存活率: {s1020['存活率'].mean():.4f}")

    if forecast_result is not None:
        print()
        print("未来队列推移预测：")
        for year, df in forecast_result["survival"].items():
            print(f"{year-10}→{year} 存活率：")
            print(df.to_string(index=False))
            print(f"平均存活率: {df['存活率'].mean():.4f}")


def _entry_projection(history_pops: dict[int, pd.Series], age_group: str, target_year: int) -> float:
    years = np.array(sorted(history_pops.keys()), dtype=float)
    values = np.array([history_pops[int(year)].loc[age_group] for year in years], dtype=float)
    slope, intercept = np.polyfit(years, values, deg=1)
    return max(float(slope * target_year + intercept), 0.0)


def forecast_asp_cohort_shift(history: dict[str, object], target_years: tuple[int, ...] = (2030, 2040)) -> dict[str, object]:
    populations = dict(history["populations"])
    hist_survival = history["survival"]
    avg_survival = {}
    s0010 = hist_survival[(2000, 2010)].set_index("起始年龄组")["存活率"]
    s1020 = hist_survival[(2010, 2020)].set_index("起始年龄组")["存活率"]
    for age in s0010.index:
        avg_survival[age] = float((s0010.loc[age] + s1020.loc[age]) / 2.0)

    forecast_survival: dict[int, pd.DataFrame] = {}
    for target_year in target_years:
        prev_year = target_year - 10
        prev_pop = populations[prev_year]
        rows = []

        projected = {
            "15-19岁": _entry_projection(populations, "15-19岁", target_year),
            "20-24岁": _entry_projection(populations, "20-24岁", target_year),
        }
        for i in range(len(AGE_GROUPS) - SHIFT):
            age_from = AGE_GROUPS[i]
            age_to = AGE_GROUPS[i + SHIFT]
            projected[age_to] = float(prev_pop.loc[age_from] * avg_survival[age_from])
            rows.append({"起始年龄组": age_from, "10年后年龄组": age_to, "存活率": avg_survival[age_from]})

        populations[target_year] = pd.Series(projected).reindex(AGE_GROUPS)
        forecast_survival[target_year] = pd.DataFrame(rows)

    return {"populations": populations, "survival": forecast_survival}


def plot_asp_dashboard(
    pop_2000: pd.Series,
    pop_2010: pd.Series,
    pop_2020: pd.Series,
    s0010: pd.DataFrame,
    s1020: pd.DataFrame,
    forecast_result: dict[str, object] | None = None,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(AGE_GROUPS, [pop_2000[a] for a in AGE_GROUPS], label="2000 年（原始）", linewidth=2)
    plt.plot(AGE_GROUPS, [pop_2010[a] for a in AGE_GROUPS], label="2010 年（反推估计）", linewidth=2, linestyle="--")
    plt.plot(AGE_GROUPS, [pop_2020[a] for a in AGE_GROUPS], label="2020 年（反推估计）", linewidth=2, linestyle="-.")
    if forecast_result is not None:
        for year in sorted(k for k in forecast_result["populations"].keys() if k >= 2030):
            pop = forecast_result["populations"][year]
            plt.plot(AGE_GROUPS, [pop[a] for a in AGE_GROUPS], label=f"{year} 年（队列推移预测）", linewidth=2, linestyle=":")
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

    if forecast_result is not None:
        plt.figure(figsize=(10, 6))
        for year, df in sorted(forecast_result["survival"].items()):
            plt.plot(df["起始年龄组"], df["存活率"], marker="o", linewidth=2, label=f"{year-10}→{year} 预测")
        plt.title("未来15-49岁女性10年队列推移存活率预测")
        plt.xlabel("起始年龄组")
        plt.ylabel("存活率")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()
