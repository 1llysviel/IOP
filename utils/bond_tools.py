from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .tfr_tools import build_tfr_summary, load_tfr_data


POPULATION_2020_FILE = "七-A0202.xls"
DEATHS_2020_FILE = "七-A0601.xls"
SEX_RATIO_AT_BIRTH = 105 / 100
MAX_AGE = 100


def _clean_label(value: object) -> str:
    return str(value).replace(" ", "").strip()


def _group_to_ages(label: str) -> list[int]:
    if label == "100岁及以上":
        return [100]
    if label.endswith("岁"):
        label = label[:-1]
    if "-" in label:
        start, end = label.split("-", 1)
        return list(range(int(start), int(end) + 1))
    if label.isdigit():
        return [int(label)]
    return []


def load_population_2020(base_dir: Path) -> pd.DataFrame:
    path = base_dir / "Dataset" / POPULATION_2020_FILE
    df = pd.read_excel(path, sheet_name=0, header=[3, 4]).dropna(how="all")
    age_col = df.columns[0]
    out = []
    for _, row in df.iterrows():
        label = _clean_label(row[age_col])
        if label in {"", "总计"}:
            continue
        ages = _group_to_ages(label)
        if not ages:
            continue
        out.append(
            {
                "年龄组": label,
                "年龄列表": ages,
                "总人口": float(pd.to_numeric(pd.Series([row[("合    计", "合计")]]), errors="coerce").iloc[0]),
                "男性人口": float(pd.to_numeric(pd.Series([row[("合    计", "男")]]), errors="coerce").iloc[0]),
                "女性人口": float(pd.to_numeric(pd.Series([row[("合    计", "女")]]), errors="coerce").iloc[0]),
            }
        )
    return pd.DataFrame(out)


def load_deaths_2020(base_dir: Path) -> pd.DataFrame:
    path = base_dir / "Dataset" / DEATHS_2020_FILE
    df = pd.read_excel(path, sheet_name=0, header=[3, 4]).dropna(how="all")
    region_col = df.columns[0]
    nationwide = df[df[region_col].map(_clean_label) == "全国"]
    if nationwide.empty:
        raise ValueError("未找到 2020 年全国死亡人口行。")
    row = nationwide.iloc[0]

    groups = []
    for group, sex in df.columns[4::3]:
        label = _clean_label(group)
        if sex != "小计":
            continue
        ages = _group_to_ages(label)
        if not ages:
            continue
        groups.append(
            {
                "年龄组": label,
                "年龄列表": ages,
                "总死亡": float(pd.to_numeric(pd.Series([row[(group, "小计")]]), errors="coerce").iloc[0]),
                "男性死亡": float(pd.to_numeric(pd.Series([row[(group, "男")]]), errors="coerce").iloc[0]),
                "女性死亡": float(pd.to_numeric(pd.Series([row[(group, "女")]]), errors="coerce").iloc[0]),
            }
        )
    return pd.DataFrame(groups)


def expand_grouped_population(pop_groups: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in pop_groups.iterrows():
        ages = row["年龄列表"]
        n = len(ages)
        for age in ages:
            rows.append(
                {
                    "年龄": age,
                    "男性人口": row["男性人口"] / n,
                    "女性人口": row["女性人口"] / n,
                }
            )
    return pd.DataFrame(rows).groupby("年龄", as_index=False).sum().sort_values("年龄")


def build_survival_table(pop_groups: pd.DataFrame, death_groups: pd.DataFrame) -> pd.DataFrame:
    pop_map = {row["年龄组"]: row for _, row in pop_groups.iterrows()}
    rows = []

    for _, death_row in death_groups.iterrows():
        label = death_row["年龄组"]
        if label == "0岁":
            pop_row = pop_map["0-4岁"]
            ages = [0]
            male_pop = pop_row["男性人口"] / 5.0
            female_pop = pop_row["女性人口"] / 5.0
            male_death = death_row["男性死亡"]
            female_death = death_row["女性死亡"]
            male_survival = np.clip(1.0 - male_death / max(male_pop, 1.0), 0.90, 0.9999)
            female_survival = np.clip(1.0 - female_death / max(female_pop, 1.0), 0.90, 0.9999)
            rows.append({"年龄": 0, "男性存活率": male_survival, "女性存活率": female_survival})
            continue

        if label == "1-4岁":
            pop_row = pop_map["0-4岁"]
            ages = [1, 2, 3, 4]
            male_pop = (pop_row["男性人口"] * 4.0 / 5.0) / 4.0
            female_pop = (pop_row["女性人口"] * 4.0 / 5.0) / 4.0
            male_death = death_row["男性死亡"] / 4.0
            female_death = death_row["女性死亡"] / 4.0
            for age in ages:
                male_survival = np.clip(1.0 - male_death / max(male_pop, 1.0), 0.90, 0.9999)
                female_survival = np.clip(1.0 - female_death / max(female_pop, 1.0), 0.90, 0.9999)
                rows.append({"年龄": age, "男性存活率": male_survival, "女性存活率": female_survival})
            continue

        if label not in pop_map:
            continue

        pop_row = pop_map[label]
        ages = pop_row["年龄列表"]
        n = len(ages)
        for age in ages:
            male_pop = pop_row["男性人口"] / n
            female_pop = pop_row["女性人口"] / n
            male_death = death_row["男性死亡"] / n
            female_death = death_row["女性死亡"] / n
            male_survival = np.clip(1.0 - male_death / max(male_pop, 1.0), 0.90, 0.9999)
            female_survival = np.clip(1.0 - female_death / max(female_pop, 1.0), 0.90, 0.9999)
            rows.append({"年龄": age, "男性存活率": male_survival, "女性存活率": female_survival})

    table = pd.DataFrame(rows).groupby("年龄", as_index=False).mean().sort_values("年龄")
    if 100 not in table["年龄"].values:
        table = pd.concat([table, pd.DataFrame([{"年龄": 100, "男性存活率": 0.90, "女性存活率": 0.92}])], ignore_index=True)
    return table


def build_annual_tfr_series(base_dir: Path, start_year: int = 2021, end_year: int = 2050) -> pd.DataFrame:
    data_by_year = load_tfr_data(base_dir)
    summary = build_tfr_summary(data_by_year)
    years = summary["年份"].to_numpy(dtype=float)
    values = summary["TFR"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(years, values, deg=1)

    rows = []
    for year in range(start_year, end_year + 1):
        tfr = max(float(slope * year + intercept), 0.6)
        rows.append({"年份": year, "TFR": tfr})
    return pd.DataFrame(rows)


def build_fertility_age_pattern(base_dir: Path) -> pd.Series:
    data_by_year = load_tfr_data(base_dir)
    base_df = data_by_year[2020]
    per_woman_rates = base_df.set_index("年龄")["年龄别生育率"] / 1000.0
    pattern = per_woman_rates / per_woman_rates.sum()
    return pattern


def _series_from_population(pop_df: pd.DataFrame, sex_col: str) -> pd.Series:
    return pop_df.set_index("年龄")[sex_col].reindex(range(0, MAX_AGE + 1), fill_value=0.0)


def simulate_bond_population(base_dir: Path, end_year: int = 2050) -> dict[str, object]:
    pop_groups = load_population_2020(base_dir)
    death_groups = load_deaths_2020(base_dir)
    pop_single = expand_grouped_population(pop_groups)
    survival = build_survival_table(pop_groups, death_groups).set_index("年龄")
    tfr_series = build_annual_tfr_series(base_dir, start_year=2021, end_year=end_year).set_index("年份")
    fertility_pattern = build_fertility_age_pattern(base_dir)

    male = _series_from_population(pop_single, "男性人口")
    female = _series_from_population(pop_single, "女性人口")

    population_by_year = {2020: pd.DataFrame({"年龄": range(0, MAX_AGE + 1), "男性人口": male.values, "女性人口": female.values})}
    metrics = []

    male_birth_share = SEX_RATIO_AT_BIRTH / (1.0 + SEX_RATIO_AT_BIRTH)
    female_birth_share = 1.0 - male_birth_share

    for year in range(2021, end_year + 1):
        next_male = pd.Series(0.0, index=range(0, MAX_AGE + 1), dtype=float)
        next_female = pd.Series(0.0, index=range(0, MAX_AGE + 1), dtype=float)

        for age in range(1, MAX_AGE):
            next_male.loc[age] = male.loc[age - 1] * survival.loc[age - 1, "男性存活率"]
            next_female.loc[age] = female.loc[age - 1] * survival.loc[age - 1, "女性存活率"]

        next_male.loc[MAX_AGE] = male.loc[MAX_AGE - 1] * survival.loc[MAX_AGE - 1, "男性存活率"] + male.loc[MAX_AGE] * survival.loc[MAX_AGE, "男性存活率"]
        next_female.loc[MAX_AGE] = female.loc[MAX_AGE - 1] * survival.loc[MAX_AGE - 1, "女性存活率"] + female.loc[MAX_AGE] * survival.loc[MAX_AGE, "女性存活率"]

        age_specific_rates = fertility_pattern * tfr_series.loc[year, "TFR"]
        births = 0.0
        for age in range(15, 50):
            births += next_female.loc[age] * float(age_specific_rates.loc[age])

        next_male.loc[0] = births * male_birth_share * survival.loc[0, "男性存活率"]
        next_female.loc[0] = births * female_birth_share * survival.loc[0, "女性存活率"]

        total = next_male + next_female
        labor_force = total.loc[20:64].sum()
        entry = total.loc[20]
        parent = total.loc[50]
        psr = total.loc[20:39].sum() / max(total.loc[50:69].sum(), 1.0)
        labor_pressure = entry / max(labor_force, 1.0)
        odr = total.loc[65:MAX_AGE].sum() / max(labor_force, 1.0)
        cdr = total.loc[0:19].sum() / max(labor_force, 1.0)

        metrics.append(
            {
                "年份": year,
                "出生人口": births,
                "TFR": float(tfr_series.loc[year, "TFR"]),
                "20岁人口": entry,
                "50岁人口": parent,
                "PSR": psr,
                "LaborPressure": labor_pressure,
                "ODR": odr,
                "CDR": cdr,
                "TDR": odr + cdr,
            }
        )

        male = next_male
        female = next_female
        population_by_year[year] = pd.DataFrame({"年龄": range(0, MAX_AGE + 1), "男性人口": male.values, "女性人口": female.values})

    history_births = pd.DataFrame(
        [
            {"年份": 2000, "出生人口": 1182138.0},
            {"年份": 2010, "出生人口": 13836187.0},
            {"年份": 2020, "出生人口": 12004692.0},
        ]
    )

    return {
        "population_2020": population_by_year[2020],
        "survival_table": survival.reset_index(),
        "tfr_series": tfr_series.reset_index(),
        "metrics": pd.DataFrame(metrics),
        "population_by_year": population_by_year,
        "history_births": history_births,
    }


def print_bond_summary(result: dict[str, object]) -> None:
    metrics = result["metrics"]
    print("人口超长期债券模型摘要：")
    print(f"- 2021-2050 年 TFR 均值: {metrics['TFR'].mean():.3f}")
    print(f"- 2021-2050 年平均出生人口: {metrics['出生人口'].mean():.0f}")
    print(f"- 2050 年 20岁人口: {metrics.loc[metrics['年份'] == 2050, '20岁人口'].iloc[0]:.0f}")
    print(f"- 2050 年 50岁人口: {metrics.loc[metrics['年份'] == 2050, '50岁人口'].iloc[0]:.0f}")
    print(f"- 2050 年 PSR: {metrics.loc[metrics['年份'] == 2050, 'PSR'].iloc[0]:.3f}")
    print(f"- 2050 年 ODR: {metrics.loc[metrics['年份'] == 2050, 'ODR'].iloc[0]:.3f}")
    print(f"- 2050 年 CDR: {metrics.loc[metrics['年份'] == 2050, 'CDR'].iloc[0]:.3f}")
    print(f"- 2050 年 TDR: {metrics.loc[metrics['年份'] == 2050, 'TDR'].iloc[0]:.3f}")


def plot_bond_dashboard(result: dict[str, object]) -> None:
    metrics = result["metrics"]
    history_births = result["history_births"]

    plt.figure(figsize=(10, 6))
    plt.plot(history_births["年份"], history_births["出生人口"], marker="o", linewidth=2, label="历史出生人口")
    plt.plot(metrics["年份"], metrics["出生人口"], linewidth=2, linestyle="--", label="预测出生人口")
    plt.title("人口债券发行量曲线")
    plt.xlabel("年份")
    plt.ylabel("出生人口")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(metrics["年份"], metrics["20岁人口"], linewidth=2, label="20岁人口")
    plt.plot(metrics["年份"], metrics["50岁人口"], linewidth=2, linestyle="--", label="50岁人口")
    plt.title("代际对比图")
    plt.xlabel("年份")
    plt.ylabel("人口规模")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(metrics["年份"], metrics["CDR"], linewidth=2, label="少儿抚养比")
    plt.plot(metrics["年份"], metrics["ODR"], linewidth=2, linestyle="--", label="老年抚养比")
    plt.plot(metrics["年份"], metrics["TDR"], linewidth=2, linestyle="-.", label="总抚养比")
    plt.title("抚养比演变图")
    plt.xlabel("年份")
    plt.ylabel("比率")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(metrics["年份"], metrics["PSR"], linewidth=2, label="PSR")
    plt.plot(metrics["年份"], metrics["LaborPressure"], linewidth=2, linestyle="--", label="Labor Pressure")
    plt.title("债券期限结构与劳动力压力")
    plt.xlabel("年份")
    plt.ylabel("指标值")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()
