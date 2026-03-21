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
RETIREMENT_AGE = 65
LIFE_EXPECTANCY = 78
OBSERVED_POP_FILES = {
    2000: "五-t0201.xls",
    2010: "六-A0201.xls",
    2020: "七-A0202.xls",
}


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


def _find_total_columns(columns: pd.Index) -> tuple[tuple[str, str], tuple[str, str], tuple[str, str]]:
    total_col = male_col = female_col = None
    for col in columns:
        if not isinstance(col, tuple) or len(col) != 2:
            continue
        top = _clean_label(col[0])
        bottom = _clean_label(col[1])
        if top == "合计" and bottom in {"合计", "小计"}:
            total_col = col
        elif top == "合计" and bottom == "男":
            male_col = col
        elif top == "合计" and bottom == "女":
            female_col = col
    if total_col is None or male_col is None or female_col is None:
        raise KeyError("未能在人口表中定位总计/男/女三列。")
    return total_col, male_col, female_col


def load_population_2020(base_dir: Path) -> pd.DataFrame:
    path = base_dir / "Dataset" / POPULATION_2020_FILE
    df = pd.read_excel(path, sheet_name=0, header=[3, 4]).dropna(how="all")
    age_col = df.columns[0]
    total_col, male_col, female_col = _find_total_columns(df.columns)
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
                "总人口": float(pd.to_numeric(pd.Series([row[total_col]]), errors="coerce").iloc[0]),
                "男性人口": float(pd.to_numeric(pd.Series([row[male_col]]), errors="coerce").iloc[0]),
                "女性人口": float(pd.to_numeric(pd.Series([row[female_col]]), errors="coerce").iloc[0]),
            }
        )
    return pd.DataFrame(out)


def load_population_snapshot(base_dir: Path, year: int) -> pd.DataFrame:
    path = base_dir / "Dataset" / OBSERVED_POP_FILES[year]
    df = pd.read_excel(path, sheet_name=0, header=[3, 4]).dropna(how="all")
    age_col = df.columns[0]
    _, male_col, female_col = _find_total_columns(df.columns)

    single_rows: dict[int, dict[str, float]] = {}
    grouped_rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        label = _clean_label(row[age_col])
        if label in {"", "总计"}:
            continue
        ages = _group_to_ages(label)
        if not ages:
            continue
        male = float(pd.to_numeric(pd.Series([row[male_col]]), errors="coerce").iloc[0])
        female = float(pd.to_numeric(pd.Series([row[female_col]]), errors="coerce").iloc[0])
        if len(ages) == 1:
            single_rows[ages[0]] = {"年龄": ages[0], "男性人口": male, "女性人口": female}
        else:
            grouped_rows.append({"年龄组": label, "年龄列表": ages, "男性人口": male, "女性人口": female})

    rows = list(single_rows.values())
    for group in grouped_rows:
        uncovered = [age for age in group["年龄列表"] if age not in single_rows]
        if not uncovered:
            continue
        n = len(uncovered)
        for age in uncovered:
            rows.append({"年龄": age, "男性人口": group["男性人口"] / n, "女性人口": group["女性人口"] / n})

    out = pd.DataFrame(rows).groupby("年龄", as_index=False).sum().sort_values("年龄")
    return out[(out["年龄"] >= 0) & (out["年龄"] <= MAX_AGE)]


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


def _collect_indicator_row(year: int, total: pd.Series, births: float | None, tfr: float | None) -> dict[str, float]:
    labor_force = total.loc[20:64].sum()
    retirement_window = total.loc[RETIREMENT_AGE:LIFE_EXPECTANCY].sum()
    row = {
        "年份": year,
        "出生人口": np.nan if births is None else births,
        "TFR": np.nan if tfr is None else tfr,
        "20岁人口": total.loc[20],
        "65岁人口": total.loc[RETIREMENT_AGE],
        "PSR": total.loc[20:39].sum() / max(retirement_window, 1.0),
        "LaborPressure": total.loc[20] / max(labor_force, 1.0),
        "ODR": retirement_window / max(labor_force, 1.0),
        "CDR": total.loc[0:19].sum() / max(labor_force, 1.0),
    }
    row["TDR"] = row["ODR"] + row["CDR"]
    return row


def _backcast_population(
    pop_2020: pd.DataFrame, survival: pd.DataFrame, target_years: tuple[int, ...] = (2010, 2000)
) -> dict[int, pd.DataFrame]:
    survival = survival.set_index("年龄")
    male = _series_from_population(pop_2020, "男性人口")
    female = _series_from_population(pop_2020, "女性人口")
    population_by_year = {2020: pd.DataFrame({"年龄": range(0, MAX_AGE + 1), "男性人口": male.values, "女性人口": female.values})}

    for year in range(2019, min(target_years) - 1, -1):
        prev_male = pd.Series(0.0, index=range(0, MAX_AGE + 1), dtype=float)
        prev_female = pd.Series(0.0, index=range(0, MAX_AGE + 1), dtype=float)
        prev_male.loc[0] = male.loc[1] / max(survival.loc[0, "男性存活率"], 1e-6)
        prev_female.loc[0] = female.loc[1] / max(survival.loc[0, "女性存活率"], 1e-6)
        for age in range(1, MAX_AGE):
            prev_male.loc[age] = male.loc[age + 1] / max(survival.loc[age, "男性存活率"], 1e-6)
            prev_female.loc[age] = female.loc[age + 1] / max(survival.loc[age, "女性存活率"], 1e-6)
        prev_male.loc[MAX_AGE] = male.loc[MAX_AGE] / max(survival.loc[MAX_AGE, "男性存活率"], 1e-6)
        prev_female.loc[MAX_AGE] = female.loc[MAX_AGE] / max(survival.loc[MAX_AGE, "女性存活率"], 1e-6)
        male = prev_male
        female = prev_female
        population_by_year[year] = pd.DataFrame({"年龄": range(0, MAX_AGE + 1), "男性人口": male.values, "女性人口": female.values})

    return {year: population_by_year[year] for year in (2020,) + target_years}


def simulate_bond_population(base_dir: Path, end_year: int = 2070) -> dict[str, object]:
    pop_groups = load_population_2020(base_dir)
    death_groups = load_deaths_2020(base_dir)
    pop_single = load_population_snapshot(base_dir, 2020)
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
        metrics.append(_collect_indicator_row(year, total, births, float(tfr_series.loc[year, "TFR"])))

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

    observed_population = {year: load_population_snapshot(base_dir, year) for year in OBSERVED_POP_FILES}
    backcast_population = _backcast_population(observed_population[2020], survival.reset_index())
    tfr_summary = build_tfr_summary(load_tfr_data(base_dir)).set_index("年份")
    validation_rows = []
    for year in (2000, 2010, 2020):
        observed_total = _series_from_population(observed_population[year], "男性人口") + _series_from_population(observed_population[year], "女性人口")
        observed_row = _collect_indicator_row(year, observed_total, None, float(tfr_summary.loc[year, "TFR"]))
        model_total = _series_from_population(backcast_population[year], "男性人口") + _series_from_population(backcast_population[year], "女性人口")
        model_row = _collect_indicator_row(year, model_total, None, float(tfr_summary.loc[year, "TFR"]) if year in tfr_summary.index else np.nan)
        validation_rows.append(
            {
                "年份": year,
                "观测20岁人口": observed_row["20岁人口"],
                "模型20岁人口": model_row["20岁人口"],
                "观测65岁人口": observed_row["65岁人口"],
                "模型65岁人口": model_row["65岁人口"],
                "观测PSR": observed_row["PSR"],
                "模型PSR": model_row["PSR"],
                "观测ODR": observed_row["ODR"],
                "模型ODR": model_row["ODR"],
            }
        )

    return {
        "population_2020": population_by_year[2020],
        "survival_table": survival.reset_index(),
        "tfr_series": tfr_series.reset_index(),
        "metrics": pd.DataFrame(metrics),
        "population_by_year": population_by_year,
        "history_births": history_births,
        "observed_population": observed_population,
        "backcast_population": backcast_population,
        "validation": pd.DataFrame(validation_rows),
    }


def build_confidence_curve(result: dict[str, object], max_year: int = 2100) -> dict[str, object]:
    validation = result["validation"].copy()
    validation["回看年数"] = 2020 - validation["年份"]

    error_rows = []
    for _, row in validation.iterrows():
        rel_errors = [
            abs(row["模型20岁人口"] - row["观测20岁人口"]) / max(row["观测20岁人口"], 1.0),
            abs(row["模型65岁人口"] - row["观测65岁人口"]) / max(row["观测65岁人口"], 1.0),
            abs(row["模型PSR"] - row["观测PSR"]) / max(row["观测PSR"], 1e-6),
            abs(row["模型ODR"] - row["观测ODR"]) / max(row["观测ODR"], 1e-6),
        ]
        avg_error = float(np.mean(rel_errors))
        confidence = float(np.exp(-avg_error))
        error_rows.append({"回看年数": int(row["回看年数"]), "平均相对误差": avg_error, "经验置信度": confidence})

    anchor = pd.DataFrame([{"回看年数": 0, "平均相对误差": 0.0, "经验置信度": 1.0}])
    fit_df = pd.concat([anchor, pd.DataFrame(error_rows)], ignore_index=True).sort_values("回看年数")

    c10 = float(fit_df.loc[fit_df["回看年数"] == 10, "经验置信度"].iloc[0])
    c20 = float(fit_df.loc[fit_df["回看年数"] == 20, "经验置信度"].iloc[0])
    e10 = max(-np.log(c10), 1e-6)
    e20 = max(-np.log(c20), e10 + 1e-6)
    p = np.clip(np.log(e20 / e10) / np.log(2.0), 0.5, 3.0)
    k = e10 / (10.0**p)

    rows = []
    for year in range(2020, max_year + 1):
        horizon = year - 2020
        confidence = float(np.exp(-(k * (horizon**p)))) if horizon > 0 else 1.0
        rows.append({"年份": year, "预测年数": horizon, "置信度": confidence})
    curve = pd.DataFrame(rows)

    recommended = int(curve.loc[curve["置信度"] >= 0.5, "年份"].max())
    cautious = int(curve.loc[curve["置信度"] >= 0.3, "年份"].max())
    return {
        "fit_points": fit_df,
        "curve": curve,
        "recommended_year": recommended,
        "cautious_year": cautious,
        "params": {"k": float(k), "p": float(p)},
    }


def print_bond_summary(result: dict[str, object]) -> None:
    metrics = result["metrics"]
    validation = result["validation"]
    confidence = build_confidence_curve(result)
    print("人口超长期债券模型摘要：")
    print(f"- 2021-2070 年 TFR 均值: {metrics['TFR'].mean():.3f}")
    print(f"- 2021-2070 年平均出生人口: {metrics['出生人口'].mean():.0f}")
    for year in (2030, 2050, 2060, 2070):
        row = metrics.loc[metrics["年份"] == year].iloc[0]
        conf = confidence["curve"].loc[confidence["curve"]["年份"] == year, "置信度"].iloc[0]
        print(f"- {year} 年: 20岁人口 {row['20岁人口']:.0f}, 65岁人口 {row['65岁人口']:.0f}, PSR {row['PSR']:.3f}, TDR {row['TDR']:.3f}, 置信度 {conf:.3f}")
    print(f"- 2050 年 20岁人口: {metrics.loc[metrics['年份'] == 2050, '20岁人口'].iloc[0]:.0f}")
    print(f"- 2050 年 65岁人口: {metrics.loc[metrics['年份'] == 2050, '65岁人口'].iloc[0]:.0f}")
    print(f"- 2050 年 PSR: {metrics.loc[metrics['年份'] == 2050, 'PSR'].iloc[0]:.3f}")
    print(f"- 2050 年 ODR: {metrics.loc[metrics['年份'] == 2050, 'ODR'].iloc[0]:.3f}")
    print(f"- 2050 年 CDR: {metrics.loc[metrics['年份'] == 2050, 'CDR'].iloc[0]:.3f}")
    print(f"- 2050 年 TDR: {metrics.loc[metrics['年份'] == 2050, 'TDR'].iloc[0]:.3f}")
    print(f"- 建议推演上限（置信度>=0.50）: {confidence['recommended_year']} 年")
    print(f"- 谨慎观察上限（置信度>=0.30）: {confidence['cautious_year']} 年")
    print("回看验证（以 2020 为基年反推）：")
    print(validation.to_string(index=False))


def plot_bond_dashboard(result: dict[str, object]) -> None:
    metrics = result["metrics"]
    history_births = result["history_births"]
    validation = result["validation"]
    confidence = build_confidence_curve(result, max_year=max(2100, int(metrics["年份"].max())))

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
    plt.plot(metrics["年份"], metrics["65岁人口"], linewidth=2, linestyle="--", label="65岁人口")
    plt.scatter(validation["年份"], validation["观测20岁人口"], color="tab:blue", marker="x", s=60, label="观测20岁人口")
    plt.scatter(validation["年份"], validation["观测65岁人口"], color="tab:orange", marker="x", s=60, label="观测65岁人口")
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
    plt.scatter(validation["年份"], validation["观测PSR"], color="tab:blue", marker="x", s=60, label="观测PSR")
    plt.title("债券期限结构与劳动力压力")
    plt.xlabel("年份")
    plt.ylabel("指标值")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(confidence["curve"]["年份"], confidence["curve"]["置信度"], linewidth=2, label="置信度曲线")
    plt.axhline(0.5, color="tab:orange", linestyle="--", alpha=0.7, label="建议阈值 0.50")
    plt.axhline(0.3, color="tab:red", linestyle=":", alpha=0.7, label="谨慎阈值 0.30")
    for year in (2030, 2050, 2060, 2070):
        if year in confidence["curve"]["年份"].values:
            value = confidence["curve"].loc[confidence["curve"]["年份"] == year, "置信度"].iloc[0]
            plt.scatter([year], [value], s=50, zorder=5)
    plt.title("模型置信度曲线")
    plt.xlabel("年份")
    plt.ylabel("置信度")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()
