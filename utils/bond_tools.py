from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .cnstats_helpers import fetch_stats, prepare_df
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
OBSERVED_DEATH_FILES = {
    2000: "五-t0604.xls",
    2010: "六-A0601.xls",
    2020: "七-A0601.xls",
}
OBSERVED_BIRTH_FILES = {
    2000: "五-l0601.xls",
    2010: "六-A0112.xls",
    2020: "七-A0110.xls",
}
A0303_STATS_FILE = "A0303-stats.xls"
A0303_COLUMN_MAP = {
    "A030301": "年末总人口",
    "A030302": "0-14岁人口",
    "A030303": "15-64岁人口",
    "A030304": "65岁及以上人口",
    "A030305": "总抚养比",
    "A030306": "少儿抚养比",
    "A030307": "老年抚养比",
}
A0303_COLUMNS = ["年份"] + list(A0303_COLUMN_MAP.values())
PERSONS_PER_10K = 10000.0
CORRECTED_HISTORICAL_COHORTS_FILE = "historical-cohorts-corrected-2000-anchor.xls"


def _resolve_dataset_dir(base_dir: Path) -> Path:
    for name in ("Dataset", "dataset"):
        path = base_dir / name
        if path.exists():
            return path
    return base_dir / "Dataset"


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


def load_population_grouped(base_dir: Path, year: int) -> pd.DataFrame:
    path = _resolve_dataset_dir(base_dir) / OBSERVED_POP_FILES[year]
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


def load_population_2020(base_dir: Path) -> pd.DataFrame:
    return load_population_grouped(base_dir, 2020)


def load_population_snapshot(base_dir: Path, year: int) -> pd.DataFrame:
    single_rows: dict[int, dict[str, float]] = {}
    grouped_rows: list[dict[str, object]] = []
    grouped_df = load_population_grouped(base_dir, year)
    for _, row in grouped_df.iterrows():
        ages = row["年龄列表"]
        male = row["男性人口"]
        female = row["女性人口"]
        if len(ages) == 1:
            single_rows[ages[0]] = {"年龄": ages[0], "男性人口": male, "女性人口": female}
        else:
            grouped_rows.append({"年龄组": row["年龄组"], "年龄列表": ages, "男性人口": male, "女性人口": female})

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


def load_birth_total(base_dir: Path, year: int) -> float:
    path = _resolve_dataset_dir(base_dir) / OBSERVED_BIRTH_FILES[year]
    df = pd.read_excel(path, sheet_name=0, header=[3, 4]).dropna(how="all")
    first_row = df.iloc[0]
    return float(pd.to_numeric(pd.Series([first_row[df.columns[1]]]), errors="coerce").iloc[0])


def load_deaths_snapshot(base_dir: Path, year: int) -> pd.DataFrame:
    path = _resolve_dataset_dir(base_dir) / OBSERVED_DEATH_FILES[year]
    df = pd.read_excel(path, sheet_name=0, header=[3, 4]).dropna(how="all")

    if year == 2000:
        age_col = df.columns[0]
        groups = []
        for _, row in df.iterrows():
            label = _clean_label(row[age_col])
            if label in {"", "总计"}:
                continue
            ages = _group_to_ages(label)
            if not ages:
                continue
            groups.append(
                {
                    "年龄组": label,
                    "年龄列表": ages,
                    "总死亡": float(pd.to_numeric(pd.Series([row[df.columns[4]]]), errors="coerce").iloc[0]),
                    "男性死亡": float(pd.to_numeric(pd.Series([row[df.columns[5]]]), errors="coerce").iloc[0]),
                    "女性死亡": float(pd.to_numeric(pd.Series([row[df.columns[6]]]), errors="coerce").iloc[0]),
                }
            )
        return pd.DataFrame(groups)

    region_col = df.columns[0]
    nationwide = df[df[region_col].map(_clean_label) == "全国"]
    if nationwide.empty:
        raise ValueError(f"未找到 {year} 年全国死亡人口行。")
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


def load_deaths_2020(base_dir: Path) -> pd.DataFrame:
    return load_deaths_snapshot(base_dir, 2020)


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


def build_average_life_table(base_dir: Path) -> pd.DataFrame:
    tables = []
    for year in (2000, 2010, 2020):
        pop_groups = load_population_grouped(base_dir, year)
        death_groups = load_deaths_snapshot(base_dir, year)
        table = build_survival_table(pop_groups, death_groups)
        table = table.rename(columns={"男性存活率": f"男性存活率_{year}", "女性存活率": f"女性存活率_{year}"})
        tables.append(table)

    merged = tables[0]
    for table in tables[1:]:
        merged = merged.merge(table, on="年龄", how="outer")
    merged = merged.sort_values("年龄").ffill().bfill()
    male_cols = [c for c in merged.columns if c.startswith("男性存活率_")]
    female_cols = [c for c in merged.columns if c.startswith("女性存活率_")]
    merged["男性存活率"] = merged[male_cols].mean(axis=1)
    merged["女性存活率"] = merged[female_cols].mean(axis=1)
    merged["总存活率"] = (merged["男性存活率"] + merged["女性存活率"]) / 2.0
    return merged[["年龄", "男性存活率", "女性存活率", "总存活率"]]


def build_a0303_stats_table(start_year: int = 1965, end_year: int = 2025) -> pd.DataFrame:
    raw = prepare_df(fetch_stats(zbcode="A0303", datestr=f"{start_year}-{end_year}", dbcode="hgnd"))
    if raw is None or raw.empty:
        return pd.DataFrame(columns=A0303_COLUMNS)

    df = raw.reset_index().copy()
    df["年份"] = pd.to_datetime(df["查询日期"]).dt.year
    wide = df.pivot_table(index="年份", columns="指标代码", values="数值", aggfunc="first").sort_index()
    wide = wide.rename(columns=A0303_COLUMN_MAP)
    wide = wide.reindex(range(start_year, end_year + 1))
    for col in A0303_COLUMN_MAP.values():
        if col not in wide.columns:
            wide[col] = np.nan
    wide = wide[list(A0303_COLUMN_MAP.values())].reset_index()
    return wide


def export_a0303_stats_table(base_dir: Path, start_year: int = 1965, end_year: int = 2025) -> Path:
    dataset_dir = _resolve_dataset_dir(base_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    table = build_a0303_stats_table(start_year=start_year, end_year=end_year)
    output_path = dataset_dir / A0303_STATS_FILE
    table.to_csv(output_path, sep="\t", index=False, encoding="utf-8")
    return output_path


def load_a0303_stats_table(base_dir: Path) -> pd.DataFrame:
    path = _resolve_dataset_dir(base_dir) / A0303_STATS_FILE
    if not path.exists():
        return pd.DataFrame(columns=A0303_COLUMNS)
    head = path.read_text(encoding="utf-8", errors="ignore")[:256].lstrip()
    if head.startswith("<table") or head.startswith("<!DOCTYPE html") or head.startswith("<html"):
        tables = pd.read_html(path)
        if not tables:
            return pd.DataFrame(columns=A0303_COLUMNS)
        df = tables[0].copy()
        if "指标代码" in df.columns:
            df = df.drop(columns=["指标代码"])
    else:
        df = pd.read_csv(path, sep="\t")
    df.columns = [str(col).strip() for col in df.columns]
    for col in A0303_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[A0303_COLUMNS]
    df["年份"] = pd.to_numeric(df["年份"], errors="coerce").astype("Int64")
    for col in A0303_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["年份"]).astype({"年份": int}).sort_values("年份").reset_index(drop=True)


def _derive_a0303_counts_from_ratios(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for idx, row in out.iterrows():
        total = row.get("年末总人口")
        child_ratio = row.get("少儿抚养比")
        old_ratio = row.get("老年抚养比")
        if pd.notna(child_ratio) and pd.notna(old_ratio) and pd.isna(row.get("总抚养比")):
            out.at[idx, "总抚养比"] = float(child_ratio) + float(old_ratio)
        if pd.isna(total) or pd.isna(child_ratio) or pd.isna(old_ratio):
            continue
        labor = total / (1.0 + child_ratio / 100.0 + old_ratio / 100.0)
        child = labor * child_ratio / 100.0
        old = labor * old_ratio / 100.0
        if pd.isna(row.get("15-64岁人口")):
            out.at[idx, "15-64岁人口"] = labor
        if pd.isna(row.get("0-14岁人口")):
            out.at[idx, "0-14岁人口"] = child
        if pd.isna(row.get("65岁及以上人口")):
            out.at[idx, "65岁及以上人口"] = old
    return out


def _collect_a0303_model_row(year: int, total: pd.Series) -> dict[str, float]:
    labor = total.loc[15:64].sum() / PERSONS_PER_10K
    child = total.loc[0:14].sum() / PERSONS_PER_10K
    old = total.loc[65:MAX_AGE].sum() / PERSONS_PER_10K
    return {
        "年份": year,
        "年末总人口": total.loc[0:MAX_AGE].sum() / PERSONS_PER_10K,
        "0-14岁人口": child,
        "15-64岁人口": labor,
        "65岁及以上人口": old,
        "少儿抚养比": child / max(labor, 1.0) * 100.0,
        "老年抚养比": old / max(labor, 1.0) * 100.0,
        "总抚养比": (child + old) / max(labor, 1.0) * 100.0,
    }


def estimate_births_from_population_and_birth_rate(base_dir: Path, year: int) -> float:
    total_df = prepare_df(fetch_stats(zbcode="A030101", datestr=str(year), dbcode="hgnd"))
    rate_df = prepare_df(fetch_stats(zbcode="A030201", datestr=str(year), dbcode="hgnd"))
    total_10k = float(total_df.iloc[0]["数值"])
    birth_rate_per_thousand = float(rate_df.iloc[0]["数值"])
    return total_10k * PERSONS_PER_10K * birth_rate_per_thousand / 1000.0


def _build_historical_cohort_series_from_anchors(
    base_dir: Path,
    birth_anchors: dict[int, float],
    start_birth_year: int = 1965,
    end_birth_year: int = 2020,
) -> pd.DataFrame:
    life_table = build_average_life_table(base_dir).set_index("年龄")
    snapshots = {year: load_population_snapshot(base_dir, year) for year in OBSERVED_POP_FILES}

    def cumulative_survival(age: int) -> float:
        if age <= 0:
            return 1.0
        vals = life_table.loc[list(range(0, min(age, MAX_AGE) + 1)), "总存活率"].to_numpy(dtype=float)
        return float(np.prod(vals))

    segment_scale = {}
    for anchor_year, ref_year in ((2000, 2000), (2010, 2010), (2020, 2020)):
        age = 0
        observed_birth = birth_anchors[anchor_year]
        cohort_pop = float(((_series_from_population(snapshots[ref_year], "男性人口") + _series_from_population(snapshots[ref_year], "女性人口")).loc[age]))
        est_birth = cohort_pop / max(cumulative_survival(age), 1e-9)
        segment_scale[anchor_year] = observed_birth / max(est_birth, 1e-9)

    rows = []
    for birth_year in range(start_birth_year, end_birth_year + 1):
        if birth_year <= 2000:
            ref_year = 2000
            scale = segment_scale[2000]
        elif birth_year <= 2010:
            ref_year = 2010
            scale = segment_scale[2010]
        else:
            ref_year = 2020
            scale = segment_scale[2020]
        age = ref_year - birth_year
        total = _series_from_population(snapshots[ref_year], "男性人口") + _series_from_population(snapshots[ref_year], "女性人口")
        cohort_pop = float(total.loc[age])
        birth_est = cohort_pop / max(cumulative_survival(age), 1e-9)
        birth_est *= scale
        rows.append(
            {
                "出生年份": birth_year,
                "参考普查年": ref_year,
                "参考年龄": age,
                "估计出生人数": birth_est,
                "20岁年份": birth_year + 20,
                "50岁年份": birth_year + 50,
                "65岁年份": birth_year + 65,
                "20岁存活人数估计": birth_est * cumulative_survival(20),
                "50岁存活人数估计": birth_est * cumulative_survival(50),
                "65岁存活人数估计": birth_est * cumulative_survival(65),
                "2000锚点": birth_anchors[2000],
                "2010锚点": birth_anchors[2010],
                "2020锚点": birth_anchors[2020],
            }
        )
    return pd.DataFrame(rows)


def build_historical_cohort_series(base_dir: Path, start_birth_year: int = 1965, end_birth_year: int = 2020) -> pd.DataFrame:
    birth_anchors = {year: load_birth_total(base_dir, year) for year in OBSERVED_BIRTH_FILES}
    return _build_historical_cohort_series_from_anchors(
        base_dir,
        birth_anchors=birth_anchors,
        start_birth_year=start_birth_year,
        end_birth_year=end_birth_year,
    )


def build_corrected_historical_cohort_series(base_dir: Path, start_birth_year: int = 1965, end_birth_year: int = 2020) -> pd.DataFrame:
    birth_anchors = {year: load_birth_total(base_dir, year) for year in OBSERVED_BIRTH_FILES}
    corrected_2000 = estimate_births_from_population_and_birth_rate(base_dir, 2000)
    birth_anchors[2000] = corrected_2000
    corrected = _build_historical_cohort_series_from_anchors(
        base_dir,
        birth_anchors=birth_anchors,
        start_birth_year=start_birth_year,
        end_birth_year=end_birth_year,
    )
    corrected["锚点口径"] = "2000年锚点修正版"
    corrected["2000锚点修正值"] = corrected_2000
    return corrected


def export_corrected_historical_cohort_series(base_dir: Path, start_birth_year: int = 1965, end_birth_year: int = 2020) -> Path:
    dataset_dir = _resolve_dataset_dir(base_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    table = build_corrected_historical_cohort_series(base_dir, start_birth_year=start_birth_year, end_birth_year=end_birth_year)
    output_path = dataset_dir / CORRECTED_HISTORICAL_COHORTS_FILE
    table.to_csv(output_path, sep="\t", index=False, encoding="utf-8")
    return output_path


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


def _add_lookback_axis(ax, offset: int = RETIREMENT_AGE) -> None:
    secax = ax.secondary_xaxis("top", functions=(lambda x: x - offset, lambda x: x + offset))
    secax.set_xlabel(f"对应回看出生年（观察年-{offset}）")


def _backcast_population_series(pop_2020: pd.DataFrame, survival: pd.DataFrame, start_year: int = 2000) -> dict[int, pd.DataFrame]:
    survival = survival.set_index("年龄")
    male = _series_from_population(pop_2020, "男性人口")
    female = _series_from_population(pop_2020, "女性人口")
    population_by_year = {2020: pd.DataFrame({"年龄": range(0, MAX_AGE + 1), "男性人口": male.values, "女性人口": female.values})}

    for year in range(2019, start_year - 1, -1):
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

    return population_by_year


def _backcast_population(
    pop_2020: pd.DataFrame, survival: pd.DataFrame, target_years: tuple[int, ...] = (2010, 2000)
) -> dict[int, pd.DataFrame]:
    population_by_year = _backcast_population_series(pop_2020, survival, start_year=min(target_years))
    return {year: population_by_year[year] for year in (2020,) + target_years}


def build_modeled_population_by_year(base_dir: Path, start_year: int = 1965, end_year: int = 2025) -> dict[int, pd.DataFrame]:
    pop_groups = load_population_2020(base_dir)
    death_groups = load_deaths_2020(base_dir)
    pop_single = load_population_snapshot(base_dir, 2020)
    survival = build_survival_table(pop_groups, death_groups)

    population_by_year = _backcast_population_series(pop_single, survival, start_year=min(start_year, 2020))
    if end_year > 2020:
        simulated = simulate_bond_population(base_dir, end_year=end_year)["population_by_year"]
        for year in range(2021, end_year + 1):
            population_by_year[year] = simulated[year]

    return {year: population_by_year[year] for year in range(start_year, end_year + 1)}


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

    metrics = pd.DataFrame(metrics)
    population_structure = build_model_population_structure_series(population_by_year, start_year=2021, end_year=end_year)
    overlap_end_year = min(end_year, 2025)
    if overlap_end_year >= 2021:
        a0303_overlap = _build_overlap_a0303_calibration(base_dir, population_by_year, start_year=2021, end_year=overlap_end_year)
        population_structure = population_structure.set_index("年份")
        population_structure.update(a0303_overlap.set_index("年份"))
        population_structure = population_structure.reset_index()
    else:
        a0303_overlap = pd.DataFrame(columns=["年份"] + list(A0303_COLUMN_MAP.values()))
    metrics = metrics.merge(population_structure, on="年份", how="left")

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
    historical_cohorts = build_historical_cohort_series(base_dir, start_birth_year=1965, end_birth_year=2020)
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
        "metrics": metrics,
        "a0303_overlap": a0303_overlap,
        "population_structure": population_structure,
        "population_by_year": population_by_year,
        "history_births": history_births,
        "observed_population": observed_population,
        "backcast_population": backcast_population,
        "validation": pd.DataFrame(validation_rows),
        "historical_cohorts": historical_cohorts,
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
    cohorts = result["historical_cohorts"]
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
    print("关键观察年对应的回看出生年（按 65 岁口径）：")
    for year in (2030, 2040, 2050, 2060, 2070):
        print(f"- {year} 年 -> {year - RETIREMENT_AGE} 年")
    print("对应回看出生年历史数据：")
    mapped = cohorts[cohorts["出生年份"].isin([1965, 1975, 1985, 1995, 2005])][
        ["出生年份", "估计出生人数", "20岁年份", "20岁存活人数估计", "50岁年份", "50岁存活人数估计", "65岁年份", "65岁存活人数估计"]
    ]
    print(mapped.to_string(index=False))
    print("回看验证（以 2020 为基年反推）：")
    print(validation.to_string(index=False))


def plot_bond_dashboard(result: dict[str, object]) -> None:
    metrics = result["metrics"]
    history_births = result["history_births"]
    validation = result["validation"]
    confidence = build_confidence_curve(result, max_year=max(2100, int(metrics["年份"].max())))
    cohorts = result["historical_cohorts"]
    population_structure = result.get("population_structure", metrics[["年份"] + list(A0303_COLUMN_MAP.values())].copy())
    overlap = result.get("a0303_overlap", pd.DataFrame(columns=["年份"] + list(A0303_COLUMN_MAP.values())))
    overlap_years = set(overlap["年份"].tolist()) if not overlap.empty else set()

    plt.figure(figsize=(10, 6))
    plt.plot(history_births["年份"], history_births["出生人口"], marker="o", linewidth=2, label="历史出生人口")
    plt.plot(metrics["年份"], metrics["出生人口"], linewidth=2, linestyle="--", label="预测出生人口")
    plt.title("人口债券发行量曲线")
    plt.xlabel("年份")
    plt.ylabel("出生人口")
    plt.grid(alpha=0.2)
    plt.legend()
    _add_lookback_axis(plt.gca())
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(cohorts["出生年份"], cohorts["估计出生人数"], linewidth=2, label="1965年后估计出生人数")
    focus = cohorts[cohorts["出生年份"].isin([1965, 1975, 1985, 1995, 2005])]
    plt.scatter(focus["出生年份"], focus["估计出生人数"], s=60, zorder=5, label="关键回看年份")
    for _, row in focus.iterrows():
        plt.annotate(f"{int(row['出生年份'])}", (row["出生年份"], row["估计出生人数"]), textcoords="offset points", xytext=(0, 6), ha="center")
    plt.title("1965年后的历史发行队列")
    plt.xlabel("出生年份")
    plt.ylabel("估计出生人数")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.plot(metrics["年份"], metrics["20岁人口"], linewidth=2, label="20岁人口")
    ax.plot(metrics["年份"], metrics["65岁人口"], linewidth=2, linestyle="--", label="65岁人口")
    ax.scatter(validation["年份"], validation["观测20岁人口"], color="tab:blue", marker="x", s=60, label="观测20岁人口")
    ax.scatter(validation["年份"], validation["观测65岁人口"], color="tab:orange", marker="x", s=60, label="观测65岁人口")

    cohort_focus = cohorts[cohorts["出生年份"].isin([1965, 1975, 1985, 1995, 2005])].copy()
    cohort_focus["对应观察年"] = cohort_focus["出生年份"] + RETIREMENT_AGE
    ax2 = ax.twinx()
    ax2.plot(
        cohort_focus["对应观察年"],
        cohort_focus["估计出生人数"],
        color="tab:green",
        marker="D",
        linestyle=":",
        linewidth=2,
        label="对应历史出生人数",
    )
    for _, row in cohort_focus.iterrows():
        ax2.annotate(
            f"{int(row['出生年份'])}",
            (row["对应观察年"], row["估计出生人数"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            color="tab:green",
        )

    ax.set_title("代际对比图")
    ax.set_xlabel("年份")
    ax.set_ylabel("人口规模")
    ax2.set_ylabel("对应历史出生人数")
    ax.grid(alpha=0.2)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="best")
    _add_lookback_axis(ax)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(11, 6))
    ax = plt.gca()
    structure_cols = [
        ("年末总人口", "年末总人口", None),
        ("0-14岁人口", "0-14岁人口", "--"),
        ("15-64岁人口", "15-64岁人口", "-."),
        ("65岁及以上人口", "65岁及以上人口", ":"),
    ]
    for col, label, linestyle in structure_cols:
        official_part = population_structure[population_structure["年份"].isin(overlap_years)]
        future_part = population_structure[~population_structure["年份"].isin(overlap_years)]
        if not official_part.empty:
            ax.plot(official_part["年份"], official_part[col], linewidth=2.4, linestyle=linestyle, label=f"{label}（A0303校准 2021-2025）")
        if not future_part.empty:
            ax.plot(future_part["年份"], future_part[col], linewidth=1.8, linestyle=linestyle, alpha=0.85, label=f"{label}（模型延伸）")
    if overlap_years:
        ax.axvline(max(overlap_years), color="tab:gray", linestyle="--", alpha=0.6)
    ax.set_title("总人口与年龄结构衔接图")
    ax.set_xlabel("年份")
    ax.set_ylabel("人口（万人）")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=9)
    _add_lookback_axis(ax)
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
    _add_lookback_axis(plt.gca())
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(metrics["年份"], metrics["PSR"], linewidth=2, label="PSR")
    plt.scatter(validation["年份"], validation["观测PSR"], color="tab:blue", marker="x", s=60, label="观测PSR")
    plt.title("潜在支持比（PSR）")
    plt.xlabel("年份")
    plt.ylabel("PSR")
    plt.grid(alpha=0.2)
    plt.legend()
    _add_lookback_axis(plt.gca())
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(metrics["年份"], metrics["LaborPressure"], linewidth=2, label="Labor Pressure")
    plt.title("劳动力压力指标")
    plt.xlabel("年份")
    plt.ylabel("指标值")
    plt.grid(alpha=0.2)
    plt.legend()
    _add_lookback_axis(plt.gca())
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
    _add_lookback_axis(plt.gca())
    plt.tight_layout()
    plt.show()


def build_calibrated_a0303_series(base_dir: Path, start_year: int = 1965, end_year: int = 2025) -> pd.DataFrame:
    official = load_a0303_stats_table(base_dir)
    official = official.set_index("年份").reindex(range(start_year, end_year + 1))
    official = _derive_a0303_counts_from_ratios(official.reset_index()).set_index("年份")

    modeled_population = build_modeled_population_by_year(base_dir, start_year=start_year, end_year=end_year)
    modeled_rows = []
    for year in range(start_year, end_year + 1):
        total = _series_from_population(modeled_population[year], "男性人口") + _series_from_population(modeled_population[year], "女性人口")
        modeled_rows.append(_collect_a0303_model_row(year, total))
    modeled = pd.DataFrame(modeled_rows).set_index("年份")

    calibrated = official.combine_first(modeled)
    count_cols = ["0-14岁人口", "15-64岁人口", "65岁及以上人口"]
    for year in calibrated.index:
        total_pop = calibrated.at[year, "年末总人口"]
        if pd.isna(total_pop):
            continue
        official_row = official.loc[year] if year in official.index else pd.Series(dtype=float)
        model_row = modeled.loc[year]
        known_cols = [col for col in count_cols if pd.notna(official_row.get(col))]
        missing_cols = [col for col in count_cols if col not in known_cols]
        known_sum = sum(float(official_row[col]) for col in known_cols)
        remaining = max(float(total_pop) - known_sum, 0.0)
        if missing_cols:
            model_missing_sum = sum(float(model_row[col]) for col in missing_cols)
            if model_missing_sum <= 0:
                even_share = remaining / len(missing_cols)
                for col in missing_cols:
                    calibrated.at[year, col] = even_share
            else:
                for col in missing_cols:
                    calibrated.at[year, col] = remaining * float(model_row[col]) / model_missing_sum
        for col in known_cols:
            calibrated.at[year, col] = float(official_row[col])
        labor = float(calibrated.at[year, "15-64岁人口"])
        child = float(calibrated.at[year, "0-14岁人口"])
        old = float(calibrated.at[year, "65岁及以上人口"])
        if pd.isna(official_row.get("少儿抚养比")):
            calibrated.at[year, "少儿抚养比"] = child / max(labor, 1.0) * 100.0
        if pd.isna(official_row.get("老年抚养比")):
            calibrated.at[year, "老年抚养比"] = old / max(labor, 1.0) * 100.0
        if pd.isna(official_row.get("总抚养比")):
            calibrated.at[year, "总抚养比"] = (child + old) / max(labor, 1.0) * 100.0
    calibrated = calibrated.reset_index()
    return calibrated[["年份"] + list(A0303_COLUMN_MAP.values())]


def _build_overlap_a0303_calibration(
    base_dir: Path, population_by_year: dict[int, pd.DataFrame], start_year: int = 2021, end_year: int = 2025
) -> pd.DataFrame:
    official = load_a0303_stats_table(base_dir)
    official = official.set_index("年份").reindex(range(start_year, end_year + 1))
    official = _derive_a0303_counts_from_ratios(official.reset_index()).set_index("年份")

    modeled_rows = []
    for year in range(start_year, end_year + 1):
        total = _series_from_population(population_by_year[year], "男性人口") + _series_from_population(population_by_year[year], "女性人口")
        modeled_rows.append(_collect_a0303_model_row(year, total))
    modeled = pd.DataFrame(modeled_rows).set_index("年份")

    calibrated = official.combine_first(modeled)
    count_cols = ["0-14岁人口", "15-64岁人口", "65岁及以上人口"]
    for year in calibrated.index:
        total_pop = calibrated.at[year, "年末总人口"]
        if pd.isna(total_pop):
            continue
        official_row = official.loc[year] if year in official.index else pd.Series(dtype=float)
        model_row = modeled.loc[year]
        known_cols = [col for col in count_cols if pd.notna(official_row.get(col))]
        missing_cols = [col for col in count_cols if col not in known_cols]
        known_sum = sum(float(official_row[col]) for col in known_cols)
        remaining = max(float(total_pop) - known_sum, 0.0)
        if missing_cols:
            model_missing_sum = sum(float(model_row[col]) for col in missing_cols)
            if model_missing_sum <= 0:
                even_share = remaining / len(missing_cols)
                for col in missing_cols:
                    calibrated.at[year, col] = even_share
            else:
                for col in missing_cols:
                    calibrated.at[year, col] = remaining * float(model_row[col]) / model_missing_sum
        for col in known_cols:
            calibrated.at[year, col] = float(official_row[col])
        labor = float(calibrated.at[year, "15-64岁人口"])
        child = float(calibrated.at[year, "0-14岁人口"])
        old = float(calibrated.at[year, "65岁及以上人口"])
        calibrated.at[year, "少儿抚养比"] = child / max(labor, 1.0) * 100.0
        calibrated.at[year, "老年抚养比"] = old / max(labor, 1.0) * 100.0
        calibrated.at[year, "总抚养比"] = (child + old) / max(labor, 1.0) * 100.0

    calibrated = calibrated.reset_index()
    return calibrated[["年份"] + list(A0303_COLUMN_MAP.values())]


def build_model_population_structure_series(
    population_by_year: dict[int, pd.DataFrame], start_year: int, end_year: int
) -> pd.DataFrame:
    rows = []
    for year in range(start_year, end_year + 1):
        total = _series_from_population(population_by_year[year], "男性人口") + _series_from_population(population_by_year[year], "女性人口")
        rows.append(_collect_a0303_model_row(year, total))
    return pd.DataFrame(rows)


def build_historical_bond_observation(base_dir: Path) -> dict[str, object]:
    observed_population = {year: load_population_snapshot(base_dir, year) for year in OBSERVED_POP_FILES}
    modeled_population = build_modeled_population_by_year(base_dir, start_year=1965, end_year=2025)
    calibrated_a0303 = build_calibrated_a0303_series(base_dir, start_year=1965, end_year=2025).set_index("年份")
    rows = []

    for year in range(1965, 2025 + 1):
        pop_df = modeled_population[year]
        total = _series_from_population(pop_df, "男性人口") + _series_from_population(pop_df, "女性人口")
        official_row = calibrated_a0303.loc[year]
        labor = float(official_row["15-64岁人口"])
        child = float(official_row["0-14岁人口"])
        old = float(official_row["65岁及以上人口"])
        entry = total.loc[20] / PERSONS_PER_10K
        parent = total.loc[50] / PERSONS_PER_10K
        row = {
            "年份": year,
            "年末总人口": float(official_row["年末总人口"]),
            "0-14岁人口": child,
            "15-64岁人口": labor,
            "65岁及以上人口": old,
            "20岁人口": entry,
            "50岁人口": parent,
            "少儿抚养比": child / max(labor, 1.0),
            "老年抚养比": old / max(labor, 1.0),
            "总抚养比": (child + old) / max(labor, 1.0),
            "PSR": total.loc[20:39].sum() / max(total.loc[50:69].sum(), 1.0),
            "LPI": entry / max(labor, 1.0),
            "PBI": parent / max(labor, 1.0),
        }
        rows.append(row)

    indicators = pd.DataFrame(rows).sort_values("年份")
    cohort_backtrack = pd.DataFrame(
        [
            {"出生年份": 1980, "2020回溯队列规模": float((_series_from_population(observed_population[2020], "男性人口") + _series_from_population(observed_population[2020], "女性人口")).loc[40])},
            {"出生年份": 1990, "2020回溯队列规模": float((_series_from_population(observed_population[2020], "男性人口") + _series_from_population(observed_population[2020], "女性人口")).loc[30])},
            {"出生年份": 2000, "2020回溯队列规模": float((_series_from_population(observed_population[2020], "男性人口") + _series_from_population(observed_population[2020], "女性人口")).loc[20])},
        ]
    )

    total_2000 = _series_from_population(observed_population[2000], "男性人口") + _series_from_population(observed_population[2000], "女性人口")
    total_2020 = _series_from_population(observed_population[2020], "男性人口") + _series_from_population(observed_population[2020], "女性人口")
    cohort_survival_1950 = float(total_2020.loc[70] / max(total_2000.loc[50], 1.0))

    tfr_data = load_tfr_data(base_dir)
    fertility_peaks = []
    for year, df in sorted(tfr_data.items()):
        peak_idx = df["年龄别生育率"].idxmax()
        fertility_peaks.append({"年份": year, "峰值年龄": int(df.loc[peak_idx, "年龄"]), "峰值生育率": float(df.loc[peak_idx, "年龄别生育率"])})

    return {
        "population": observed_population,
        "modeled_population": modeled_population,
        "a0303": calibrated_a0303.reset_index(),
        "indicators": indicators,
        "cohort_backtrack": cohort_backtrack,
        "cohort_survival_1950": cohort_survival_1950,
        "fertility_peaks": pd.DataFrame(fertility_peaks),
    }


def print_historical_bond_summary(result: dict[str, object]) -> None:
    print("历史人口债券指标：")
    print(result["indicators"].to_string(index=False))
    print()
    print("2020 回溯出生队列：")
    print(result["cohort_backtrack"].to_string(index=False))
    print()
    print(f"1950 队列 20年实际存活率校验（2000年50岁 -> 2020年70岁）: {result['cohort_survival_1950']:.4f}")
    print("年龄别生育率峰值年龄变化：")
    print(result["fertility_peaks"].to_string(index=False))


def plot_historical_bond_dashboard(result: dict[str, object]) -> None:
    indicators = result["indicators"]
    cohort_backtrack = result["cohort_backtrack"]

    plt.figure(figsize=(10, 6))
    plt.plot(cohort_backtrack["出生年份"], cohort_backtrack["2020回溯队列规模"], marker="o", linewidth=2)
    plt.title("历史债券发行曲线（由2020年回溯）")
    plt.xlabel("出生年份")
    plt.ylabel("队列规模")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(indicators["年份"], indicators["20岁人口"], marker="o", linewidth=2, label="20岁人口")
    plt.plot(indicators["年份"], indicators["50岁人口"], marker="o", linewidth=2, linestyle="--", label="50岁人口")
    plt.title("劳动力入场与父母代规模对比图")
    plt.xlabel("年份")
    plt.ylabel("人口规模")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(indicators["年份"], indicators["少儿抚养比"], marker="o", linewidth=2, label="少儿抚养比")
    plt.plot(indicators["年份"], indicators["老年抚养比"], marker="o", linewidth=2, linestyle="--", label="老年抚养比")
    plt.plot(indicators["年份"], indicators["总抚养比"], marker="o", linewidth=2, linestyle="-.", label="总抚养比")
    plt.title("历史抚养比变化趋势")
    plt.xlabel("年份")
    plt.ylabel("比率")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(indicators["年份"], indicators["PSR"], marker="o", linewidth=2, label="PSR")
    plt.title("历史潜在支持比（PSR）")
    plt.xlabel("年份")
    plt.ylabel("PSR")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(indicators["年份"], indicators["LPI"], marker="o", linewidth=2, linestyle="--", label="LPI")
    plt.plot(indicators["年份"], indicators["PBI"], marker="o", linewidth=2, linestyle="-.", label="PBI")
    plt.title("历史债券周期时间线")
    plt.xlabel("年份")
    plt.ylabel("指标值")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()
