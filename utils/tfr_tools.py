from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .demography import parse_single_age

PARITY_RATE_COLUMNS = ["一孩生育率", "二孩生育率", "三孩及以上生育率"]
PARITY_COUNT_COLUMNS = ["一孩", "二孩", "三孩及以上"]
DEFAULT_TFR_TABLES = [
    {"filename": "五-l0606.xls", "sheet": "长表6-6", "year": 2000, "kind": "asfr"},
    {"filename": "六-B0603.xls", "sheet": "Sheet1", "year": 2010, "kind": "asfr"},
    {"filename": "七-B0603.xls", "sheet": "Sheet1", "year": 2020, "kind": "asfr"},
]


def load_age_metrics(path: Path, sheet: str, kind: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet, header=[3, 4]).dropna(how="all")
    rows: list[dict[str, float | int]] = []
    col0 = raw.columns[0]

    for _, row in raw.iterrows():
        first_col = row[col0]
        first_text = "" if pd.isna(first_col) else str(first_col).strip()
        if kind == "parity" and first_text in {"未上过学", "小  学", "初  中", "高  中", "大  专", "大  学", "研究生"}:
            break

        age = parse_single_age(first_col)
        if age is None:
            continue

        if kind == "asfr":
            women = pd.to_numeric(pd.Series([row.iloc[1]]), errors="coerce").iloc[0]
            total = pd.to_numeric(pd.Series([row.iloc[2]]), errors="coerce").iloc[0]
            first = pd.to_numeric(pd.Series([row.iloc[4]]), errors="coerce").iloc[0]
            first_rate = pd.to_numeric(pd.Series([row.iloc[5]]), errors="coerce").iloc[0]
            second = pd.to_numeric(pd.Series([row.iloc[6]]), errors="coerce").iloc[0]
            second_rate = pd.to_numeric(pd.Series([row.iloc[7]]), errors="coerce").iloc[0]
            third = pd.to_numeric(pd.Series([row.iloc[8]]), errors="coerce").iloc[0]
            third_rate = pd.to_numeric(pd.Series([row.iloc[9]]), errors="coerce").iloc[0]
            asfr = pd.to_numeric(pd.Series([row.iloc[3]]), errors="coerce").iloc[0]
        else:
            women = pd.to_numeric(pd.Series([row.iloc[1]]), errors="coerce").iloc[0]
            total = pd.NA
            first = pd.to_numeric(pd.Series([row.iloc[4]]), errors="coerce").iloc[0]
            first_rate = pd.NA
            second = pd.to_numeric(pd.Series([row.iloc[7]]), errors="coerce").iloc[0]
            second_rate = pd.NA
            third = pd.to_numeric(pd.Series([row.iloc[10]]), errors="coerce").iloc[0]
            third_rate = pd.NA
            asfr = pd.NA

        if pd.isna(women) or women == 0:
            continue

        rows.append(
            {
                "年龄": age,
                "妇女人数": float(women),
                "总出生人数": float(total) if pd.notna(total) else 0.0,
                "一孩": float(first) if pd.notna(first) else 0.0,
                "二孩": float(second) if pd.notna(second) else 0.0,
                "三孩及以上": float(third) if pd.notna(third) else 0.0,
                "一孩生育率": first_rate,
                "二孩生育率": second_rate,
                "三孩及以上生育率": third_rate,
                "年龄别生育率": asfr,
            }
        )

    df = pd.DataFrame(rows).sort_values("年龄")
    total_births = df["总出生人数"].replace(0, np.nan)
    df["一孩占比"] = df["一孩"] / total_births
    df["二孩占比"] = df["二孩"] / total_births
    df["三孩及以上占比"] = df["三孩及以上"] / total_births
    return df


def load_tfr_data(base_dir: Path, tables: list[dict[str, object]] | None = None) -> dict[int, pd.DataFrame]:
    data_by_year = {}
    for cfg in tables or DEFAULT_TFR_TABLES:
        path = base_dir / "Dataset" / str(cfg["filename"])
        df = load_age_metrics(path, str(cfg["sheet"]), str(cfg["kind"]))
        data_by_year[int(cfg["year"])] = df
    return data_by_year


def _linear_extrapolate(years: np.ndarray, values: np.ndarray, target_year: int) -> float:
    slope, intercept = np.polyfit(years, values, deg=1)
    return float(slope * target_year + intercept)


def build_tfr_summary(data_by_year: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for year, df in sorted(data_by_year.items()):
        row = {
            "年份": year,
            "TFR": float(df["年龄别生育率"].sum() / 1000.0),
        }
        for col in PARITY_RATE_COLUMNS:
            row[col] = float(df[col].fillna(0).sum() / 1000.0)
        rows.append(row)
    return pd.DataFrame(rows)


def project_female_age_structure(data_by_year: dict[int, pd.DataFrame], target_years: list[int]) -> dict[int, pd.DataFrame]:
    historical = {}
    for year, df in data_by_year.items():
        historical[year] = df[["年龄", "妇女人数"]].copy()

    target_map: dict[int, pd.DataFrame] = {}
    hist_years = np.array(sorted(historical.keys()), dtype=float)
    for target_year in target_years:
        rows = []
        for age in historical[int(hist_years[0])]["年龄"]:
            values = np.array([historical[int(year)].loc[historical[int(year)]["年龄"] == age, "妇女人数"].iloc[0] for year in hist_years], dtype=float)
            projected = max(_linear_extrapolate(hist_years, values, target_year), 0.0)
            rows.append({"年龄": int(age), "妇女人数": projected})
        target_map[target_year] = pd.DataFrame(rows)
    return target_map


def forecast_tfr_simplified(data_by_year: dict[int, pd.DataFrame], target_years: tuple[int, ...] = (2030, 2040)) -> dict[str, object]:
    """
    缺少 W_k(a,t) 时，按 Todo 中“阶段七”采用简化法：
    1. 对一/二/三孩总和率做线性外推；
    2. 保持 2020 年年龄模式不变，只按总量缩放；
    3. 结合分年龄妇女人数的线性外推，计算预测出生人数。
    """
    summary = build_tfr_summary(data_by_year)
    hist_years = summary["年份"].to_numpy(dtype=float)
    base_year = int(summary["年份"].max())
    base_df = data_by_year[base_year].copy()

    projections = []
    projected_curves: dict[int, pd.DataFrame] = {}
    age_structure = project_female_age_structure(data_by_year, list(target_years))

    for target_year in target_years:
        projected = {"年份": target_year}
        projected_sum_rates = np.zeros(len(base_df), dtype=float)
        projected_df = base_df[["年龄"]].copy()
        for col in PARITY_RATE_COLUMNS:
            hist_values = summary[col].to_numpy(dtype=float)
            total_hat = max(_linear_extrapolate(hist_years, hist_values, target_year), 0.0)
            base_rates = base_df[col].fillna(0).to_numpy(dtype=float)
            base_total = max(base_rates.sum() / 1000.0, 1e-9)
            scaled_rates = base_rates * (total_hat / base_total)
            projected_df[col] = scaled_rates
            projected_sum_rates += scaled_rates
            projected[f"{col}总和率"] = total_hat

        projected_df["年龄别生育率"] = projected_sum_rates
        projected_df["妇女人数"] = age_structure[target_year]["妇女人数"].to_numpy(dtype=float)
        projected_df["预测出生人数"] = projected_df["妇女人数"] * projected_df["年龄别生育率"] / 1000.0
        projected["TFR"] = float(projected_df["年龄别生育率"].sum() / 1000.0)
        projected["预测出生人数"] = float(projected_df["预测出生人数"].sum())
        projections.append(projected)
        projected_curves[target_year] = projected_df

    return {
        "summary": summary,
        "forecast": pd.DataFrame(projections),
        "projected_curves": projected_curves,
    }


def print_tfr_summary(data_by_year: dict[int, pd.DataFrame], forecast_result: dict[str, object] | None = None) -> None:
    summary = build_tfr_summary(data_by_year)
    print("历史总和生育率(TFR)估计：")
    for _, row in summary.iterrows():
        print(
            f"- {int(row['年份'])} 年: TFR ≈ {row['TFR']:.3f} "
            f"(一孩 {row['一孩生育率']:.3f}, 二孩 {row['二孩生育率']:.3f}, 三孩及以上 {row['三孩及以上生育率']:.3f})"
        )

    if forecast_result is not None:
        print("未来 TFR 预测（按 Todo 第七阶段简化法）：")
        forecast = forecast_result["forecast"]
        for _, row in forecast.iterrows():
            print(f"- {int(row['年份'])} 年: TFR ≈ {row['TFR']:.3f}, 预测出生人数 ≈ {row['预测出生人数']:.0f}")


def plot_tfr_dashboard(data_by_year: dict[int, pd.DataFrame], forecast_result: dict[str, object] | None = None) -> None:
    summary = build_tfr_summary(data_by_year)

    plt.figure(figsize=(10, 6))
    has_asfr = False
    for year, df in sorted(data_by_year.items()):
        if "年龄别生育率" not in df.columns or df["年龄别生育率"].isna().all():
            continue
        has_asfr = True
        plt.plot(df["年龄"], df["年龄别生育率"], label=f"{year} 年", linewidth=2)
    if has_asfr:
        plt.xlabel("年龄")
        plt.ylabel("生育率")
        plt.title("年龄别生育率对比")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()
    else:
        print("未提取到可用于绘图的年龄别生育率数据。")

    plt.figure(figsize=(10, 6))
    plt.plot(summary["年份"], summary["TFR"], marker="o", linewidth=2, label="历史 TFR")
    if forecast_result is not None:
        forecast = forecast_result["forecast"]
        plt.plot(forecast["年份"], forecast["TFR"], marker="o", linewidth=2, linestyle="--", label="预测 TFR")
        merged_years = pd.concat([summary["年份"], forecast["年份"]], ignore_index=True)
        merged_tfr = pd.concat([summary["TFR"], forecast["TFR"]], ignore_index=True)
        plt.plot(merged_years, merged_tfr, color="tab:gray", alpha=0.3)
    plt.xlabel("年份")
    plt.ylabel("TFR")
    plt.title("年份总和生育率曲线")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    share_cols = ["一孩占比", "二孩占比", "三孩及以上占比"]
    titles = ["一孩占比", "二孩占比", "三孩及以上占比"]

    for ax, col, title in zip(axes, share_cols, titles):
        for year, df in sorted(data_by_year.items()):
            ax.plot(df["年龄"], df[col], label=f"{year} 年", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("年龄")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("占比")
    axes[-1].legend(loc="best")
    fig.suptitle("分年龄孩次占比对比", y=1.02)
    fig.tight_layout()
    plt.show()

    if forecast_result is not None:
        projected_curves = forecast_result["projected_curves"]
        plt.figure(figsize=(10, 6))
        plt.plot(data_by_year[2020]["年龄"], data_by_year[2020]["年龄别生育率"], linewidth=2, label="2020 年")
        for year, df in sorted(projected_curves.items()):
            plt.plot(df["年龄"], df["年龄别生育率"], linewidth=2, linestyle="--", label=f"{year} 年预测")
        plt.xlabel("年龄")
        plt.ylabel("生育率")
        plt.title("未来年龄别生育率预测")
        plt.grid(alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.show()
