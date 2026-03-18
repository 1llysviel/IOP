from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .demography import parse_single_age


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
            total = pd.to_numeric(pd.Series([row.iloc[2]]), errors="coerce").iloc[0]
            first = pd.to_numeric(pd.Series([row.iloc[4]]), errors="coerce").iloc[0]
            second = pd.to_numeric(pd.Series([row.iloc[6]]), errors="coerce").iloc[0]
            third = pd.to_numeric(pd.Series([row.iloc[8]]), errors="coerce").iloc[0]
            asfr = pd.to_numeric(pd.Series([row.iloc[3]]), errors="coerce").iloc[0]
        else:
            total = pd.to_numeric(pd.Series([row.iloc[1]]), errors="coerce").iloc[0]
            first = pd.to_numeric(pd.Series([row.iloc[4]]), errors="coerce").iloc[0]
            second = pd.to_numeric(pd.Series([row.iloc[7]]), errors="coerce").iloc[0]
            third = pd.to_numeric(pd.Series([row.iloc[10]]), errors="coerce").iloc[0]
            asfr = pd.NA

        if pd.isna(total) or total == 0:
            continue

        rows.append(
            {
                "年龄": age,
                "总量": float(total),
                "一孩": float(first) if pd.notna(first) else 0.0,
                "二孩": float(second) if pd.notna(second) else 0.0,
                "三孩及以上": float(third) if pd.notna(third) else 0.0,
                "年龄别生育率": asfr,
            }
        )

    df = pd.DataFrame(rows).sort_values("年龄")
    df["一孩占比"] = df["一孩"] / df["总量"]
    df["二孩占比"] = df["二孩"] / df["总量"]
    df["三孩及以上占比"] = df["三孩及以上"] / df["总量"]
    return df


def print_tfr_summary(data_by_year: dict[int, pd.DataFrame]) -> None:
    print("可计算年份的总和生育率(TFR)估计：")
    for year, df in sorted(data_by_year.items()):
        if df["年龄别生育率"].isna().all():
            print(f"- {year} 年: 无年龄别生育率列，无法直接估算。")
            continue
        tfr = df["年龄别生育率"].sum() / 1000.0
        print(f"- {year} 年: TFR ≈ {tfr:.3f}")


def plot_tfr_dashboard(data_by_year: dict[int, pd.DataFrame]) -> None:
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
