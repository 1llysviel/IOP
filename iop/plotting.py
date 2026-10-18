"""统一的绘图样式：中文字体、Okabe–Ito 色盲友好配色、预测扇形图。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Okabe & Ito (2008) 色盲友好配色
COLORS = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}
SEX_STYLE = {
    "male": {"color": COLORS["blue"], "ls": "-", "label": "男"},
    "female": {"color": COLORS["vermillion"], "ls": "--", "label": "女"},
}
YEAR_STYLE = {
    2000: {"color": COLORS["sky"], "ls": ":", "marker": "s"},
    2010: {"color": COLORS["orange"], "ls": "--", "marker": "^"},
    2020: {"color": COLORS["blue"], "ls": "-", "marker": "o"},
}


def setup() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "Heiti SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 110,
            "figure.figsize": (8, 4.2),
            "figure.constrained_layout.use": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
            "axes.prop_cycle": plt.cycler(color=[COLORS[c] for c in ("blue", "vermillion", "green", "orange", "purple", "sky", "black")]),
        }
    )


def fan(ax, fan_df: pd.DataFrame, color: str, label: str) -> None:
    """fan_df 需含 p2.5/p10/p50/p90/p97.5 列，index 为年份。"""
    x = fan_df.index.to_numpy()
    ax.fill_between(x, fan_df["p2.5"], fan_df["p97.5"], color=color, alpha=0.15, lw=0, label=f"{label} 95% 区间")
    ax.fill_between(x, fan_df["p10"], fan_df["p90"], color=color, alpha=0.30, lw=0, label=f"{label} 80% 区间")
    ax.plot(x, fan_df["p50"], color=color, lw=2, label=f"{label} 中位数")


def percentiles(paths: np.ndarray, index) -> pd.DataFrame:
    levels = (2.5, 10, 50, 90, 97.5)
    return pd.DataFrame({f"p{lv:g}": np.percentile(paths, lv, axis=0) for lv in levels}, index=index)


def pyramid(ax, pop: pd.DataFrame, title: str, scale: float = 1e6) -> None:
    ages = pop.index.to_numpy()
    ax.barh(ages, -pop["male"] / scale, height=1.0, color=COLORS["blue"], label="男")
    ax.barh(ages, pop["female"] / scale, height=1.0, color=COLORS["vermillion"], label="女")
    lim = max(pop["male"].max(), pop["female"].max()) / scale * 1.05
    ax.set_xlim(-lim, lim)
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{abs(t):g}" for t in ticks])
    ax.set(title=title, xlabel="人口（百万）", ylabel="年龄")
