"""把分组数据拆成单岁。

做法：在组边界上构造累计人数 N(x)，用 PCHIP（保单调的三次 Hermite 样条）插值，
再对整数年龄差分。累计量单调 ⇒ 每个单岁都非负，且各组合计严格保持不变。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


def split_groups(starts: np.ndarray, counts: np.ndarray, open_start: int | None = None) -> pd.Series:
    """starts: 各组起始年龄（升序，末组为开放组时其起点 = open_start）；counts: 各组人数。

    返回以单岁为索引的 Series；开放组原样保留在 open_start 上。
    """
    starts = np.asarray(starts, dtype=int)
    counts = np.asarray(counts, dtype=float)
    if open_start is not None:
        closed = starts < open_start
        open_count = counts[~closed].sum()
        starts, counts = starts[closed], counts[closed]
        end = open_start
    else:
        open_count = None
        end = starts[-1] + (starts[-1] - starts[-2])

    edges = np.append(starts, end)
    cum = np.concatenate([[0.0], np.cumsum(counts)])
    spline = PchipInterpolator(edges, cum)
    ages = np.arange(edges[0], end + 1)
    single = np.diff(spline(ages))
    out = pd.Series(single, index=ages[:-1], dtype=float)
    if open_count is not None:
        out.loc[open_start] = open_count
    out.index.name = "age"
    return out


def interpolate_log_rates(starts: np.ndarray, widths: np.ndarray, rates: np.ndarray, max_age: int) -> pd.Series:
    """把分组死亡率 m 插值到单岁：在组中点上对 log m 做 PCHIP。

    0 岁与 1-4 岁这类宽度不同的组也适用；单岁组直接取原值。
    """
    starts = np.asarray(starts, dtype=float)
    widths = np.asarray(widths, dtype=float)
    rates = np.asarray(rates, dtype=float)
    mids = starts + widths / 2.0
    spline = PchipInterpolator(mids, np.log(rates), extrapolate=True)
    ages = np.arange(0, max_age + 1)
    out = pd.Series(np.exp(spline(ages + 0.5)), index=ages)
    for s, w, r in zip(starts, widths, rates):
        if w == 1:
            out.loc[int(s)] = r
    out.index.name = "age"
    return out
