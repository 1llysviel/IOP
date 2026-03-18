from __future__ import annotations

import re

import matplotlib.pyplot as plt
import pandas as pd


def configure_matplotlib_cn() -> None:
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False


def strip_spaces(value: object) -> str:
    return str(value).replace(" ", "").strip()


def parse_single_age(value: object, min_age: int = 15, max_age: int = 49) -> int | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    matched = re.match(r"^(\d{2})\s*岁?$", text)
    if not matched:
        return None
    age = int(matched.group(1))
    if min_age <= age <= max_age:
        return age
    return None
