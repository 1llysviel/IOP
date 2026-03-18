import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from cnstats.stats import stats

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]


def fetch_stats(zbcode, datestr, dbcode="hgnd"):
    return stats(zbcode=zbcode, datestr=datestr, dbcode=dbcode, as_df=True)


def prepare_df(df):
    if df is None:
        return df
    df = df.copy()
    if "查询日期" in df.columns:
        df["查询日期"] = pd.to_datetime(df["查询日期"])
        df = df.sort_values("查询日期").set_index("查询日期")
    return df


def numeric_cols(df):
    if df is None:
        return []
    return df.select_dtypes(include=["number"]).columns.tolist()


def plot_population(
    dfs,
    labels,
    styles=None,
    figsize=(10, 6),
    title="人口趋势",
    mark_negative=True,
    negative_marker="o",
    negative_color="red",
    fill_negative=True,
    fill_alpha=0.2,
    return_ax=False,
):
    plt.figure(figsize=figsize)
    ax = plt.gca()
    styles = styles or [None] * len(dfs)
    for df, label, style in zip(dfs, labels, styles):
        if df is None:
            continue
        for col in numeric_cols(df):
            series = df[col]
            ax.plot(df.index, series, label=f"{label} - {col}", linestyle=style)
            if mark_negative:
                mask = series < 0
                if mask.any():
                    ax.scatter(df.index[mask], series[mask], color=negative_color, marker=negative_marker, s=20, zorder=5)
            if fill_negative:
                ax.fill_between(df.index, series, 0, where=(series < 0), interpolate=True, color=negative_color, alpha=fill_alpha)
    ax.xaxis.set_major_locator(mdates.YearLocator(base=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.legend()
    plt.xlabel("时间")
    plt.ylabel("数值")
    plt.title(title)
    plt.gcf().autofmt_xdate(rotation=45)
    plt.tight_layout()
    fig = plt.gcf()
    if return_ax:
        return fig, ax
    return fig
