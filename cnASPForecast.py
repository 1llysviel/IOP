from __future__ import annotations

from pathlib import Path

from utils.asp_tools import forecast_asp_cohort_shift, load_asp_history, plot_asp_dashboard, print_asp_summary
from utils.demography import configure_matplotlib_cn


BASE_DIR = Path(__file__).resolve().parent
configure_matplotlib_cn()


def main() -> None:
    history = load_asp_history(BASE_DIR)
    forecast_result = forecast_asp_cohort_shift(history, target_years=(2030, 2040))

    populations = history["populations"]
    survival = history["survival"]

    print_asp_summary(history, forecast_result=forecast_result)
    print("说明：未来预测中，25-49岁按历史平均10年存活率做队列推移，15-24岁入口队列按历史线性趋势补入。")
    plot_asp_dashboard(
        populations[2000],
        populations[2010],
        populations[2020],
        survival[(2000, 2010)],
        survival[(2010, 2020)],
        forecast_result=forecast_result,
    )


if __name__ == "__main__":
    main()
