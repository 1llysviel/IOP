from __future__ import annotations

from pathlib import Path

from utils.asp_tools import load_asp_history, plot_asp_dashboard, print_asp_summary
from utils.demography import configure_matplotlib_cn


BASE_DIR = Path(__file__).resolve().parent
configure_matplotlib_cn()


def main() -> None:
    history = load_asp_history(BASE_DIR)
    populations = history["populations"]
    survival = history["survival"]

    print_asp_summary(history)
    plot_asp_dashboard(populations[2000], populations[2010], populations[2020], survival[(2000, 2010)], survival[(2010, 2020)])


if __name__ == "__main__":
    main()
