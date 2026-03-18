from __future__ import annotations

from pathlib import Path

from utils.asp_tools import (
    cohort_survival,
    estimate_population,
    load_2000_population_and_mortality,
    load_national_female_deaths,
    plot_asp_dashboard,
)
from utils.demography import configure_matplotlib_cn


BASE_DIR = Path(__file__).resolve().parent
configure_matplotlib_cn()


def main() -> None:
    f2000 = BASE_DIR / "Dataset/五-t0604.xls"
    f2010 = BASE_DIR / "Dataset/六-A0601.xls"
    f2020 = BASE_DIR / "Dataset/七-A0601.xls"

    base_2000 = load_2000_population_and_mortality(f2000)
    d2010 = load_national_female_deaths(f2010)
    d2020 = load_national_female_deaths(f2020)

    p2010 = estimate_population(d2010, base_2000)
    p2020 = estimate_population(d2020, base_2000)

    pop_2000 = base_2000.set_index("年龄组")["女性人口"]
    pop_2010 = p2010.set_index("年龄组")["女性人口估计"]
    pop_2020 = p2020.set_index("年龄组")["女性人口估计"]

    s0010 = cohort_survival(pop_2000, pop_2010)
    s1020 = cohort_survival(pop_2010, pop_2020)

    print("说明：2010/2020 年女性人口由“女性死亡人数 / 2000年同年龄女性死亡率”反推得到。")
    print("2000→2010 存活率（15-19→25-29 ... 35-39→45-49）：")
    print(s0010.to_string(index=False))
    print(f"平均存活率: {s0010['存活率'].mean():.4f}")
    print()
    print("2010→2020 存活率（15-19→25-29 ... 35-39→45-49）：")
    print(s1020.to_string(index=False))
    print(f"平均存活率: {s1020['存活率'].mean():.4f}")

    plot_asp_dashboard(pop_2000, pop_2010, pop_2020, s0010, s1020)


if __name__ == "__main__":
    main()
