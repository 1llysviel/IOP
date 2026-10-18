import numpy as np
import pandas as pd
import pytest

from iop import census, fertility, graduation, leecarter, lifetable, wpp


def test_split_groups_preserves_totals_and_is_nonnegative():
    starts = np.arange(0, 105, 5)
    counts = np.r_[np.linspace(100, 60, 20), 5.0]
    single = graduation.split_groups(starts, counts, open_start=100)
    assert (single >= 0).all()
    for s, c in zip(starts[:-1], counts[:-1]):
        assert single.loc[s : s + 4].sum() == pytest.approx(c)
    assert single.loc[100] == pytest.approx(5.0)


def test_life_table_constant_hazard():
    m = pd.Series(0.02, index=range(0, 111))
    lt = lifetable.life_table(m, "female")
    assert (np.diff(lt["lx"]) <= 0).all()
    assert lt["ex"].iloc[0] == pytest.approx(1 / 0.02, rel=0.02)


@pytest.mark.parametrize("sex,col", [("male", "LExMale"), ("female", "LExFemale")])
def test_life_table_reproduces_wpp_e0(sex, col):
    m = wpp.mortality_rates()[sex][2019]
    e0 = lifetable.e0(lifetable.kannisto_close(m), sex)
    assert e0 == pytest.approx(wpp.indicators().loc[2019, col], abs=0.15)


def test_lee_carter_recovers_rank_one_structure():
    ages, years = np.arange(0, 90), np.arange(1990, 2020)
    a = -9 + 0.09 * ages
    b = np.full(len(ages), 1 / len(ages))
    k = -2.0 * (years - years.mean())
    log_m = pd.DataFrame(a[:, None] + b[:, None] * k[None, :], index=ages, columns=years)
    model = leecarter.fit(log_m)
    assert model.explained > 0.999
    assert model.drift == pytest.approx(-2.0, rel=1e-6)


def test_hadwiger_schedule_hits_targets():
    s = fertility.schedule(1.2, 30.5, 3.9)
    assert s.sum() == pytest.approx(1.2)
    assert fertility.mean_age(s.index.to_numpy() + 0.5, s.to_numpy()) == pytest.approx(30.5, abs=1e-6)


def test_tfr_ar1_recovers_parameters():
    rng = np.random.default_rng(1)
    mu, rho, sigma = np.log(1.4), 0.8, 0.03
    y = [np.log(1.0)]
    for _ in range(200):
        y.append(mu + rho * (y[-1] - mu) + rng.normal(0, sigma))
    y = np.array(y)
    series = pd.Series(np.exp(y), index=range(1800, 2001))
    post = fertility.fit_tfr_ar1(series, fertility.TfrPrior(median=1.3, log_sd=0.5))
    m_rho, m_mu = post.marginal_rho(), post.marginal_mu()
    # 弱先验 + 长序列时，后验均值应与 OLS 估计几乎一致
    c, r = np.linalg.lstsq(np.c_[np.ones(len(y) - 1), y[:-1]], y[1:], rcond=None)[0]
    assert (m_rho.index * m_rho).sum() == pytest.approx(r, abs=0.02)
    assert (m_mu.index * m_mu).sum() == pytest.approx(np.exp(c / (1 - r)), abs=0.02)


def test_census_sample_fractions_and_2020_split():
    for year in census.YEARS:
        assert 0.08 < census.sample_fraction(year) < 0.12
    groups = census.population_groups(2020)
    pop = census.population(2020)
    assert pop["male"].sum() == pytest.approx(groups["male"].sum())
    assert (pop >= 0).all().all()
