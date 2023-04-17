import numpy as np
import pandas as pd

from alphasim.portfolio import _discretize, allocate, distribute_longshort
from alphasim.util import like


def test_distribute():
    norm_wts = pd.Series(
        {
            "BNBUSDT": -0.008050777797295872,
            "BTCUSDT": 0.06898261754809881,
            "ETHUSDT": 0.09226318314358928,
            "ADAUSDT": -0.08615190821239778,
            "FOOTOKEN": None,
            "XRPUSDT": -0.09277929793210976,
            "LINKUSD": 0.21050327180819334,
            "MATICUSDT": 0.08260193011449973,
            "DOGEUSDT": -0.23104263169596148,
            "CHZUSDT": -0.09942534056084149,
            "SOLUSDT": 0.028199041187012476,
            "BARTOKEN": None,
        }
    )

    assert norm_wts.abs().sum() == 1

    x = distribute_longshort(norm_wts, max=0.2)

    print(x.round(4))

    assert x.max() <= 0.2


def test_allocate():
    capital = 1000
    prices = pd.Series({"FOO": 100, "BAR": 100})
    lots = pd.Series({"FOO": 10, "BAR": 10})
    port = pd.Series({"FOO": 200, "BAR": -200})
    weights = pd.Series({"FOO": 0.2, "BAR": -0.4})
    tb = 0

    rebal = allocate(capital, prices, port, weights, tb, lots)
    (_, _, _, adj_delta_weight, base_qty, quote_qty) = rebal
    print(adj_delta_weight, base_qty, quote_qty)

    assert np.array_equal(adj_delta_weight.sort_index(), [-0.2, 0])
    assert np.array_equal(base_qty.sort_index(), [-2, 0])
    assert np.array_equal(quote_qty.sort_index(), [-200, 0])


def test_allocate_shortratio():
    # To reduce the relative capital allocated to short positions
    # we can specifiy a param that expresses the ideal long/short ratio.
    # Target weights will be adjusted according to the short ratio.

    # Portfolio configured with half size short position
    capital = 1000
    prices = pd.Series({"FOO": 100, "BAR": 100})
    port = pd.Series({"FOO": 200, "BAR": -100})

    # Target weights are not 'short ratio' aware but
    # will be adjusted during allocation according the short_f param
    weights = pd.Series({"FOO": 0.2, "BAR": -0.4})

    # 0.5 = half the capital allocated to shorts vs longs
    short_f = 0.5

    rebal = allocate(
        capital,
        prices,
        port,
        weights,
        trade_buffer=0,
        lot_size=like(port, 10),
        short_f=short_f,
    )
    (_, _, _, adj_delta_weight, base_qty, quote_qty) = rebal
    print(adj_delta_weight, base_qty, quote_qty)

    # Start weight: BAR: -0.1 FOO: 0.2
    # Short adj target weight: BAR: -0.2 FOO: 0.2
    # Expect BAR short position rebalance to be increased by 1 share
    assert np.array_equal(adj_delta_weight.sort_index(), [-0.1, 0])
    assert np.array_equal(base_qty.sort_index(), [-1, 0])
    assert np.array_equal(quote_qty.sort_index(), [-100, 0])


def test_allocate_shortratio_initial():
    capital = 1000
    prices = pd.Series({"BAR": 100, "FOO": 100})
    port = pd.Series({"BAR": 0, "FOO": 0})

    weights = pd.Series({"BAR": -0.4, "FOO": 0.2})

    short_f = 0.5

    rebal = allocate(
        capital,
        prices,
        port,
        weights,
        trade_buffer=0,
        lot_size=like(port, 10),
        short_f=short_f,
    )
    (_, _, _, adj_delta_weight, base_qty, quote_qty) = rebal
    print(adj_delta_weight, base_qty, quote_qty)

    assert np.array_equal(adj_delta_weight.sort_index(), [-0.2, 0.2])
    assert np.array_equal(base_qty.sort_index(), [-2, 2])
    assert np.array_equal(quote_qty.sort_index(), [-200, 200])


def test_allocate_tradebuffer_and_shortf():
    capital = 1000
    prices = pd.Series({"BAR": 100, "FOO": 100})
    port = pd.Series({"BAR": 0, "FOO": 0})
    weights = pd.Series({"BAR": -0.4, "FOO": 0.2})

    short_f = 0.5
    trade_buffer = 0.05

    rebal = allocate(
        capital,
        prices,
        port,
        weights,
        trade_buffer=trade_buffer,
        lot_size=like(port, 10),
        short_f=short_f,
    )
    (_, _, _, adj_delta_weight, base_qty, quote_qty) = rebal
    print(adj_delta_weight, base_qty, quote_qty)

    assert np.array_equal(adj_delta_weight.round(3).sort_index(), [-0.175, 0.15])
    assert np.array_equal(base_qty.sort_index(), [-1.8, 1.5])
    assert np.array_equal(quote_qty.sort_index(), [-180, 150])


def test_discretize():
    capital = 1000.003
    lots = pd.Series({"FOO": 1, "BAR": 1})
    weights = pd.Series({"FOO": np.NaN, "BAR": 0.4})

    print(_discretize(capital, weights, lots))
