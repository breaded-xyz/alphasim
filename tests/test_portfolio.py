import numpy as np
import pandas as pd

from alphasim.portfolio import _discretize, allocate, distribute


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

    x = norm_wts.abs().dropna()
    x[:] = distribute(x, max_weight=0.2)

    # Reapply the sign of the forecast
    x = np.copysign(x, norm_wts)
    print(x.round(4))

    assert x.max() <= 0.2


def test_allocate():

    capital = 1000
    prices = pd.Series({"FOO": 100, "BAR": 100})
    lots = pd.Series({"FOO": 10, "BAR": 10})
    port = pd.Series({"FOO": 200, "BAR": -200})
    weights = pd.Series({"FOO": 0.2, "BAR": -0.4})

    rebal = allocate(capital, prices, port, weights, 0, lots)
    (_, _, _, adj_delta_weight, trade_size, trade_value) = rebal
    print(adj_delta_weight, trade_size, trade_value)

    assert np.array_equal(adj_delta_weight.sort_index(), [-0.2, 0])
    assert np.array_equal(trade_size.sort_index(), [-2, 0])
    assert np.array_equal(trade_value.sort_index(), [-200, 0])


def test_discretize():

    capital = 1000.003
    lots = pd.Series({"FOO": 1, "BAR": 1})
    weights = pd.Series({"FOO": np.NaN, "BAR": 0.4})

    print(_discretize(capital, weights, lots))
