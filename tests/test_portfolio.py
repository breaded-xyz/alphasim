import numpy as np
import pandas as pd

from alphasim.portfolio import distribute, to_weights


def test_distribute():

    norm_wts = pd.Series(
        {
        "BNBUSDT":-0.008050777797295872,
        "BTCUSDT":0.06898261754809881,
        "ETHUSDT":0.09226318314358928,
        "ADAUSDT":-0.08615190821239778,
        "XRPUSDT":-0.09277929793210976,
        "LINKUSD":0.21050327180819334,
        "MATICUSDT":0.08260193011449973,
        "DOGEUSDT":-0.23104263169596148,
        "CHZUSDT":-0.09942534056084149,
        "SOLUSDT":0.028199041187012476,
        }
    )

    assert norm_wts.abs().sum() == 1


    x = norm_wts.abs()
    x[:] = distribute(x, max_weight=0.2)

    # Reapply the sign of the forecast
    x = np.copysign(x, norm_wts)
    print(x.round(4))
                          
    assert x.max() <= 0.4