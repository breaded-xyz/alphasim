import numpy as np
import pandas as pd

from alphasim.portfolio import distribute, norm_signed


def test_distribute():

    forecast = pd.Series(
        {"Acme": 100, "Foo": -20, "Bar": 30}
    )
    norm_wts = norm_signed(forecast)
    print(norm_wts)

    # Sort weights to ensure redistribution is proportial to original weight
    x = norm_wts.abs().sort_values(ascending=False)
    x[:] = distribute(x, max_weight=0.4)

    # Apply the sign of the forecast back
    x = np.copysign(x, norm_wts)
    print(x)
                          
    assert np.array_equal(x.round(3), [0.400, 0.333, -0.267])



