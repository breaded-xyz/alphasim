import numpy as np
import pandas as pd

from alphasim.portfolio import distribute, weight


def test_distribute():

    forecast = pd.Series(
        {"Acme": 100, "Foo": -20, "Bar": 19, "Coyote": 500}
    )
    norm_wts = weight(forecast)
    print(norm_wts)

    x = norm_wts.abs()
    x[:] = distribute(x, max_weight=0.4)

    # Reapply the sign of the forecast
    x = np.copysign(x, norm_wts)
    print(x)
                          
    assert np.array_equal(x.round(3), [0.284, -0.159, 0.157, 0.400])



