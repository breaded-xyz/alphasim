from alphasim.backtest import backtest


def test_backtest():
    assert backtest(1, 2, 3, 4, 5) == 0
