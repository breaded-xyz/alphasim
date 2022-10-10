# Alphasim

Alphasim is a minimalist backtester inspired by vthe excellent <https://github.com/Robot-Wealth/rsims>.

The backtest inputs are the natural outputs of a typical quant research process: dataframe's of asset prices and weights.

Trade costs are optimised using a 'trade buffer' heuristic (see the rsims repo).

See the notebook in the vignettes directory for an example backtest of a risk premia harvesting strategy.

If you're looking for a more traditional backtester and algo development platform (in Go) check out my other project <https://github.com/thecolngroup/alphakit>.
