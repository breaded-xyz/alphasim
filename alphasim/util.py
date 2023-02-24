import numpy as np
import pandas as pd

def like(source):

    match type(source):
        case pd.DataFrame:
            copied = pd.DataFrame(np.zeros(source.shape, dtype=np.float64))
            copied.index = source.index
            copied.columns = source.columns
        case pd.Series:
            copied = pd.Series(np.zeros(source.shape, dtype=np.float64))
            copied.index = source.index
        case _:
            copied = np.zeros(source.shape, dtype=np.float64)

    return copied

def norm(x: pd.Series) -> pd.Series:
    weights = x.abs()
    weights = x / x.sum()
    return np.copysign(weights, x)