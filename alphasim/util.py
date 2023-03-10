import numpy as np
import pandas as pd


def like(
    source: pd.DataFrame | pd.Series | np.ndarray, fill_value: float = 0
) -> pd.DataFrame | pd.Series | np.ndarray:

    match type(source):
        case pd.DataFrame:
            source = pd.DataFrame(source)
            copied = pd.DataFrame(np.zeros(source.shape, dtype=np.float64))
            copied.index = source.index
            copied.columns = source.columns
        case pd.Series:
            source = pd.Series(source)
            copied = pd.Series(np.zeros(source.shape, dtype=np.float64))
            copied.index = source.index
        case np.ndarray:
            copied = np.zeros(source.shape, dtype=np.float64)
        case _:
            raise ValueError("unknown source type")

    copied[:] = fill_value

    return copied
