import pandas as pd


def fillcopy(df, x=0.0):

    match type(df):
        case pd.DataFrame:
            return df.copy(deep=True).apply(lambda y: x, result_type="broadcast")
        case pd.Series:
            return df.copy(deep=True).apply(lambda y: x)

    return None
