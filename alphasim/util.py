def fillcopy(df, x=0.0):
    return df.copy(deep=True).apply(lambda y: x, result_type="broadcast")
