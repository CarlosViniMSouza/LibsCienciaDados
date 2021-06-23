import numpy as np
import pandas as pd

var = pd.Series(['C', "py", 1, 2.5, True, (1, -1, 0), [1.5, -1.5, 0.5, -0.5], {"key1": 'Param1'}])
print(var)
# create a default integer index

days = pd.date_range('20010108', periods=7)
print(days)
# a datetime index and labeled columns

df1 = pd.DataFrame(np.random.randn(7, 4), index=days, columns=list("WXYZ"))
print(df1)
# passing a dict of objects that can be converted to series-like.

df2 = pd.DataFrame(
    {
        "A": 1.0,
        "B": pd.Timestamp("20210622"),
        "C": pd.Series(1, index=list(range(4)), dtype="float64"),
        "D": np.array([10] * 4, dtype="int64"),
        "E": pd.Categorical(["test", "train", "production", "developing"]),
        "F": "P.O.O."
    }) # our dictionary is ready fo analysis
print(df2)
# a dict of objects that can be converted to series-like

print(df2.dtypes) # The columns of the resulting DataFrame have different dtypes.
