import pandas as pd

"""
The purpose of this method is to subset a dataframe

:param df: Dataframe being subsetted
:type df: Pandas.Dataframe
:param cols: columns that we are subsetting from the Dataframe.
:type cols: list or set.
:returns: pandas dataframe subsetted
"""

def subset(df, cols):
    df2 = pd.DataFrame()
    for name in cols:
        df2[name] = df[name]
    return(df2)
