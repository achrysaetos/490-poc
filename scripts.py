import numpy as np
import pandas as pd
import os

"""
Calculate log returns of each company as a new column in the csv.
"""
def calc_log_return():
    directory = "s&p500" # folder containing data
    for filename in os.scandir(directory):
        if filename.is_file():
            file = filename.path
            df = pd.read_csv(file)
            print(file, end="")

            df.loc[0, "Log Return"] = 0
            for i in range(1, 522):
                df.loc[i, "Log Return"] = np.log(df.loc[i, "Adj Close"]/df.loc[i-1, "Adj Close"])

            tot = 0
            start, end = 0, 522 # dates of investing, inclusive
            for i in range(start+1, end):
                tot += df.loc[i, "Log Return"]
            X = 2.718281828459**tot
            print(": "+str(X)+"x")

            df.to_csv(file, index=False)

calc_log_return()