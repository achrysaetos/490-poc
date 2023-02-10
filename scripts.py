import numpy as np
import pandas as pd
import os


"""
Calculate log returns of each company as a new column in the csv.
"""
def calc_log_return(files):
    for file in files:
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

def calc_covariance(files):
    companies = []
    for file in files:
        df = pd.read_csv(file)
        log_returns = df["Log Return"].tolist()
        returns = [2.718281828459**lr-1 for lr in log_returns]
        companies.append(returns)
    for i in range(len(companies)):
        for j in range(len(companies[i])):
            companies[i][j] *= 100
    for i in range(len(companies)-1):
        for j in range(i+1, len(companies)):
            cov = np.cov(companies[i][-52:], companies[j][-52:])[0][1]
            if cov < -1: print(cov, files[i], files[j])


directory = "s&p500" # folder containing data
files = []
for filename in os.scandir(directory):
    if filename.is_file():
        files.append(filename.path)

# calc_log_return(files)
calc_covariance(files)