import numpy as np
import pandas as pd
import os


"""
Calculate log returns of each company as a new column in the csv.
"""
def calc_log_return(files):
    for file in files:
        df = pd.read_csv(file)
        df.loc[0, "Log Return"] = 0
        for i in range(1, 522):
            df.loc[i, "Log Return"] = np.log(df.loc[i, "Adj Close"]/df.loc[i-1, "Adj Close"])
        df.to_csv(file, index=False)

"""
Calculate covariances of weekly returns of each company.
"""
def calc_covariance(files):
    companies, arr = [], []
    for file in files:
        df = pd.read_csv(file)
        log_returns = df["Log Return"].tolist()
        change = [2.718281828459**lr-1 for lr in log_returns]
        companies.append(change)
    for i in range(len(companies)-1):
        for j in range(i+1, len(companies)):
            cov = np.cov(companies[i][-52:], companies[j][-52:])[0][1]
            if cov < -.0001: 
                # print(cov, files[i], files[j])
                arr.append(files[i]) # print_portfolio(files[i])
                arr.append(files[j]) # print_portfolio(files[j])
    return arr

"""
Calculate cumulative returns of given company.
"""
def calc_return(file):
    df = pd.read_csv(file)
    start, end = 0, 522 # dates of investing, inclusive
    tot = 0
    for i in range(start+1, end):
        tot += df.loc[i, "Log Return"]
    X = 2.718281828459**tot
    # print(file+": "+str(X)+"x")
    return X

def print_portfolio(files):
    for file in files:
        print("\""+file+"\"", end=", ")

def main():
    directory = "s&p500" # folder containing data
    files = []
    for filename in os.scandir(directory):
        if filename.is_file():
            files.append(filename.path)

    # calc_log_return(files)
    arr = calc_covariance(files)
    tot, count = 0, 0
    for a in arr:
        tot += calc_return(a)
        count += 1
    print(tot/count)
    

if __name__ == "__main__":
    main()