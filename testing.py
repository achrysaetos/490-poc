import numpy as np
import pandas as pd
import os
from lstm.univariate import univariate, split_sequence_univariate


def calc_return(file, start, end): # dates of investing, inclusive
    df = pd.read_csv(file)
    tot = 0
    for i in range(start+1, end+1):
        tot += df.loc[i, "Log Return"]
    X = 2.718281828459**tot
    return X


def main():
    # directory = "s&p500"; file = "s&p500/AAPL.csv"
    # files = []
    # for filename in os.scandir(directory):
    #     if filename.is_file():
    #         files.append(filename.path)
    file = "s&p500.csv"

    # according to dollar cost averaging
    invested, allowance, default_portfolio, lstm_portfolio = 0, 100, 0, 0
    correct, incorrect = 0, 0

    # choose a window and a number of time steps
    seq_size, n_steps = 52, 4
    # choose a batch size and a number of epochs
    batch_size, num_epochs = 50, 200
    # define input sequence
    df = pd.read_csv(file)
    raw_seq = df["Adj Close"].tolist()
    # split into samples and run model
    for i in range(len(raw_seq)-seq_size):
        inputs, outputs = split_sequence_univariate(raw_seq[i:seq_size+i], n_steps)
        # prediction = raw_seq[i:seq_size+i+1][-1]
        prediction = univariate(inputs, outputs, raw_seq[i:seq_size+i], n_steps, batch_size, num_epochs, type="vanilla")
        invested += 100
        r = 1 + (raw_seq[i:seq_size+i][-1] - raw_seq[i:seq_size+i][-2]) / raw_seq[i:seq_size+i][-2]
        default_portfolio = default_portfolio * r + 100
        lstm_portfolio *= r
        if prediction > raw_seq[i:seq_size+i][-1]:
            lstm_portfolio += allowance
            print(f"Invest {allowance} on week {seq_size+i-1}({seq_size+i+1}) => ${round(lstm_portfolio, 2)} vs ${round(default_portfolio, 2)} from ${invested}", end=" ")
            if raw_seq[i:seq_size+i+1][-1] > raw_seq[i:seq_size+i][-1]: correct += 1; print(f"(Correct x{correct})")
            else: incorrect += 1; print(f"(Incorrect x{incorrect})")
            allowance = 100
        else:
            print(f"Skip week {seq_size+i-1}({seq_size+i+1}) => ${round(lstm_portfolio+allowance, 2)} vs ${round(default_portfolio, 2)} from ${invested}", end=" ")
            if raw_seq[i:seq_size+i+1][-1] < raw_seq[i:seq_size+i][-1]: correct += 1; print(f"(Correct x{correct})")
            else: incorrect += 1; print(f"(Incorrect x{incorrect})")
            allowance += 100
        

if __name__ == "__main__":
    main()