import numpy as np
import pandas as pd
import os
from lstm.univariate import univariate, split_sequence_univariate


def calc_return(file, start, end): # dates of investing, inclusive
    df = pd.read_csv(file)
    tot = 0
    for i in range(start+1, end):
        tot += df.loc[i, "Log Return"]
    X = 2.718281828459**tot
    return X


def main():
    directory = "s&p500" # folder containing data
    files = []
    for filename in os.scandir(directory):
        if filename.is_file():
            files.append(filename.path)

    invested, default_portfolio, lstm_portfolio = 0, 0, 0
    correct, incorrect = 0, 0

    # choose a window and a number of time steps
    seq_size, n_steps = 52, 4
    # choose a batch size and a number of epochs
    batch_size, num_epochs = 50, 200
    # define input sequence
    df = pd.read_csv("s&p500/AAPL.csv")
    raw_seq = df["Adj Close"].tolist()
    # split into samples and run model
    for i in range(len(raw_seq)-seq_size):
        inputs, outputs = split_sequence_univariate(raw_seq[i:seq_size+i], n_steps)
        prediction = univariate(inputs, outputs, raw_seq[i:seq_size+i], n_steps, batch_size, num_epochs, type="vanilla")
        lstm_portfolio *= 1 + (raw_seq[i:seq_size+i][-1] - raw_seq[i:seq_size+i][-2]) / raw_seq[i:seq_size+i][-2]
        if prediction > raw_seq[i:seq_size+i][-1]:
            invested += 100
            lstm_portfolio += 100
            default_portfolio = invested * calc_return("s&p500/AAPL.csv", 52, seq_size+i)
            print(f"Invest 100 on week {seq_size+i}({seq_size+i+2}) => ${round(lstm_portfolio, 2)} vs ${round(default_portfolio, 2)} from ${invested}", end=" ")
            if raw_seq[i:seq_size+i+1][-1] > raw_seq[i:seq_size+i][-1]: correct += 1; print(f"(Correct x{correct})")
            else: incorrect += 1; print(f"(Incorrect x{incorrect})")
        else:
            default_portfolio = invested * calc_return("s&p500/AAPL.csv", 52, seq_size+i)
            print(f"Skip week {seq_size+i}({seq_size+i+2}) => ${round(lstm_portfolio, 2)} vs ${round(default_portfolio, 2)} from ${invested}", end=" ")
            if raw_seq[i:seq_size+i+1][-1] < raw_seq[i:seq_size+i][-1]: correct += 1; print(f"(Correct x{correct})")
            else: incorrect += 1; print(f"(Incorrect x{incorrect})")
        

if __name__ == "__main__":
    main()