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
    file = "combined.csv"

    # according to dollar cost averaging
    default_portfolio, lstm_portfolio = 0, 0
    correct, incorrect = 0, 0

    # choose a window and a number of time steps
    seq_size, n_steps = 52, 4
    # choose a batch size and a number of epochs
    batch_size, num_epochs = 50, 200
    # define input sequence
    df = pd.read_csv(file)
    raw_seq = df["vix"].tolist()
    log_ret = df["s&p(log)"].tolist()
    data = log_ret[seq_size:]
    week = df["Date"].tolist()[seq_size:]
    default_log_ret, lstm_log_ret = [], []
    # split into samples and run model
    for i in range(len(raw_seq)-seq_size):
        inputs, outputs = split_sequence_univariate(raw_seq[i:seq_size+i], n_steps)
        # prediction = raw_seq[i:seq_size+i+1][-1]
        prediction = univariate(inputs, outputs, raw_seq[i:seq_size+i], n_steps, batch_size, num_epochs, type="vanilla")
        default_log_ret.append(data[i])
        default_portfolio = 2.718281828459**sum(default_log_ret)
        # print(f"Prev: {raw_seq[i:seq_size+i][-1]}, Prediction: {prediction}, Actual: {raw_seq[i:seq_size+i+1][-1]}")
        # print(f"Week: {week[i]}, Data: {data[i]}, Focus: {raw_seq[i:seq_size+i+1][-1]}")
        if prediction < raw_seq[i:seq_size+i][-1]:
            lstm_log_ret.append(data[i])
            lstm_portfolio = 2.718281828459**sum(lstm_log_ret)
            print(f"Buy on {week[i]} => {round(lstm_portfolio, 2)} vs {round(default_portfolio, 2)}", end=" ")
            if raw_seq[i:seq_size+i+1][-1] < raw_seq[i:seq_size+i][-1]: correct += 1; print(f"(Correct x{correct}, {round(correct*100/(correct+incorrect), 1)}%)")
            else: incorrect += 1; print(f"(Incorrect x{incorrect}, {round(incorrect*100/(correct+incorrect), 1)}%)")
        else:
            lstm_log_ret.append(0)
            lstm_portfolio = 2.718281828459**sum(lstm_log_ret)
            print(f"Sell on {week[i]} => {round(lstm_portfolio, 2)} vs {round(default_portfolio, 2)}", end=" ")
            if raw_seq[i:seq_size+i+1][-1] > raw_seq[i:seq_size+i][-1]: correct += 1; print(f"(Correct x{correct}, {round(correct*100/(correct+incorrect), 1)}%)")
            else: incorrect += 1; print(f"(Incorrect x{incorrect}, {round(incorrect*100/(correct+incorrect), 1)}%)")
        

if __name__ == "__main__":
    main()