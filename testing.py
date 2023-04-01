import pandas as pd
import csv
from lstm.univariate import univariate, split_sequence_univariate


def main():
    file = "data/weekly.csv" # location of data
    df = pd.read_csv(file)

    # choose a window and a number of time steps
    seq_size, n_steps = 52, 4
    # choose a batch size and a number of epochs
    batch_size, num_epochs = 50, 200

    # organize data
    raw_seq = df["vix"].tolist()
    log_ret = df["s&p(log)"].tolist()[seq_size:]
    week = df["Date"].tolist()[seq_size:]
    default_log_ret, lstm_log_ret = [], []
    default_portfolio, lstm_portfolio = 0, 0
    correct, incorrect = 0, 0

    # split into samples and run model
    with open('testing.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Week", "Date", "LSTM", "Passive", "Correct"])
        for i in range(len(raw_seq)-seq_size):
            inputs, outputs = split_sequence_univariate(raw_seq[i:seq_size+i], n_steps)
            # prediction = raw_seq[i:seq_size+i+1][-1]
            prediction = univariate(inputs, outputs, raw_seq[i:seq_size+i], n_steps, batch_size, num_epochs, type="vanilla")
            default_log_ret.append(log_ret[i])
            default_portfolio = 2.718281828459**sum(default_log_ret)
            # print(f"Prev: {raw_seq[i:seq_size+i][-1]}, Prediction: {prediction}, Actual: {raw_seq[i:seq_size+i+1][-1]}")
            if prediction < raw_seq[i:seq_size+i][-1]:
                lstm_log_ret.append(log_ret[i])
                lstm_portfolio = 2.718281828459**sum(lstm_log_ret)
                print(f"Buy on ({seq_size+i+2}) {week[i]} => {round(lstm_portfolio, 2)} vs {round(default_portfolio, 2)}", end=" ")
                if raw_seq[i:seq_size+i+1][-1] < raw_seq[i:seq_size+i][-1]: correct += 1; print(f"(Correct x{correct}, {round(correct*100/(correct+incorrect), 1)}%)")
                else: incorrect += 1; print(f"(Incorrect x{incorrect}, {round(incorrect*100/(correct+incorrect), 1)}%)")
            else:
                lstm_log_ret.append(0)
                lstm_portfolio = 2.718281828459**sum(lstm_log_ret)
                print(f"Sell on ({seq_size+i+2}) {week[i]} => {round(lstm_portfolio, 2)} vs {round(default_portfolio, 2)}", end=" ")
                if raw_seq[i:seq_size+i+1][-1] > raw_seq[i:seq_size+i][-1]: correct += 1; print(f"(Correct x{correct}, {round(correct*100/(correct+incorrect), 1)}%)")
                else: incorrect += 1; print(f"(Incorrect x{incorrect}, {round(incorrect*100/(correct+incorrect), 1)}%)")
            writer.writerow([seq_size+i+2, week[i], round(lstm_portfolio, 3), round(default_portfolio, 3), round(correct/(correct+incorrect), 5)])
        

if __name__ == "__main__":
    main()