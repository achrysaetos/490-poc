from numpy import array, hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional

from prep_data import split_sequences_multivariate

# choose a window and a number of time steps
seq_size, n_steps = 360, 3
# choose a batch size and a number of epochs
batch_size, num_epochs = 50, 200
# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# convert into input/output
inputs, outputs = split_sequences_multivariate(dataset, n_steps)


def multivariate(inputs, outputs, n_steps, in_seq1, in_seq2, batch_size, num_epochs, type="vanilla"):
  # the dataset knows the number of features, e.g. 2
  n_features = inputs.shape[2]
  # define model
  model = Sequential()
  if type == "vanilla":
    model.add(LSTM(batch_size, activation='relu', input_shape=(n_steps, n_features)))
  elif type == "stacked":
    model.add(LSTM(batch_size, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(batch_size, activation='relu'))
  elif type == "bidirectional":
    model.add(Bidirectional(LSTM(batch_size, activation='relu'), input_shape=(n_steps, n_features)))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mse')
  # fit model
  model.fit(inputs, outputs, epochs=num_epochs, verbose=0)
  # demonstrate prediction
  x_input = array([[a,b] for a,b in zip(in_seq1[-n_steps:],in_seq2[-n_steps:])]) # change this (don't overfit)
  x_input = x_input.reshape((1, n_steps, n_features))
  yhat = model.predict(x_input, verbose=0)
  return float(yhat[0][0])


""" TESTING """
print(multivariate(inputs, outputs, n_steps, in_seq1, in_seq2, batch_size, num_epochs, type="vanilla"))
print(multivariate(inputs, outputs, n_steps, in_seq1, in_seq2, batch_size, num_epochs, type="stacked"))
print(multivariate(inputs, outputs, n_steps, in_seq1, in_seq2, batch_size, num_epochs, type="bidirectional"))
