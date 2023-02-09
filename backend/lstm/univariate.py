from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional

from prep_data import split_sequence_univariate

# choose a window and a number of time steps
seq_size, n_steps = 360, 3
# choose a batch size and a number of epochs
batch_size, num_epochs = 50, 200
# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# split into samples
inputs, outputs = split_sequence_univariate(raw_seq, n_steps)


def univariate(inputs, outputs, raw_seq, n_steps, batch_size, num_epochs, type="vanilla"):
  # reshape from [samples, timesteps] into [samples, timesteps, features]
  n_features = 1
  inputs = inputs.reshape((inputs.shape[0], inputs.shape[1], n_features))
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
  x_input = array(raw_seq[-n_steps:])
  x_input = x_input.reshape((1, n_steps, n_features))
  yhat = model.predict(x_input, verbose=0)
  return float(yhat[0][0])


""" TESTING """
print(univariate(inputs, outputs, raw_seq, n_steps, batch_size, num_epochs, type="vanilla"))
print(univariate(inputs, outputs, raw_seq, n_steps, batch_size, num_epochs, type="stacked"))
print(univariate(inputs, outputs, raw_seq, n_steps, batch_size, num_epochs, type="bidirectional"))
