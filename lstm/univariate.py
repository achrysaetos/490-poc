from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional


def split_sequence_univariate(sequence, n_steps):
	inputs, outputs = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		inputs.append(seq_x)
		outputs.append(seq_y)
	return array(inputs), array(outputs)


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


def main():
    # choose a window and a number of time steps
    seq_size, n_steps = 360, 3
    # choose a batch size and a number of epochs
    batch_size, num_epochs = 50, 200
    # define input sequence
    raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    # split into samples
    inputs, outputs = split_sequence_univariate(raw_seq, n_steps)

    print(univariate(inputs, outputs, raw_seq, n_steps, batch_size, num_epochs, type="vanilla"))
    print(univariate(inputs, outputs, raw_seq, n_steps, batch_size, num_epochs, type="stacked"))
    print(univariate(inputs, outputs, raw_seq, n_steps, batch_size, num_epochs, type="bidirectional"))
    

if __name__ == "__main__":
    main()
