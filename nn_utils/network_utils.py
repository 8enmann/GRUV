from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM, GRU
from keras.optimizers import Adam

def create_lstm_network(num_timesteps, num_frequency_dimensions, num_hidden_dimensions=512, num_recurrent_units=2):
	model = Sequential()
	#This layer converts frequency space to hidden space
	model.add(TimeDistributed(Dense(num_hidden_dimensions), input_shape=(num_timesteps, num_frequency_dimensions)))
	for cur_unit in xrange(num_recurrent_units):
		model.add(LSTM(input_dim=num_hidden_dimensions, output_dim=num_hidden_dimensions, return_sequences=True))
	#This layer converts hidden space back to frequency space
	model.add(TimeDistributed(Dense(input_dim=num_hidden_dimensions, output_dim=num_frequency_dimensions)))
        # Can't use Adam... will OOM
	model.compile(loss='mean_squared_error', optimizer=Adam())
	return model

def create_gru_network(num_frequency_dimensions, num_hidden_dimensions, num_recurrent_units=1):
	model = Sequential()
	#This layer converts frequency space to hidden space
	model.add(TimeDistributedDense(input_dim=num_frequency_dimensions, output_dim=num_hidden_dimensions))
	for cur_unit in xrange(num_recurrent_units):
		model.add(GRU(input_dim=num_hidden_dimensions, output_dim=num_hidden_dimensions, return_sequences=True))
	#This layer converts hidden space back to frequency space
	model.add(TimeDistributedDense(input_dim=num_hidden_dimensions, output_dim=num_frequency_dimensions))
	model.compile(loss='mean_squared_error', optimizer='rmsprop')
	return model
