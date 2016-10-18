from keras.layers.core import Dense, Permute
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.optimizers import Adam

def create_lstm_network(num_timesteps, num_frequency_dimensions, num_hidden_dimensions=512, num_recurrent_units=2):
	model = Sequential()
	#This layer converts frequency space to hidden space
	model.add(TimeDistributed(Dense(num_hidden_dimensions), input_shape=(num_timesteps, num_frequency_dimensions)))
	for cur_unit in xrange(num_recurrent_units):
          model.add(LSTM(input_dim=num_hidden_dimensions, output_dim=num_hidden_dimensions, return_sequences=True))
        # Swap time & freq dims
        model.add(Permute((2, 1)))
	for cur_unit in xrange(num_recurrent_units):
	  model.add(LSTM(input_dim=num_timesteps, output_dim=num_timesteps, return_sequences=True))
        # Swap back
        model.add(Permute((2, 1)))
	#This layer converts hidden space back to frequency space
	model.add(TimeDistributed(Dense(input_dim=num_hidden_dimensions, output_dim=num_frequency_dimensions)))
        # Note Adam contrains network size due to OOM, rmsprop doesn't
	model.compile(loss='mean_squared_error', optimizer=Adam())
	return model

