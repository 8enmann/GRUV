from __future__ import absolute_import
from __future__ import print_function
from keras.callbacks import ProgbarLogger, ModelCheckpoint, EarlyStopping, TensorBoard
import argparse
import config.nn_config as nn_config
import nn_utils.network_utils as network_utils
import numpy as np
import time
import os

def main(model_name, num_files):
  config = nn_config.get_neural_net_configuration()
  inputFile = config['model_file']

  # Load up the training data
  print ('Loading training data')
  #X_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
  #y_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
  X_train = np.load(inputFile + '_x.npy')
  y_train = np.load(inputFile + '_y.npy')
  print ('Finished loading training data')

  #Figure out how many frequencies we have in the data
  freq_space_dims = X_train.shape[2]
  num_timesteps = X_train.shape[1]

  #Creates a lstm network
  model = network_utils.create_lstm_network(num_timesteps=num_timesteps, num_frequency_dimensions=freq_space_dims)
  #You could also substitute this with a RNN or GRU
  #model = network_utils.create_gru_network()

  if model_name:
    model.load_weights(model_name)
    # TODO: verify that loaded model is compatible with new model shape
    print ('Loaded parameters from', model_name)

  time_str = time.strftime('%Y%m%d-%H%M')
  with open('./models/%s.json' % time_str, 'w') as f:
    f.write(model.to_json())

  model.summary()

  checkpointer = ModelCheckpoint(filepath='./models/%s.hdf5' % time_str,
                                 verbose=1, save_best_only=True)
  tensorboard_reporter = TensorBoard(log_dir='./tensorboard/%s' % time_str,
                                     histogram_freq=20,
                                     write_graph=True)
  history = model.fit(
    X_train, y_train,
    # Number of training examples pushed to the GPU per batch.
    # Larger batch sizes require more memory, but training will be faster
    batch_size=16,
    nb_epoch=500,
    verbose=1,
    callbacks=[checkpointer, tensorboard_reporter, ],
    validation_split=0.05)

  print ('Training complete!')
  model.save_weights('./models/weights_%s_final.hdf5' % time_str)

parser = argparse.ArgumentParser()
parser.add_argument('--numfiles', default=20)
parser.add_argument('--model_name', default=None)
args = parser.parse_args()

if __name__ == '__main__':
  main(num_files=int(args.numfiles),
       model_name=args.model_name,
  )
