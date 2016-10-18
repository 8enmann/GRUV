from __future__ import absolute_import
from __future__ import print_function
from data_utils.parse_files import *
from keras.models import model_from_json
import config.nn_config as nn_config
import gen_utils.seed_generator as seed_generator
import gen_utils.sequence_generator as sequence_generator
import nn_utils.network_utils as network_utils
import numpy as np
import os

def main(model_name, output_name):
  config = nn_config.get_neural_net_configuration()
  sample_frequency = config['sampling_frequency']
  inputFile = config['model_file']
  model_basename = config['model_basename']

  # Load up the training data
  print ('Loading training data')
  # X_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
  # y_train is a tensor of size (num_train_examples, num_timesteps, num_frequency_dims)
  # X_mean is a matrix of size (num_frequency_dims,) containing the mean for each frequency dimension
  # X_var is a matrix of size (num_frequency_dims,) containing the variance for each frequency dimension
  X_train = np.load(inputFile + '_x.npy')
  y_train = np.load(inputFile + '_y.npy')
  X_mean = np.load(inputFile + '_mean.npy')
  X_var = np.load(inputFile + '_var.npy')
  print ('Finished loading training data')

  # Load existing weights if available
  if os.path.isfile(model_name):
    with open(model_name) as f:
      model = model_from_json(f.read())
      model.load_weights(model_filename.replace('json', 'hdf5'))
  else:
    print('Model filename ' + model_filename + ' could not be found!')

  print ('Starting generation!')
  # Here's the interesting part
  # We need to create some seed sequence for the algorithm to start with
  # Currently, we just grab an existing seed sequence from our training data and use that
  # However, this will generally produce verbatum copies of the original songs
  # In a sense, choosing good seed sequences = how you get interesting compositions
  # There are many, many ways we can pick these seed sequences such as taking linear combinations of certain songs
  # We could even provide a uniformly random sequence, but that is highly unlikely to produce good results
  seed_len = 1
  seed_seq = seed_generator.generate_copy_seed_sequence(seed_length=seed_len, training_data=X_train)

  max_seq_len = 10; # Defines how long the final song is. Total song length in samples = max_seq_len * example_len
  output = sequence_generator.generate_from_seed(model=model, seed=seed_seq, 
    sequence_length=max_seq_len, data_variance=X_var, data_mean=X_mean)
  print ('Finished generation!')

  # Save the generated sequence to a WAV file
  save_generated_example(output_filename, output, sample_frequency=sample_frequency)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default=None)
parser.add_argument('--output', default='./generated.wav')
args = parser.parse_args()

if __name__ == '__main__':
  main(model_name=args.model_name,
       output_name=args.output,
  )

