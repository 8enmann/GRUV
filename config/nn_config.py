def get_neural_net_configuration():
  return {
    'sampling_frequency': 44100,
    # Number of hidden dimensions.
    # For best results, this should be >= freq_space_dims, but most consumer GPUs can't handle large sizes
    'hidden_dimension_size': 1024,
    # The weights filename for saving/loading trained models
    'model_basename':'./YourMusicLibraryNPWeights',
    # The model filename for the training data
    'model_file': './datasets/YourMusicLibraryNP',
    # The dataset directory
    'dataset_directory': './datasets/YourMusicLibrary/',
  }

