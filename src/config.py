# Import various dependencies, including the relevant modules from the Perch
# repository. Note that "chirp" is the old name that the Perch team used, so any
# chirp modules imported here were installed as part of the Perch repository in
# one of the previous cells.

import collections
from etils import epath
from IPython.display import HTML
import matplotlib.pyplot as plt
from ml_collections import config_dict
import numpy as np
from scipy.io import wavfile
import tensorflow as tf
import tqdm

from chirp.inference import colab_utils
colab_utils.initialize(use_tf_gpu=True, disable_warnings=True)

from chirp import audio_utils
from chirp import config_utils
from chirp import path_utils
from chirp.inference import embed_lib
from chirp.inference import models
from chirp.inference import tf_examples
from chirp.models import metrics
from chirp.inference.search import bootstrap
from chirp.inference.search import search
from chirp.inference.search import display
from chirp.inference.classify import classify
from chirp.inference.classify import data_lib

# We should see a GPU in the list of devices, if connected to a Colab GPU.
tf.config.list_physical_devices()


# Model specific parameters: PLEASE DO NOT CHANGE THE CODE IN THIS CELL.
config = config_dict.ConfigDict()
embed_fn_config = config_dict.ConfigDict()
embed_fn_config.model_key = 'taxonomy_model_tf'
model_config = config_dict.ConfigDict()

# The size of each "chunk" of audio.
model_config.window_size_s = 5.0

# The hop size (aka model 'stride') is the offset in seconds between successive
# chunks of audio. When hop_size is equal to window size, the chunks of audio
# will not overlap at all. Choosing a smaller hop_size (a common choice is half
# of the window_size) may be useful for capturing interesting data points that
# correspond to audio on the boundary between two windows. However, a smaller
# hop size may also lead to a larger embedding dataset because each instant of
# audio is now pesent in multiple windows. As a consequence, you might need to
# "de-dupe" your matches since multiple embedded data points may borreopnd to
# the same snippet of raw audio.
model_config.hop_size_s = 5.0

# All audio in this tutorial is resampled to 32 kHz.
model_config.sample_rate = 32000

# The location of the pre-trained model.
model_config.model_path = drive_shared_data_folder + 'bird-vocalization-classifier/'

# Only write embeddings to reduce size. The Perch codebase supports serializing
# a variety of metadata along with the embeddings, but for the purposes of this
# tutorial we will not need to make use of those features.
embed_fn_config.write_embeddings = True
embed_fn_config.write_logits = False
embed_fn_config.write_separated_audio = False
embed_fn_config.write_raw_audio = False

config.embed_fn_config = embed_fn_config
embed_fn_config.model_config = model_config

# We have functionality to break large inputs up into smaller chunks;
# this is especially helpful for dealing with long files or very large datasets.
# get in touch if you think you may need this.
config.shard_len_s = -1
config.num_shards_per_file = -1

# Number of parent directories to include in the filename. This allows us to
# provess raw audio that lives in multiple directories.
config.embed_fn_config.file_id_depth = 1

# Number of TFRecord files to create.
config.tf_record_shards = 1


# Specify a glob pattern matching any number of .wav files.
# This can look like '/path/to/audio/*.wav'
# For the purposes of a quick demo, we'll work with a subset of the full
# Powdermill dataset.  This allows us to understand how the embedding step works
# without having to embed the entire dataset (which can take a while).
unlabeled_audio_pattern = drive_shared_data_folder + 'Powdermill Embeddings/Recording_4/Recording_4_Segment_2*.wav' #@param

# Specify a directory where the embeddings will be written.
embedding_output_dir = drive_output_directory + 'raw_embeddings/' #@param

config.output_dir = embedding_output_dir
config.source_file_patterns = [unlabeled_audio_pattern]

# Create output directory and write the configuration.
output_dir = epath.Path(config.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

# The path to an empty directory where the generated labeled samples will be
# placed. Each labeled sample will be placed into a subdirectory corresponding
# to the target class that we select for that sample.
labeled_data_path = drive_output_directory + 'labeled_outputs/'  #@param