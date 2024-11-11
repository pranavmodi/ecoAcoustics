# services/embedding_service.py

import os
import json
import tensorflow as tf
from chirp.inference import embed_lib
from ml_collections import config_dict
from typing import Dict, Any
from chirp import audio_utils
from chirp.inference import tf_examples
from tqdm import tqdm
from etils import epath

class EmbeddingService:
    def __init__(self, model_path: str, input_dir: str = None, output_dir: str = None):
        """Initialize the embedding service with model and data paths.
        
        Args:
            model_path: Path to the downloaded bird-vocalization-classifier model
            input_dir: Directory containing input audio files
            output_dir: Base directory for outputs
        """
        self.model_path = model_path
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Create specific output directories
        self.embedding_output_dir = os.path.join(output_dir, "raw_embeddings")
        self.labeled_data_path = os.path.join(output_dir, "labeled_outputs")
        
        # Create all required directories
        for directory in [self.output_dir, self.embedding_output_dir, self.labeled_data_path]:
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
        
        # Set up audio file pattern for source infos
        self.unlabeled_audio_pattern = os.path.join(input_dir, "*.wav")
            
        self.config = self._create_config(model_path)
        self.embed_fn = None
        
        # Create source infos for audio processing
        self.source_infos = self.create_source_infos()
        
        # Save configuration to JSON
        self.save_config()
        
    def _create_config(self, model_path: str) -> config_dict.ConfigDict:
        """Create the configuration for the embedding model."""
        config = config_dict.ConfigDict()
        embed_fn_config = config_dict.ConfigDict()
        model_config = config_dict.ConfigDict()

        # Model configuration
        embed_fn_config.model_key = 'taxonomy_model_tf'
        
        # Add file_id_depth parameter
        embed_fn_config.file_id_depth = 1  # Adjust this value based on your file structure
        
        # Audio processing parameters
        model_config.window_size_s = 5.0
        model_config.hop_size_s = 5.0
        model_config.sample_rate = 32000
        model_config.model_path = model_path

        # Embedding output configuration
        embed_fn_config.write_embeddings = True
        embed_fn_config.write_logits = False
        embed_fn_config.write_separated_audio = False
        embed_fn_config.write_raw_audio = False
        embed_fn_config.model_config = model_config
        
        # Add input/output paths to config
        config.input_dir = self.input_dir
        config.output_dir = self.embedding_output_dir
        config.source_file_patterns = [self.unlabeled_audio_pattern]
        
        config.embed_fn_config = embed_fn_config
        return config

    def load_model(self):
        """Load the embedding model."""
        if self.embed_fn is None:
            self.embed_fn = embed_lib.EmbedFn(**self.config.embed_fn_config)
            self.embed_fn.setup()
            
            # Test run to ensure model is loaded correctly
            test_audio = tf.zeros([int(self.config.embed_fn_config.model_config.sample_rate * 
                                     self.config.embed_fn_config.model_config.window_size_s)])
            self.embed_fn.embedding_model.embed(test_audio)
        
        return self.embed_fn

    def generate_embedding(self, audio: tf.Tensor) -> tf.Tensor:
        """Generate embedding for audio input.
        
        Args:
            audio: Audio tensor with shape matching model requirements
                  (sample_rate * window_size_s)
        
        Returns:
            Embedding tensor
        """
        if self.embed_fn is None:
            self.load_model()
            
        embedded = self.embed_fn.embedding_model.embed(audio)
        return embedded.embeddings[:, 0, :]

    def config_to_dict(self) -> Dict[str, Any]:
        """Convert ConfigDict to regular dictionary for JSON serialization."""
        return {
            "paths": {
                "model_path": self.model_path,
                "input_dir": self.input_dir,
                "output_dir": self.output_dir,
            },
            "model_config": {
                "window_size_s": float(self.config.embed_fn_config.model_config.window_size_s),
                "hop_size_s": float(self.config.embed_fn_config.model_config.hop_size_s),
                "sample_rate": int(self.config.embed_fn_config.model_config.sample_rate),
                "model_key": self.config.embed_fn_config.model_key,
            },
            "embedding_config": {
                "write_embeddings": bool(self.config.embed_fn_config.write_embeddings),
                "write_logits": bool(self.config.embed_fn_config.write_logits),
                "write_separated_audio": bool(self.config.embed_fn_config.write_separated_audio),
                "write_raw_audio": bool(self.config.embed_fn_config.write_raw_audio),
                "file_id_depth": int(self.config.embed_fn_config.file_id_depth),
            }
        }
    
    def save_config(self):
        """Save configuration to JSON file."""
        if self.output_dir:
            config_path = os.path.join(self.output_dir, "embedding_config.json")
            with open(config_path, 'w') as f:
                json.dump(self.config_to_dict(), f, indent=2)

    def create_source_infos(self):
        """Create source info configuration for processing audio files.
        
        SourceInfos help manage how audio files are split into chunks for processing:
        - Tracks which chunks came from which original audio file
        - Manages sharding of large files
        - Keeps track of offsets and timestamps
        
        Returns:
            List of SourceInfo objects containing metadata about audio chunks
        """
        # Use the audio pattern from class initialization
        self.config.source_file_patterns = [self.unlabeled_audio_pattern]
        self.config.num_shards_per_file = 1
        self.config.shard_len_s = -1
        
        source_infos = embed_lib.create_source_infos(
            self.config.source_file_patterns,
            self.config.num_shards_per_file, 
            self.config.shard_len_s
        )
        
        print(f'Constructed {len(source_infos)} source infos')
        return source_infos

    def generate_embeddings(self):
        """Generate embeddings for all audio files in the input directory.
        
        This method:
        1. Sets up an audio iterator to process files in chunks
        2. Generates embeddings for each chunk
        3. Saves embeddings to TFRecord files
        
        Returns:
            Tuple[int, int]: Count of (successful, failed) embedding generations
        """
        # Set minimum audio length requirement
        self.embed_fn.min_audio_s = 1.0
        
        # Create output path for embeddings
        output_dir = epath.Path(self.embedding_output_dir)
        record_file = (output_dir / 'embeddings.tfrecord').as_posix()
        
        # Initialize counters
        successful_embeddings = 0
        failed_embeddings = 0
        
        # Create audio iterator for processing files in chunks
        audio_iterator = audio_utils.multi_load_audio_window(
            filepaths=[s.filepath for s in self.source_infos],
            offsets=[s.shard_num * s.shard_len_s for s in self.source_infos],
            sample_rate=self.config.embed_fn_config.model_config.sample_rate,
            window_size_s=self.config.shard_len_s,
        )

        # Write embeddings to TFRecord files
        with tf_examples.EmbeddingsTFRecordMultiWriter(
            output_dir=output_dir, 
            num_files=1  # You can adjust this if needed
        ) as file_writer:
            
            # Process each audio chunk
            for source_info, audio in tqdm(
                zip(self.source_infos, audio_iterator), 
                total=len(self.source_infos),
                desc="Generating embeddings"
            ):
                # Skip if audio is too short
                if audio.shape[0] < self.embed_fn.min_audio_s * self.config.embed_fn_config.model_config.sample_rate:
                    failed_embeddings += 1
                    continue
                
                # Get file identifier
                file_id = source_info.file_id(self.config.embed_fn_config.file_id_depth)
                offset_s = source_info.shard_num * source_info.shard_len_s
                
                # Generate embedding
                example = self.embed_fn.audio_to_example(file_id, offset_s, audio)
                
                if example is None:
                    failed_embeddings += 1
                    continue
                    
                # Write embedding to file
                file_writer.write(example.SerializeToString())
                successful_embeddings += 1
                
            file_writer.flush()
            
        print(f'\nEmbedding Generation Complete:')
        print(f'- Successfully processed: {successful_embeddings} files')
        print(f'- Failed: {failed_embeddings} files')
        
        # Create TensorFlow dataset from generated embeddings
        embedding_files = [fn for fn in output_dir.glob('embeddings-*')]
        ds = tf.data.TFRecordDataset(embedding_files)
        parser = tf_examples.get_example_parser()
        ds = ds.map(parser)
        
        return ds, (successful_embeddings, failed_embeddings)

    def verify_embeddings(self, dataset):
        """Verify the generated embeddings by checking the first example.
        
        Args:
            dataset: TensorFlow dataset containing embeddings
            
        Returns:
            bool: True if verification passes
        """
        try:
            for example in dataset.as_numpy_iterator():
                print('Embedding verification:')
                print(f'- Recording filename: {example["filename"]}')
                print(f'- Embedding shape: {example["embedding"].shape}')
                return True
        except Exception as e:
            print(f"Embedding verification failed: {str(e)}")
            return False
