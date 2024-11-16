import json
import numpy as np
import os
from pathlib import Path
import tensorflow as tf
from chirp.inference import tf_examples  # Add this import if you have access to chirp

def read_embedding_info(config_path="data/output/embedding_config.json"):
    """
    Read embeddings and print their information from TFRecord files
    """
    # Read config file
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    output_dir = Path(config['paths']['output_dir'])
    embeddings_dir = output_dir / "raw_embeddings"
    
    if not embeddings_dir.exists():
        print(f"No embeddings directory found at {embeddings_dir}")
        return
    
    print(f"\nAnalyzing embeddings in: {embeddings_dir}\n")
    print("-" * 80)
    
    # Look for embeddings TFRecord files
    embedding_files = list(embeddings_dir.glob("embeddings-*"))
    
    if not embedding_files:
        print(f"No embedding files found in {embeddings_dir}")
        print(f"Available files in directory:")
        for file in embeddings_dir.iterdir():
            print(f"- {file.name}")
        return

    # Create TensorFlow dataset from the embeddings
    ds = tf.data.TFRecordDataset([str(f) for f in embedding_files])
    parser = tf_examples.get_example_parser()  # If you have access to chirp
    # If you don't have access to chirp, you'll need to define your own parser
    ds = ds.map(parser)
    
    # Count total examples
    total_examples = sum(1 for _ in ds)
    print(f"Total number of embeddings: {total_examples}\n")
    
    # Reset dataset
    ds = tf.data.TFRecordDataset([str(f) for f in embedding_files])
    ds = ds.map(parser)
    
    # Analyze the first few examples
    for i, example in enumerate(ds.take(min(5, total_examples))):
        file_size = os.path.getsize(embedding_files[0]) / (1024 * 1024)  # Size of first file
        embedding = example['embedding'].numpy()
        
        print(f"Example {i+1}:")
        print(f"Source file: {example['filename'].numpy().decode()}")
        print(f"Embedding dimensions: {embedding.shape}")
        print(f"Data type: {embedding.dtype}")
        print(f"File size: {file_size:.2f} MB")
        print(f"Value range: [{embedding.min():.3f}, {embedding.max():.3f}]")
        print("-" * 80)

if __name__ == "__main__":
    read_embedding_info()




# # Define classes
# curl -X POST http://localhost:5000/api/classes \
#   -H "Content-Type: application/json" \
#   -d '{"classes": ["wood thrush", "unknown"]}'

# # Get current classes
# curl http://localhost:5000/api/classes


# Upload file and generate embedding
# curl -X POST http://localhost:5000/api/upload \
#   -F "file=@/path/to/your/audio.wav"


# Search for similar audio files
# curl -X POST http://localhost:5000/api/search \
#   -H "Content-Type: application/json" \
#   -d '{"filename": "query_audio.wav"}'


# Play audio file (in browser or curl)
# curl http://localhost:5000/api/audio/example.wav

# # Add label to audio file
# curl -X POST http://localhost:5000/api/label \
#   -H "Content-Type: application/json" \
#   -d '{
#     "filename": "example.wav",
#     "label": "wood thrush",
#     "confidence": 0.95
#   }'

# # Get labels for a file
# curl http://localhost:5000/api/labels?filename=example.wav