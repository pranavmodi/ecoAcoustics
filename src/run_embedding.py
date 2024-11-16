# main.py or app.py

import os
from services.embedding_service import EmbeddingService

def init_embedding_service():
    # Get the project root directory
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define paths
    MODEL_PATH = os.path.join(PROJECT_ROOT, "model")
    INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "input")  # For raw audio files
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output")  # Base output directory
    
    print(f"MODEL_PATH: {MODEL_PATH}")
    print(f"INPUT_DIR: {INPUT_DIR}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    
    # Initialize the service
    embedding_service = EmbeddingService(
        model_path=MODEL_PATH,
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR
    )
    
    # Load the model
    embedding_service.load_model()
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    embeddings_ds, (successful, failed) = embedding_service.generate_embeddings()
    
    # Verify embeddings
    if embedding_service.verify_embeddings(embeddings_ds):
        print("\nEmbedding generation successful!")
    else:
        print("\nWarning: Embedding verification failed")
    
    # Print the directory structure
    print(f"\nDirectory Structure:")
    print(f"- Embeddings saved to: {embedding_service.embedding_output_dir}")
    print(f"- Labeled data will be saved to: {embedding_service.labeled_data_path}")
    print(f"- Configuration saved to: {os.path.join(OUTPUT_DIR, 'embedding_config.json')}")
    
    return embedding_service, embeddings_ds

# Initialize the service and generate embeddings
embedding_service, embeddings_ds = init_embedding_service()