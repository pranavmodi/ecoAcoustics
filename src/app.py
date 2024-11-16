from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
from werkzeug.utils import secure_filename
from chirp.inference import tf_examples  # Assuming you have this
import numpy as np
from pathlib import Path
import sqlite3
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Directory to save uploaded audio files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Add this near the top with other global variables
CLASSES_FILE = 'classes.json'

ALLOWED_EXTENSIONS = {'wav', 'mp3'}  # Add other audio formats as needed

# Add these constants near the top
EMBEDDINGS_DIR = Path("data/output/raw_embeddings")
RESULTS_LIMIT = 5  # Number of top matches to return
DATABASE = 'labels.db'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_db():
    """Initialize SQLite database for storing labels"""
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            label TEXT NOT NULL,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Call this when your app starts
init_db()

@app.route('/api/classes', methods=['GET', 'POST'])
def manage_classes():
    if request.method == 'POST':
        # Expect format: {"classes": ["wood thrush", "unknown", ...]}
        data = request.get_json()
        if not data or 'classes' not in data:
            return jsonify({'error': 'Invalid request format'}), 400
        
        classes = data['classes']
        if not isinstance(classes, list) or not all(isinstance(c, str) for c in classes):
            return jsonify({'error': 'Classes must be a list of strings'}), 400
        
        # Save classes to file
        with open(CLASSES_FILE, 'w') as f:
            json.dump({'classes': classes}, f)
        
        return jsonify({'message': 'Classes updated successfully', 'classes': classes})
    
    else:  # GET request
        try:
            with open(CLASSES_FILE, 'r') as f:
                classes = json.load(f)
            return jsonify(classes)
        except FileNotFoundError:
            return jsonify({'classes': []})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Generate embedding
        try:
            embedding = generate_embedding(filepath)
            return jsonify({
                'message': 'File uploaded and embedding generated',
                'filename': filename,
                'embedding_shape': embedding.shape
            })
        except Exception as e:
            return jsonify({'error': f'Error generating embedding: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

def generate_embedding(audio_path):
    """
    Generate embedding for uploaded audio file
    You'll need to implement this based on your chirp setup
    """
    # This is a placeholder - implement based on your chirp configuration
    model = tf_examples.get_model()  # Get your embedding model
    audio_data = tf_examples.load_audio(audio_path)  # Load audio appropriately
    embedding = model.predict(audio_data)
    return embedding

@app.route('/api/embed', methods=['POST'])
def generate_embeddings():
    # Generate embeddings for uploaded audio
    # Store embeddings
    # Return embedding reference

@app.route('/api/search', methods=['POST'])
def similarity_search():
    """
    Find similar audio files based on embedding similarity
    Expects the query filename that was previously uploaded
    """
    data = request.get_json()
    if not data or 'filename' not in data:
        return jsonify({'error': 'Query filename required'}), 400
    
    query_filename = secure_filename(data['filename'])
    query_path = os.path.join(app.config['UPLOAD_FOLDER'], query_filename)
    
    if not os.path.exists(query_path):
        return jsonify({'error': 'Query file not found'}), 404
    
    try:
        # Get query embedding
        query_embedding = generate_embedding(query_path)
        
        # Find similar embeddings
        similar_files = find_similar_embeddings(query_embedding)
        
        return jsonify({
            'matches': similar_files
        })
    
    except Exception as e:
        return jsonify({'error': f'Search error: {str(e)}'}), 500

def find_similar_embeddings(query_embedding):
    """
    Find similar embeddings using cosine similarity
    Returns list of (filename, similarity_score) tuples
    """
    # Load stored embeddings from TFRecord files
    embedding_files = list(EMBEDDINGS_DIR.glob("embeddings-*"))
    if not embedding_files:
        raise FileNotFoundError("No embedding files found in database")
    
    # Create TensorFlow dataset
    ds = tf.data.TFRecordDataset([str(f) for f in embedding_files])
    parser = tf_examples.get_example_parser()
    ds = ds.map(parser)
    
    similarities = []
    for example in ds:
        stored_embedding = example['embedding'].numpy()
        filename = example['filename'].numpy().decode()
        
        # Calculate cosine similarity
        similarity = cosine_similarity(query_embedding, stored_embedding)
        similarities.append({
            'filename': filename,
            'similarity': float(similarity),  # Convert to Python float for JSON serialization
        })
    
    # Sort by similarity score (highest first) and get top matches
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    return similarities[:RESULTS_LIMIT]

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.route('/api/train', methods=['POST'])
def train_classifier():
    # Take labeled examples
    # Train linear classifier
    # Return model metrics

@app.route('/api/classify', methods=['POST'])
def classify_audio():
    # Run classification on new audio
    # Return predictions

@app.route('/api/audio/<path:filename>')
def serve_audio(filename):
    """Stream audio file to client"""
    try:
        # Check both uploads folder and original dataset location
        if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
            return send_file(
                os.path.join(app.config['UPLOAD_FOLDER'], filename),
                mimetype='audio/wav'
            )
        # Add your dataset path here
        elif os.path.exists(os.path.join('data/audio', filename)):
            return send_file(
                os.path.join('data/audio', filename),
                mimetype='audio/wav'
            )
        else:
            return jsonify({'error': 'Audio file not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Error serving audio: {str(e)}'}), 500

@app.route('/api/label', methods=['POST'])
def label_audio():
    """Label an audio file"""
    data = request.get_json()
    required_fields = ['filename', 'label']
    if not data or not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400
    
    filename = secure_filename(data['filename'])
    label = data['label']
    confidence = data.get('confidence', 1.0)  # Optional confidence score
    
    # Verify label is valid
    try:
        with open(CLASSES_FILE, 'r') as f:
            valid_classes = json.load(f)['classes']
        if label not in valid_classes:
            return jsonify({'error': f'Invalid label. Must be one of: {valid_classes}'}), 400
    except FileNotFoundError:
        return jsonify({'error': 'No classes defined'}), 400
    
    # Store label in database
    try:
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute('''
            INSERT INTO labels (filename, label, confidence)
            VALUES (?, ?, ?)
        ''', (filename, label, confidence))
        conn.commit()
        conn.close()
        
        return jsonify({
            'message': 'Label added successfully',
            'filename': filename,
            'label': label,
            'confidence': confidence
        })
    
    except Exception as e:
        return jsonify({'error': f'Error storing label: {str(e)}'}), 500

@app.route('/api/labels', methods=['GET'])
def get_labels():
    """Get all labels for a file"""
    filename = request.args.get('filename')
    if not filename:
        return jsonify({'error': 'Filename parameter required'}), 400
    
    try:
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute('''
            SELECT label, confidence, created_at
            FROM labels
            WHERE filename = ?
            ORDER BY created_at DESC
        ''', (filename,))
        labels = [{
            'label': row[0],
            'confidence': row[1],
            'created_at': row[2]
        } for row in c.fetchall()]
        conn.close()
        
        return jsonify({'filename': filename, 'labels': labels})
    
    except Exception as e:
        return jsonify({'error': f'Error retrieving labels: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
