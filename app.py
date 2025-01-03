from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from utils.audio_processor import predict_emotion
import os
import logging
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)

# Configuration
app.config.update(
    UPLOAD_FOLDER='uploads',
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file size
    MODEL_FOLDER='models',
    ALLOWED_EXTENSIONS={'wav', 'mp3'},
    SECRET_KEY=os.urandom(24)  # For session management
)

# Ensure required folders exist
for folder in ['uploads', 'models']:
    os.makedirs(os.path.join(app.root_path, folder), exist_ok=True)

class AudioAnalysisError(Exception):
    """Custom exception for audio analysis errors"""
    pass

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_model_paths():
    """Get model file paths with validation"""
    model_paths = {
        'json': os.path.join(app.config['MODEL_FOLDER'], 'CNN_model.json'),
        'weights': os.path.join(app.config['MODEL_FOLDER'], 'CNN_model_weights.h5'),
        'scaler': os.path.join(app.config['MODEL_FOLDER'], 'scaler2.pickle'),
        'encoder': os.path.join(app.config['MODEL_FOLDER'], 'encoder2.pickle')
    }
    
    # Validate all model files exist
    for path_name, path in model_paths.items():
        if not os.path.exists(path):
            raise AudioAnalysisError(f"Model file not found: {path_name}")
    
    return model_paths

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/result')
def result():
    """Render the results page"""
    emotion = request.args.get('emotion', 'Unknown')
    confidence = request.args.get('confidence', '0%')
    return render_template('result.html', emotion=emotion, confidence=confidence)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle audio file upload and emotion prediction"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            raise AudioAnalysisError('No file provided')
        
        file = request.files['file']
        if file.filename == '':
            raise AudioAnalysisError('No file selected')
        
        if not file or not allowed_file(file.filename):
            raise AudioAnalysisError('Invalid file type')
        
        # Generate unique filename
        unique_filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save file
        try:
            file.save(filepath)
            logging.info(f"File saved successfully: {filepath}")
        except Exception as e:
            raise AudioAnalysisError(f"Error saving file: {str(e)}")
        
        try:
            # Get model paths
            model_paths = get_model_paths()
            
            # Predict emotion
            emotion, confidence = predict_emotion(
                filepath,
                model_paths['json'],
                model_paths['weights'],
                model_paths['scaler'],
                model_paths['encoder']
            )
            
            logging.info(f"Prediction successful - Emotion: {emotion}, Confidence: {confidence:.2%}")
            
            return jsonify({
                'success': True,
                'emotion': emotion,
                'confidence': f"{confidence:.2%}",
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            raise AudioAnalysisError(f"Error during prediction: {str(e)}")
        
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
                logging.info(f"Cleaned up file: {filepath}")
    
    except AudioAnalysisError as e:
        logging.error(f"Audio analysis error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'An unexpected error occurred'
        }), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB'
    }), 413

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        # Check if model files exist
        get_model_paths()
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    # Check if model files exist before starting
    try:
        get_model_paths()
        logging.info("All model files found. Starting server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logging.error(f"Server startup failed: {str(e)}")