import numpy as np
import librosa
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Sequential, model_from_json, load_model
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dropout, Flatten, Dense
import joblib
import warnings
import json
warnings.filterwarnings('ignore')

class EmotionDetector:
    def __init__(self, model_json_path, model_weights_path, scaler_path, encoder_path):
        self.emotions = {
            0: {'name': 'Neutral', 'emoji': '😐'},
            1: {'name': 'Happy', 'emoji': '😊'},
            2: {'name': 'Sad', 'emoji': '😔'},
            3: {'name': 'Angry', 'emoji': '😠'},
            4: {'name': 'Fear', 'emoji': '😱'},
            5: {'name': 'Disgust', 'emoji': '🤢'},
            6: {'name': 'Surprise', 'emoji': '😲'}
        }
        
        # Load scaler
        try:
            with open(scaler_path, 'rb') as f:
                self.scaler = joblib.load(f)
        except Exception as e:
            raise Exception(f"Failed to load scaler: {str(e)}")

        # Load model
        try:
            # Try to load model architecture from JSON
            with open(model_json_path, 'r') as json_file:
                model_json = json_file.read()
            
            # Create a new Sequential model with the same architecture
            self.model = Sequential([
                Conv1D(512, 5, activation='relu', padding='same', input_shape=(2376, 1)),
                BatchNormalization(),
                MaxPooling1D(5, 2, padding='same'),
                
                Conv1D(512, 5, activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling1D(5, 2, padding='same'),
                Dropout(0.2),
                
                Conv1D(256, 5, activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling1D(5, 2, padding='same'),
                
                Conv1D(256, 3, activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling1D(5, 2, padding='same'),
                Dropout(0.2),
                
                Conv1D(128, 3, activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling1D(3, 2, padding='same'),
                Dropout(0.2),
                
                Flatten(),
                Dense(512, activation='relu'),
                BatchNormalization(),
                Dense(7, activation='softmax')
            ])
            
            # Load weights
            self.model.load_weights(model_weights_path)
            
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def extract_features(self, data, sr=22050, frame_length=2048, hop_length=512):
        zcr = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
        rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
        mfcc = librosa.feature.mfcc(y=data, sr=sr)
        
        features = np.hstack([
            np.squeeze(zcr),
            np.squeeze(rmse),
            np.ravel(mfcc.T)
        ])
        
        return features

    def predict_emotion(self, audio_path):
        data, sr = librosa.load(audio_path, duration=2.5, offset=0.6)
        features = self.extract_features(data)
        features = features.reshape(1, -1)
        scaled_features = self.scaler.transform(features)
        scaled_features = np.expand_dims(scaled_features, axis=2)
        
        prediction = self.model.predict(scaled_features)
        emotion_index = np.argmax(prediction)
        emotion_data = self.emotions[emotion_index]
        confidence = prediction[0][emotion_index]
        
        return {
            'emotion': emotion_data['name'],
            'emoji': emotion_data['emoji'],
            'confidence': float(confidence)
        }