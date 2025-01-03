import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import joblib
import keras
import warnings

warnings.filterwarnings('ignore')
keras.saving.register_keras_serializable()

def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=data, sr=sr)
    
    features = np.hstack([
        np.squeeze(zcr),
        np.squeeze(rmse),
        np.ravel(mfcc.T)
    ])
    
    return features

def prepare_audio_for_prediction(audio_path, scaler):
    data, sr = librosa.load(audio_path, duration=2.5, offset=0.6)
    features = extract_features(data)
    features = features.reshape(1, -1)
    scaled_features = scaler.transform(features)
    scaled_features = np.expand_dims(scaled_features, axis=2)
    return scaled_features

emotions = {
    0: 'Neutral', 
    1: 'Happy', 
    2: 'Sad', 
    3: 'Angry', 
    4: 'Fear', 
    5: 'Disgust', 
    6: 'Surprise'
}

def predict_emotion(audio_path, model_json_path, model_weights_path, scaler_path, encoder_path):
    with open(scaler_path, 'rb') as f:
        scaler = joblib.load(f)
    
    with open(encoder_path, 'rb') as f:
        encoder = joblib.load(f)
    
    with open(model_json_path, 'r') as json_file:
        model_json = json_file.read()
    
    custom_objects = {
        'GlorotUniform': tf.keras.initializers.GlorotUniform,
        'Sequential': tf.keras.Sequential
    }
    
    model = model_from_json(model_json, custom_objects=custom_objects)
    model.load_weights(model_weights_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    prepared_audio = prepare_audio_for_prediction(audio_path, scaler)
    prediction = model.predict(prepared_audio)
    
    emotion_index = np.argmax(prediction)
    emotion = emotions[emotion_index]
    confidence = prediction[0][emotion_index]
    
    return emotion, confidence