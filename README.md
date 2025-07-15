
🎙️ Speech Emotion Detection using Deep Learning
An advanced Convolutional Neural Network (CNN)-based system that detects human emotions—like anger, happiness, sadness, or neutrality—directly from audio speech signals. This project demonstrates the power of deep learning for real-time emotional intelligence in speech-based applications.

📌 Project Summary
This project implements a Speech Emotion Recognition (SER) system that processes WAV audio inputs and classifies emotions using spectrogram-based features like MFCCs, STFT, and Spectral Contrast. Our model achieves ~87.5% accuracy, outperforming traditional machine learning models like SVM and Random Forest.

The model is trained on well-known public datasets like:

TESS / Target Words Dataset (2,800 WAV files, 7 emotions, 2 speakers)

SAVEE Dataset (480 WAV files, 4 speakers, 7 emotions)

⚙️ Technologies Used
🧠 Deep Learning: CNN using TensorFlow & Keras

🎧 Signal Processing: Librosa for MFCCs, STFT, Spectral Contrast

🌐 Deployment: Flask-based web app

🚀 Acceleration: GPU training, Dropout, EarlyStopping, Adam Optimizer

🛠️ Parallelism: Joblib to handle large-scale dataset feature extraction

🧪 Core Pipeline (Algorithm & Architecture)
🔊 1. Preprocessing
Noise reduction via spectral gating (300–3000Hz bandpass)

Normalization to balance audio amplitudes

Segmentation (20–40ms frames with overlap)

Data augmentation using pitch shift & time stretch

🎚️ 2. Feature Extraction
Using Librosa, the system extracts:

MFCCs (13 coefficients per frame)

STFT for time-frequency representation

Spectral Contrast for energy variation

🧠 3. Model Architecture
Input Layer: Spectrogram-based features

Convolutional Layers: Learn spatial patterns (pitch, tone)

MaxPooling: Dimensionality reduction

Fully Connected Layers: Dense representation for classification

Softmax Output: Predicts 1 of 7 emotion classes

Optimizer: Adam | Loss: Categorical Crossentropy

💡 Emotions Detected:
Anger, Disgust, Fear, Happiness, Sadness, Surprise, Neutral

🌍 Deployment
The system is deployed via a Flask web app. It supports:

Real-time predictions from uploaded .wav files

Confidence score for detected emotion

Future-ready Dockerized deployment (AWS, Heroku support)

📊 Results
Emotion	Precision	Recall	F1-Score
Anger	91%	89%	90%
Happiness	88%	92%	90%
Sadness	85%	80%	82%
Neutral	79%	75%	77%

CNN Accuracy: 87.5%

SVM Accuracy: 74.2%

Random Forest Accuracy: 68.5%

⚠️ Missing Files Notice
Some supplementary files (e.g., large model checkpoints, private datasets, training logs) are not included in this repository due to size limitations or licensing restrictions. However, all core scripts, model logic, and deployment pipeline are available and reproducible.

🔮 Future Enhancements
🔁 Real-time audio stream processing (WebSocket-based)

🎭 Multimodal Emotion Recognition (video + audio)

📱 Mobile app version

🌐 Support for multiple languages and accents

🔐 Privacy-first emotion analysis for healthcare

🤝 Contributions & Citations
Developed as a major project by final-year students at Bajaj Institute of Technology, Wardha under the guidance of Mr. Gajanan Tikhe.

For academic use, please cite relevant parts of this work and reference the project contributors in your acknowledgements.
