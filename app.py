from flask import Flask, request, render_template
import librosa
import numpy as np
import joblib
import tensorflow as tf
import os

app = Flask(__name__)

# Load model and one-hot encoder
model = tf.keras.models.load_model('speech_model.h5')
encoder = joblib.load('label_encoder.pkl') 

model.summary()

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc.reshape(1, -1, 1)  # Reshape for LSTM input: (batch, timesteps, features)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        audio = request.files.get('audio')
        if audio:
            file_path = os.path.join('uploads', audio.filename)
            audio.save(file_path)

            try:
                # Extract features and predict
                features = extract_mfcc(file_path)
                prediction = model.predict(features)
                predicted_label_index = np.argmax(prediction)

                # Decode class label using OneHotEncoder
                predicted_label = encoder.categories_[0][predicted_label_index]

                
                os.remove(file_path)

                return render_template('index.html', prediction=predicted_label)
            except Exception as e:
                
                if os.path.exists(file_path):
                    os.remove(file_path)
                return render_template('index.html', prediction="Error: {}".format(str(e)))

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
