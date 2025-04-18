# project informarion
This is a speech emotion recognition project that seeks to classify speechs based on emotions e.g anger,fear,disgust. We made us of the LSTM Network in order to train our model to have a model accuracy of 99%.



# Dataset Information
There are a set of 200 target words were spoken in the carrier phrase "Say the word _' by two actresses (aged 26 and 64 years) and recordings were made of the set portraying each of seven emotions (anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral). There are 2800 data points (audio files) in total.
you can get this dataset here:  https://www.kaggle.com/dmitrybabko/speech-emotion-recognition-en


# Output Attributes
    anger
    disgust
    fear
    happiness
    pleasant surprise
    sadness
    neutral


# installation
    - clone this repository
    - cd into the folder where the cloned repository exist


# Create and activate environment using te followong:
    - python -m venv venv
    - venv\scripts\activation
    - python -m  pip install --upgrade pip
    - pip install requirements.txt

# run the flask app for testing using:
    - python app.py