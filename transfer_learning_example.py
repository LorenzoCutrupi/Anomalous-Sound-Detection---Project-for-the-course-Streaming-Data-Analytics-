from tensorflow import keras
import tensorflow as tf
import numpy as np
from spectogram.dsp import generate_features
import librosa
import os
from river import anomaly
from river import metrics

CUT = 'flatten'  # choose at which layer to cut the neural network

model = keras.models.load_model(os.path.join(os.path.dirname(__file__), 'spectrogram_model.h5'))

print(model.summary())
print()
if CUT == 'flatten':  # output shape 800
    new_model = tf.keras.models.Sequential(model.layers[:-3])
else:  # CUT == 'dense'  output shape 20
    new_model = tf.keras.models.Sequential(model.layers[:-2])
print(new_model.summary())

# Configuration for spectrogram function #

raw_axes = [1]
draw_graphs = False
show_axes = False
frame_length = 0.02  # The length of each frame in seconds
frame_stride = 0.01  # The step between successive frames in seconds
fft_length = 128  # Number of FFT points
noise_floor_db = -50  # Everything below this loudness will be dropped

#

raw_features, frequency = librosa.load('14_01_2023-00_32_09.wav',  # example file
                                       sr=16000,  # the sampling frequency, equal to 16kHz
                                       duration=2,  # number of second of the window
                                       dtype=np.float64)

spectrogram_processed = generate_features(3, draw_graphs, raw_features, raw_axes, frequency,
                                          frame_length, frame_stride, fft_length, show_axes, noise_floor_db)


print("Dimensionality before NN Feature Extractor", len(spectrogram_processed['features']))

# use the following command to create the features from a single instance of data on which to train the SML Anomaly Detector
features = new_model.predict(np.array(spectrogram_processed['features'])[None])

print("Dimensionality after NN Feature Extractor", len(features[0]))
