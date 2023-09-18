from sklearn.metrics import roc_auc_score
from tensorflow import keras
import tensorflow as tf
import numpy as np

import spectogram
from pipeline import calculate_true_label
from spectogram.dsp import generate_features
import librosa
import os
from tqdm import tqdm

CUT = 'flatten'  # choose at which layer to cut the neural network

model = keras.models.load_model(os.path.join(os.path.dirname(__file__), 'spectrogram_model.h5'))
new_model = tf.keras.models.Sequential(model.layers[:])

raw_axes = [1]
draw_graphs = False
show_axes = False
frame_length = 0.02  # The length of each frame in seconds
frame_stride = 0.01  # The step between successive frames in seconds
fft_length = 128  # Number of FFT points
noise_floor_db = -50  # Everything below this loudness will be dropped

test_directory = 'test'
scores_n, scores_a, labels = [], [], []
for iteration, filename in enumerate(os.listdir(test_directory)):
    path = os.path.join(test_directory, filename)

    for i in range(9):

        # audio sub windowing
        raw_features, frequency = librosa.load(path,  # example file
                                               sr=16000,  # the sampling frequency, equal to 16kHz
                                               duration=2,  # number of second of the window
                                               offset=i,
                                               dtype=np.float64)

        processed = spectogram.dsp.generate_features(3, draw_graphs, raw_features, raw_axes, frequency,
                                                     frame_length, frame_stride, fft_length, show_axes,
                                                     noise_floor_db)

        if len(processed['features']) != 12935:
            list_of_zeros = [0] * (12935 - len(processed['features']))
            processed['features'].extend(list_of_zeros)

        # anomaly detection
        score = new_model.predict(np.array(processed['features'])[None])
        y = calculate_true_label(filename)
        if 'abnormal' in path:
            scores_a.append(score)
        else:
            scores_n.append(score)
        labels.append(y)

print(scores_n)
print('\n\n\n\n\n\n\n')
print(scores_a)
'''
roc = roc_auc_score(labels, scores)
print("TESTING: " + str(roc))'''
