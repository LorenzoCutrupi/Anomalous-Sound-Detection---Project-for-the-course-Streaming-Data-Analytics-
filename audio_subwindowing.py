import os

import numpy as np
import librosa
from spectogram.dsp import generate_features

# Configuration for spectogram function #

raw_axes = [1]
draw_graphs = False
show_axes = False
frame_length = 0.02     # The length of each frame in seconds
frame_stride = 0.01     # The step between successive frames in seconds
fft_length = 128        # Number of FFT points
noise_floor_db = -50    # Everything below this loudness will be dropped

# use the following code to iterate on the same file in order to obtain 9 sequences of 2 seconds from a single audio
# of 10 second

for i in range(9):
    raw_features, frequency = librosa.load('27_12_2022-00_00_08.wav',  # example file
                                           sr=16000,  # the sampling frequency, equal to 16kHz
                                           duration=2,  # number of second of the window
                                           offset=i,  # start reading after this time (in seconds)
                                           dtype=np.float64)

    feat = generate_features(3, draw_graphs, raw_features, raw_axes, frequency,
                          frame_length, frame_stride, fft_length, show_axes, noise_floor_db)
