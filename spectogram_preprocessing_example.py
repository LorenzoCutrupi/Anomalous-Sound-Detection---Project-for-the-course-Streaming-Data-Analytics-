import numpy as np
from spectogram.dsp import generate_features
import librosa

# Configuration for spectrogram function #

raw_axes = [1]
draw_graphs = False
show_axes = False
frame_length = 0.02     # The length of each frame in seconds
frame_stride = 0.01     # The step between successive frames in seconds
fft_length = 128        # Number of FFT points
noise_floor_db = -50    # Everything below this loudness will be dropped

#

raw_features, frequency = librosa.load('27_12_2022-00_00_08.wav',  # example file
                                       sr=16000, duration=2, dtype=np.float64)

spectogram_processed = generate_features(3, draw_graphs, raw_features, raw_axes, frequency,
                                    frame_length, frame_stride, fft_length, show_axes, noise_floor_db)

print("Dimensionality of original features: ", len(raw_features))
print("Dimensionality of features processed with Spectrogram function: ", len(spectogram_processed['features']))
# print(spectogram_processed['features'])
