import numpy as np
from mfe.dsp import generate_features
import librosa

# Configuration for mfe function #

raw_axes = [1]
draw_graphs = False
frame_length = 0.032     # The length of each frame in seconds
frame_stride = 0.016     # The step between successive frames in seconds
num_filters = 40  # Filter number should be at least 2
fft_length = 256  # Number of FFT points
low_frequency = 300  # Lowest band edge of mel filters
noise_floor_db = -52    # Everything below this loudness will be dropped

#

raw_features, frequency = librosa.load('27_12_2022-00_00_08.wav',  # example file
                                       sr=16000,  # the sampling frequency, equal to 16kHz
                                       duration=2,  # number of second of the window
                                       dtype=np.float64)

mfe_processed = generate_features(3,
                                  draw_graphs,
                                  raw_features,
                                  raw_axes,
                                  sampling_freq=frequency,
                                  frame_length=frame_length,
                                  frame_stride=frame_stride,
                                  num_filters=num_filters,
                                  fft_length=fft_length,
                                  low_frequency=low_frequency,
                                  noise_floor_db=noise_floor_db)

print("Dimensionality of original features: ", len(raw_features))
print("Dimensionality of features processed with MFE function: ", len(mfe_processed['features']))