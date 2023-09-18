import numpy as np
import re
import sys
from pipeline import build_ad_model, build_tl_model, calculate_standard_len, calculate_true_label
import spectogram
import mfe
import librosa
import os
from river import anomaly, compose, preprocessing
from sklearn.metrics import roc_auc_score
from deep_river.anomaly import Autoencoder, ProbabilityWeightedAutoencoder, RollingAutoencoder
from tqdm import tqdm
from torch import nn

# Configuration for spectrogram function #
raw_axes_s = [1]
draw_graphs_s = False
show_axes_s = False
frame_length_s = 0.02  # The length of each frame in seconds
frame_stride_s = 0.01  # The step between successive frames in seconds
fft_length_s = 128  # Number of FFT points
noise_floor_db_s = -50  # Everything below this loudness will be dropped

# Configuration for mfe function #
raw_axes_m = [1]
draw_graphs_m = False
frame_length_m = 0.032     # The length of each frame in seconds
frame_stride_m = 0.016     # The step between successive frames in seconds
num_filters_m = 40  # Filter number should be at least 2
fft_length_m = 256  # Number of FFT points
low_frequency_m = 300  # Lowest band edge of mel filters
noise_floor_db_m = -52    # Everything below this loudness will be dropped

sys.setrecursionlimit(2500)

def build_model(model_name):
    model, num_params = build_ad_model(model_name=model_name)
    return model, num_params


def train_model(preprocess_method, model, num_params, transfer_learning=None, train_directory=None):
    # checking everything is inserted correctly
    if preprocess_method.lower() != 'mfe' and preprocess_method.lower() != 'spectrogram':
        raise TypeError('Wrong preprocessing method: insert either MFE or Spectrogram')
    if transfer_learning is not None and transfer_learning.lower() != 'flatten' and transfer_learning.lower() != 'dense':
        raise TypeError('Wrong transfer learning choice: insert either Flatten or Dense')
    if transfer_learning is not None:
        model_fe = build_tl_model(preprocess_method, transfer_learning)
    standard_len = calculate_standard_len(preprocess_method)
    
    scores, labels = [], []
    # iterating on all the files of the directory
    for iteration, filename in enumerate(tqdm(os.listdir(train_directory))):
        path = os.path.join(train_directory, filename)

        for i in range(9):

            # audio sub windowing
            raw_features, frequency = librosa.load(path,  # example file
                                                   sr=16000,  # the sampling frequency, equal to 16kHz
                                                   duration=2,  # number of second of the window
                                                   offset=i,
                                                   dtype=np.float64)

            # preprocessing methods
            if preprocess_method.lower() == 'mfe':
                processed = mfe.dsp.generate_features(3,
                                                      draw_graphs_s,
                                                      raw_features,
                                                      raw_axes_s,
                                                      sampling_freq=frequency,
                                                      frame_length=frame_length_m,
                                                      frame_stride=frame_stride_m,
                                                      num_filters=num_filters_m,
                                                      fft_length=fft_length_m,
                                                      low_frequency=low_frequency_m,
                                                      noise_floor_db=noise_floor_db_m)
            else:
                processed = spectogram.dsp.generate_features(3, draw_graphs_s, raw_features, raw_axes_s, frequency,
                                                             frame_length_s, frame_stride_s, fft_length_s, show_axes_s,
                                                             noise_floor_db_s)
            if len(processed['features']) != standard_len:
                list_of_zeros = [0] * (standard_len - len(processed['features']))
                processed['features'].extend(list_of_zeros)

            # transfer learning feature extraction
            if transfer_learning is not None:
                # use the following command to create the features from a single instance of data on which to train the SML Anomaly Detector
                features = model_fe.predict(np.array(processed['features'])[None], verbose=0)
                x = {str(i + 1): val for i, val in enumerate(features[0])}
            else:
                x = {str(i + 1): val for i, val in enumerate(processed['features'])}

            # anomaly detection
            score = model.score_one(x)
            y = calculate_true_label(filename)
            if num_params == 2:
                model = model.learn_one(x, y)
            else:
                model = model.learn_one(x)
            scores.append(score)
            labels.append(y)
    if all(label == 0 for label in labels):
        labels[0] = 1
    roc = roc_auc_score(labels, scores)
    print("TRAINING: rocauc score using "+transfer_learning+": " + str(roc))
    return model


def test_model(preprocess_method, model, num_params, transfer_learning=None,  test_directory=None):
    # checking everything is inserted correctly
    if preprocess_method.lower() != 'mfe' and preprocess_method.lower() != 'spectrogram':
        raise TypeError('Wrong preprocessing method: insert either MFE or Spectrogram')
    if transfer_learning is not None and transfer_learning.lower() != 'flatten' and transfer_learning.lower() != 'dense':
        raise TypeError('Wrong transfer learning choice: insert either Flatten or Dense')
    if transfer_learning is not None:
        model_fe = build_tl_model(preprocess_method, transfer_learning)
    standard_len = calculate_standard_len(preprocess_method)

    scores, labels = [], []
    pathscores = []
    for iteration, filename in enumerate(tqdm(os.listdir(test_directory))):
        path = os.path.join(test_directory, filename)

        for i in range(9):

            # audio sub windowing
            raw_features, frequency = librosa.load(path,  # example file
                                                   sr=16000,  # the sampling frequency, equal to 16kHz
                                                   duration=2,  # number of second of the window
                                                   offset=i,
                                                   dtype=np.float64)

            # preprocessing methods
            if preprocess_method.lower() == 'mfe':
                processed = mfe.dsp.generate_features(3,
                                                      draw_graphs_s,
                                                      raw_features,
                                                      raw_axes_s,
                                                      sampling_freq=frequency,
                                                      frame_length=frame_length_m,
                                                      frame_stride=frame_stride_m,
                                                      num_filters=num_filters_m,
                                                      fft_length=fft_length_m,
                                                      low_frequency=low_frequency_m,
                                                      noise_floor_db=noise_floor_db_m)
            else:
                processed = spectogram.dsp.generate_features(3, draw_graphs_s, raw_features, raw_axes_s, frequency,
                                                             frame_length_s, frame_stride_s, fft_length_s, show_axes_s,
                                                             noise_floor_db_s)
            if len(processed['features']) != standard_len:
                list_of_zeros = [0] * (standard_len - len(processed['features']))
                processed['features'].extend(list_of_zeros)

            # transfer learning feature extraction
            if transfer_learning is not None:
                # use the following command to create the features from a single instance of data on which to train the SML Anomaly Detector
                features = model_fe.predict(np.array(processed['features'])[None], verbose=0)
                x = {str(i + 1): val for i, val in enumerate(features[0])}
            else:
                x = {str(i + 1): val for i, val in enumerate(processed['features'])}

            # anomaly detection
            score = model.score_one(x)
            y = calculate_true_label(filename)
            scores.append(score)
            labels.append(y)
            pathscores.append((path,score))
    roc = roc_auc_score(labels, scores)
    print("TESTING: rocauc score using " + transfer_learning + ": " + str(roc))
    return pathscores


def test_sample(preprocess_method, model, num_params, transfer_learning=None,  test_directory=None):
    # checking everything is inserted correctly
    if preprocess_method.lower() != 'mfe' and preprocess_method.lower() != 'spectrogram':
        raise TypeError('Wrong preprocessing method: insert either MFE or Spectrogram')
    if transfer_learning is not None and transfer_learning.lower() != 'flatten' and transfer_learning.lower() != 'dense':
        raise TypeError('Wrong transfer learning choice: insert either Flatten or Dense')
    if transfer_learning is not None:
        model_fe = build_tl_model(preprocess_method, transfer_learning)
    standard_len = calculate_standard_len(preprocess_method)

    scores, labels = [], []
    pathscores = []
    for iteration, filename in enumerate(tqdm(os.listdir(test_directory))):
        path = os.path.join(test_directory, filename)

        for i in range(9):

            # audio sub windowing
            raw_features, frequency = librosa.load(path,  # example file
                                                   sr=16000,  # the sampling frequency, equal to 16kHz
                                                   duration=2,  # number of second of the window
                                                   offset=i,
                                                   dtype=np.float64)

            # preprocessing methods
            if preprocess_method.lower() == 'mfe':
                processed = mfe.dsp.generate_features(3,
                                                      draw_graphs_s,
                                                      raw_features,
                                                      raw_axes_s,
                                                      sampling_freq=frequency,
                                                      frame_length=frame_length_m,
                                                      frame_stride=frame_stride_m,
                                                      num_filters=num_filters_m,
                                                      fft_length=fft_length_m,
                                                      low_frequency=low_frequency_m,
                                                      noise_floor_db=noise_floor_db_m)
            else:
                processed = spectogram.dsp.generate_features(3, draw_graphs_s, raw_features, raw_axes_s, frequency,
                                                             frame_length_s, frame_stride_s, fft_length_s, show_axes_s,
                                                             noise_floor_db_s)
            if len(processed['features']) != standard_len:
                list_of_zeros = [0] * (standard_len - len(processed['features']))
                processed['features'].extend(list_of_zeros)

            # transfer learning feature extraction
            if transfer_learning is not None:
                # use the following command to create the features from a single instance of data on which to train the SML Anomaly Detector
                features = model_fe.predict(np.array(processed['features'])[None], verbose=0)
                x = {str(i + 1): val for i, val in enumerate(features[0])}
            else:
                x = {str(i + 1): val for i, val in enumerate(processed['features'])}

            # anomaly detection
            score = model.score_one(x)
            y = calculate_true_label(filename)
            scores.append(score)
            labels.append(y)
            pathscores.append((path, score))
    return pathscores