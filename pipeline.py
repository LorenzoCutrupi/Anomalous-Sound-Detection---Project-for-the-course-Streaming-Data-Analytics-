from tensorflow import keras
import tensorflow as tf
import numpy as np
import re
import sys
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


class MyAutoEncoder(nn.Module):
    def __init__(self, n_features, latent_dim=3):
        super(MyAutoEncoder, self).__init__()
        self.linear1 = nn.Linear(n_features, latent_dim)
        self.non_linear = nn.LeakyReLU()
        self.linear2 = nn.Linear(latent_dim, n_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_X):
        input_X = self.linear1(input_X)
        input_X = self.non_linear(input_X)
        input_X = self.linear2(input_X)
        return self.sigmoid(input_X)


def build_tl_model(preprocess_method, transfer_learning):
    """
    Builds the feature extraction method used to reduce the dimensionality of each sample
    :param preprocess_method: preprocessing method (es. mfe or spectogram)
    :param transfer_learning: last layer kept from the neural network (flatten or dense)
    :return: transfer learning model that reduces the features to 800 or 20
    """
    if preprocess_method.lower() == 'mfe':
        model_to_return = keras.models.load_model(os.path.join(os.path.dirname(__file__), 'mfe_model.h5'))
        # output shape 1536
        return tf.keras.models.Sequential(model_to_return.layers[:-2])
    else:
        model_to_return = keras.models.load_model(os.path.join(os.path.dirname(__file__), 'spectrogram_model.h5'))
    if transfer_learning.lower() == 'flatten':  # output shape 800
        model_to_return = tf.keras.models.Sequential(model_to_return.layers[:-3])
    else:  # CUT == 'dense'  output shape 20
        model_to_return = tf.keras.models.Sequential(model_to_return.layers[:-2])
    return model_to_return


def build_ad_model(model_name):
    """
    Builds and returns the model to be used for anomaly detection among those of river and deep river libraries
    :param model_name: name of the model as a string
    :return: built model
    """
    if model_name.lower() == 'hst':
        model = compose.Pipeline(
            preprocessing.MinMaxScaler(),
            anomaly.HalfSpaceTrees()
        )
        return model, 2
    if model_name.lower() == 'ilof':
        model = compose.Pipeline(
            preprocessing.MinMaxScaler(),
            anomaly.ILOF()
        )
        return model, 2
    if model_name.lower() == 'kitnet':
        model = compose.Pipeline(
            preprocessing.MinMaxScaler(),
            anomaly.KitNet()
        )
        return model, 2
    if model_name.lower() == 'svm':
        model = anomaly.QuantileThresholder(
            anomaly.OneClassSVM(nu=0.2),
            q=0.995
        )
        return model, 1
    if model_name.lower() == 'rrcf':
        model = anomaly.RobustRandomCutForest(num_trees=10, tree_size=25)
        return model, 1
    if model_name.lower() == 'ae':
        model = compose.Pipeline(
            preprocessing.MinMaxScaler(),
            Autoencoder(module=MyAutoEncoder, lr=0.005)
        )
        return model, 1
    if model_name.lower() == 'wae':
        model = compose.Pipeline(
            preprocessing.MinMaxScaler(),
            ProbabilityWeightedAutoencoder(module=MyAutoEncoder, lr=0.005)
        )
        return model, 1
    if model_name.lower() == 'rae':
        model = compose.Pipeline(
            preprocessing.MinMaxScaler(),
            RollingAutoencoder(module=MyAutoEncoder, lr=0.005)
        )
        return model, 1
    raise TypeError('Invalid anomaly detection method selected')


def calculate_standard_len(preprocess_method):
    """
    Calculates the amount of features extracted by the preprocessing method
    :param preprocess_method: preprocessing method as string
    :return: number of features
    """
    if preprocess_method.lower() == 'mfe':
        return 4960
    else:
        return 12935


def calculate_true_label(string):
    if "abnormal" in string:
        return 1
    else:
        return 0


def anomaly_detection(preprocess_method, model_name, transfer_learning=None, train_directory=None, test_directory=None):
    """
    Pipeline to fully perform the anomaly detection process. First, a model is trained on a dataset, than such model is
    tested on a new set. Each data sample is preprocessed splitting it in sub samples of 2 seconds length, then numerical
    features are generated based on a criteria (mfe or spectogram) and the features are eventually extracted to reduce
    dimensionality.
    :param preprocess_method: mfe or spectogram, used to generate numerical features from an audio
    :param model_name: name of the model to build
    :param transfer_learning: level to maintain of the pre-trained neural network used for feature extraction
    :param train_directory: directory of the training dataset
    :param test_directory: directory of the test dataset
    :return:
    """
    # checking everything is inserted correctly
    if preprocess_method.lower() != 'mfe' and preprocess_method.lower() != 'spectrogram':
        raise TypeError('Wrong preprocessing method: insert either MFE or Spectrogram')
    if transfer_learning is not None and transfer_learning.lower() != 'flatten' and transfer_learning.lower() != 'dense':
        raise TypeError('Wrong transfer learning choice: insert either Flatten or Dense')
    if transfer_learning is not None:
        model_fe = build_tl_model(preprocess_method, transfer_learning)
    standard_len = calculate_standard_len(preprocess_method)

    model, num_params = build_ad_model(model_name=model_name)  # BUILDING MODEL FOR ANOMALY DETECTION

    scores, labels = [], []
    # iterating on all the files of the training set directory
    for iteration, filename in enumerate(tqdm(os.listdir(train_directory))):
        path = os.path.join(train_directory, filename)

        for i in range(9):

            # audio sub windowing
            raw_features, frequency = librosa.load(path,
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

            # sometimes the length of the input is not as expected, so we handle it concatenating zeros
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

            # sample analysis for anomaly detection and training on the new sample
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
    if transfer_learning is not None:
        print("TRAINING: " + model_name + " rocauc score using "+transfer_learning+": " + str(roc))
    else:
        print("TRAINING: " + model_name + " rocauc: " + str(roc))
    scores, labels = [], []

    # same procedure as above but for the training set (now obviously the anomaly detection model won't learn)
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
    roc = roc_auc_score(labels, scores)
    if transfer_learning is not None:
        print("TESTING: " + model_name + " rocauc score using " + transfer_learning + ": " + str(roc))
    else:
        print("TESTING: " + model_name + " rocauc: " + str(roc))
