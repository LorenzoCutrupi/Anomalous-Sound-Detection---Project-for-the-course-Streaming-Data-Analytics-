# ANOMALOUS SOUND DETECTION USING MACHINE LEARNING METHODS

## Overview

This project focuses on developing an anomalous sound 
detection system using various machine learning methods. 
The objective is to detect unusual or anomalous sounds 
within a given dataset. Anomalies can be defined as sounds 
that deviate significantly from the expected or normal 
acoustic patterns. By leveraging the power of machine 
learning, this project aims to identify these anomalous 
sounds and flag them for further investigation.

## Datasets

This project utilizes multiple datasets for evaluating the anomalous sound detection methods. The datasets used are as follows:

1. **CreditCard Dataset:**
   - **Description:** A well-known dataset for anomaly detection in credit card transactions.
   - **Source:** River [2] library
   - **Size:** Only 5000 samples were used
   - **Format:** CSV

2. **Edge Impulse Dataset:**
   - **Description:** A dataset containing sounds from industrial machines collected using Edge Impulse platform.
   - **Source:** EdgeImpulse [1]
   - **Size:** 400 samples
   - **Format:** WAV

3. **Project's Dataset:**
   - **Description:** A dataset provided by the professors of the course, comprising sounds from industrial machines (Heat Pumps).
   - **Source:** Private
   - **Size:** 20000 samples
   - **Format:** WAV

## Machine Learning Methods
This project employs various machine learning methods for 
anomalous sound detection from River [2] and DeepRiver [3] libraries. The methods used and their 
configurations are as follows:

1. **Half Space Trees:**
   - **Description:** An algorithm that constructs binary trees to separate normal and anomalous instances.
2. **One-Class SVM:**
   - **Description:** A support vector machine algorithm designed for unsupervised anomaly detection.
3. **Autoencoder:**
   - **Description:** A neural network architecture used for unsupervised learning and anomaly detection.
4. **Weighted Autoencoder:**
   - **Description:** An extension of the autoencoder that incorporates weighted loss for anomaly detection.
5. **Rolling Autoencoder:**
   - **Description:**  An autoencoder variation designed for anomaly detection in sequential data (like the heat pump of the study case).
6. **ILOF:**
   - **Description:** An extension of the Local Outlier Factor (LOF) algorithm, which is a popular unsupervised anomaly detection technique.
7. **KitNET:**
   - **Description:** A lightweight unsupervised anomaly detection method based on neural networks.
8. **Robust Random Cut Forest:**
   - **Description:** An ensemble method for identifying anomalies in high-dimensional data.

## Approach for anomaly detection

In this project, the approach for anomaly detection in sound involves several steps. The following describes the overall process:
1. **Audio Segmentation:**
   - Each 10-second audio clip is divided into 9 mini-audios of 2 seconds each. This segmentation step allows for a more granular analysis of the sound data.
2. **Feature Generation:**
   - The segmented audio clips are processed to extract relevant features. Two preprocessing models, namely "mfe" and "spectrogram," are available for feature generation. It is possible to choose among these models to generate the features.
3. **Feature Extraction:**
   - A pre-trained neural network model is utilized for further feature extraction from the generated audio features. This model is designed specifically for capturing high-level representations and patterns within the audio data, and each of the feature generation models has its own extraction model.
4. **Anomaly Scoring:**
   - The extracted features are used to compute anomaly scores for each mini-audio clip. 
5. **Model Training:**
   - The anomaly detection model used for scoring the sample is now trained using the label of the mini-audio clips. The training process involves feeding the labeled data, where anomalies are identified based on known anomalous samples.

## Evaluation and Results

1. **Evaluation Metric:**
   - The Receiver Operating Characteristic Area Under the Curve (ROC AUC) is used as an evaluation metric to assess the performance of the anomaly detection system. ROC AUC measures the ability of the system to discriminate between normal and anomalous instances based on the generated anomaly scores. It provides a quantitative measure of the system's overall classification performance. 
2. **Results:**
   - The anomaly detection system demonstrated exceptional results on the CreditCard dataset (almost all methods have a score above 0.8) and EdgeImpulse dataset (where methods have a score over 0.9) but they have insufficient performance (0.56) on the project's dataset.

## Conclusions

The methods show impressive performance on well-known, 
stable and numerical dataset like the one in CreditCard: 
this means that the anomaly detection models work overall 
correctly in this context (although svm and ilof should be 
improved). Moreover, using the pipeline to extract 
meaningful information from a .wav file great performance
are achieved when applied on Edge Impulse dataset, while 
results are poor on the project's one.
This could be for multiple reasons:
- very unbalanced datasets like the one of the project 
(where anomalies were 40 samples among 20.000) could teach the
model to simply predict normal for all the samples
- the neural network used for feature extraction is not
good for all the sound anomaly detection but only for the
one it was trained on (Edge Impulse), because it looks for
different attributes based on what the anomalies of the
original dataset looked like.

## References
[1] (https://studio.edgeimpulse.com/public/88093/latest)

[2] (https://github.com/lucasczz/DAADS/tree/main/river)

[3] (https://github.com/online-ml/deep-river/tree/master/deep_river/anomaly)
