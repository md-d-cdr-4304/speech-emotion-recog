# Speech Emotion Recognition Using MFCC and SVM

This repository contains a Python-based implementation of a Speech Emotion Recognition (SER) system that uses **Mel-frequency cepstral coefficients (MFCCs)** for feature extraction and a **Support Vector Machine (SVM)** for classification. The dataset used in this project is the **RAVDESS dataset**.

## Overview

The goal of this project is to classify speech samples into different emotion categories based on their acoustic features. This system extracts MFCC features from audio files and trains an SVM classifier to predict the corresponding emotion labels.

## Dataset

The **RAVDESS dataset** is a widely used dataset for speech and song emotion recognition. The dataset includes emotional speech recordings from multiple actors and is labeled with emotion categories like calm, happy, sad, angry, etc.

The file naming format follows: `03-01-XX-XX-XX-XX-XX.wav`, where the **third component** (`XX`) represents the emotion label.

### Emotion Mapping (as per dataset):
- 01: Neutral
- 02: Calm
- 03: Happy
- 04: Sad
- 05: Angry
- 06: Fearful
- 07: Disgust
- 08: Surprised

## Feature Extraction

We use **MFCC (Mel-frequency cepstral coefficients)** to extract features from the audio files:
- Each audio file is loaded using `librosa`.
- MFCCs are computed with a default of 40 coefficients.
- The mean value of each MFCC is used to represent the features of an audio sample.

## Model

A **Support Vector Machine (SVM)** classifier with a linear kernel is used to train on the MFCC features. The model is evaluated on a test set to predict emotion labels and compute the following metrics:
- **Accuracy**
- **F1 Score (Weighted)**

## Code

### Dependencies
- `numpy`
- `librosa`
- `scikit-learn`

Install the required packages using:
```bash
pip install numpy librosa scikit-learn
