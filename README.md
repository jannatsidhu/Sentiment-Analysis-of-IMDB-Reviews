# Sentiment Analysis of IMDB Reviews

This project implements sentiment analysis on the IMDB movie reviews dataset using Natural Language Processing (NLP) techniques and neural networks.

## Overview

The project includes:
1. **Data Preprocessing**: Tokenization, padding, and preparing the dataset for training and testing.
2. **Model Design**: Building a Convolutional Neural Network (CNN) combined with a Bidirectional GRU layer.
3. **Training and Evaluation**: Training the model on the training dataset and evaluating its performance on the test dataset.
4. **Prediction**: Using the trained model to predict sentiment labels for the test data.

## Dataset

The dataset used is the **IMDB Reviews** dataset, a collection of 50,000 movie reviews labeled as positive or negative. This dataset is available as part of the [TensorFlow Datasets library](https://www.tensorflow.org/datasets).

## Features

- **Text Tokenization and Padding**:
  - Tokenizes the reviews into sequences of word indices.
  - Pads sequences to ensure uniform input length for the model.
- **Embedding Layer**: Learns a dense vector representation for the input words.
- **Conv1D Layer**: Extracts features from sequences.
- **Bidirectional GRU Layer**: Captures temporal dependencies in both directions.
- **Dense Layers**: Final layers for classification.

## Requirements

- Python 3.8+
- Libraries:
  - TensorFlow
  - NumPy
  - Matplotlib
  - TensorFlow Datasets
  - NLTK


## Code Structure

### Files:
- **`train_NLP.py`**:
  - Prepares and preprocesses the IMDB dataset for training.
  - Implements a Convolutional Neural Network (CNN) combined with Bidirectional GRU.
  - Trains the model using the processed dataset.
  - Saves the trained model for later use.

- **`test_NLP.py`**:
  - Loads the trained model.
  - Prepares the test data.
  - Evaluates the model's performance on unseen test data.

### Workflow:
1. Preprocess the dataset using tokenization and padding.
2. Train the neural network using the `train_NLP.py` script.
3. Evaluate the trained model using the `test_NLP.py` script.

---

## Model Architecture

1. **Embedding Layer**:
   - Converts words into dense vector representations.

2. **Convolutional Layer (Conv1D)**:
   - Extracts features from sequential data.

3. **Bidirectional GRU**:
   - Processes the sequence in both forward and backward directions to capture temporal dependencies.

4. **Fully Connected Layers**:
   - Dense layers process the extracted features.
   - The final layer uses a sigmoid activation function for binary classification.

---

## Training Details

### Hyperparameters:
- **Vocabulary Size**: 20,000
- **Embedding Dimensions**: 128
- **Maximum Sequence Length**: 400
- **Truncation Type**: Post
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Epochs**: 2

---

## Evaluation Metrics

The model is evaluated using:
- **Accuracy**: Percentage of correct predictions.
- **Loss**: Binary crossentropy loss function.

---

## Future Work

To further enhance this project, the following improvements can be made:

Experiment with other architectures like LSTMs, Transformers, or pre-trained models like BERT.
Apply hyperparameter optimization for better accuracy.
Explore data augmentation techniques to improve model robustness.

