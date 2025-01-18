# import required packages
import tensorflow as tf
import tensorflow_datasets as tfds  # library to import the IMDB dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

if __name__ == "__main__": 
	# 1. Load your saved model
    model = tf.keras.models.load_model(".\\models\\20833793_NLP_model.h5")

	# 2. Load your testing data
	# downloading the dataset
    train_data, test_data = tfds.load(name='imdb_reviews', split=('train', 'test'), as_supervised=True)

    # creating empty list for training set
    train_sent = []
    train_out = []

    # creating empty list for test set
    test_sent = []
    test_out = []

    # for loop to obtain train set from the downloaded dataset
    for s, l in train_data:
        train_sent.append(str(s.numpy()))
        train_out.append(l.numpy())

    # for loop to obtain test set from the downloaded dataset
    for s, l in test_data:
        test_sent.append(str(s.numpy()))
        test_out.append(l.numpy())

    # converting the label dataset to array formate
    train_out = np.array(train_out)
    test_out = np.array(test_out)

    # hyper parameters - writing them at one place so that it's easy to modify them in future
    vocab_size = 20000  # 
    embedding_dim = 128   
    max_length = 400     
    trunc_type = 'post'
    oov_tok = "<OOV>"
    num_epochs = 10

    # initializing the tokenizer using hyper parameters.
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok) # number of words for the number of words to tokenize, oov_token is for unknown words 
    tokenizer.fit_on_texts(train_sent)  # applying the tokenizer on the input texts
    word_index = tokenizer.word_index   # obtaining the word index
    sequences = tokenizer.texts_to_sequences(train_sent)    # converting the input texts to their relative word_indexes
    padded = pad_sequences(sequences,maxlen=max_length,truncating=trunc_type)   # padding to make all the input texts of the same length
    test_sent = tokenizer.texts_to_sequences(test_sent) #tokenizing the test dataset
    test_padded = pad_sequences(test_sent,maxlen=max_length)    # padding the test dataset

	# 3. Run prediction on the test data and print the test accuracy
    test_loss, test_acc = model.evaluate(test_padded,  test_out)

    print("The final test accuracy is: ",test_acc)
    print("The final test loss is: ",test_loss)