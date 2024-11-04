# 1) Data Preprocessing

#import os
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
#print(tf.__version__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input,GlobalMaxPooling1D
from tensorflow.keras.layers import LSTM, Embedding
from tensorflow.keras.models import Model

df = pd.read_csv('D:/MyProject/Naghme/NLP/Text Classification with CNN/spam.csv', encoding='ISO-8859-1')
print(df.head())

df = df.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"], axis=1)
print(df.head())

df.columns = ['labels','data']
print(df.head())

# create binary labels (0 and 1)
df['b_labels'] = df['labels'].map({'ham' : 0 , 'spam' : 1}) # create new column as 'b_labels'
y = df['b_labels'].values

print(y)

# split the data
x_train, x_test, y_train, y_test = train_test_split(df['data'], y, test_size=0.33)

max_vocab_size = 20000
tokenizer = Tokenizer(num_words=max_vocab_size)

tokenizer.fit_on_texts(x_train)

sequences_train = tokenizer.texts_to_sequences(x_train)
sequences_test = tokenizer.texts_to_sequences(x_test)

print(x_train[0])

print(len(sequences_train[0]))

print(len(sequences_train[1]))

#convert sentences too sequences
max_vocab_size = 20000
tokenizer = Tokenizer(num_words=max_vocab_size)
tokenizer.fit_on_texts(x_train)
sequences_train = tokenizer.texts_to_sequences(x_train)
sequences_test = tokenizer.texts_to_sequences(x_test)

# check word index mapping (to check the number of words in vocabulary)
word2idx = tokenizer.word_index
V = len(word2idx)
print('Total number of unique tokens are: %s' %V)

# pad sequences (to get N x T matrix)
data_train = pad_sequences(sequences_train)
print('Shape of data train tensor:', data_train.shape)

# N : number of samples and T : Number of time steps

print(data_train[0])
print(len(data_train[0]))

print(len(data_train[1]))

# set the value of T to get sequence lenght
T = data_train.shape[1]
print(T)

# pad the test set
data_test = pad_sequences(sequences_test, maxlen=T)
# maxlen = T, to truncate longer sequences in test set
print('Shape of data test tensor:', data_test.shape)

# Create the model

# Choose embedding dimensionality

D = 20 # this is a hyper parameter, we can choose any word vector size that want

# Hidden state vectorize (dimensionality)
M = 15

# Input layer
i = Input(shape=(T,)) # input layer takes in sequences of integers, so shape is T

# Embedding layer
x = Embedding(V+1,D)(i) # This takes in sequences of integers and returns sequences of word vectors
# This will be an N * T * D array
# we want size of embedding to (V+1) x D, because first word index starts from 1 and not 0

# LSTM layer
x = LSTM(M, return_sequences=True)(x)
x = GlobalMaxPooling1D()(x)
 
from tensorflow.keras.layers import Flatten

# Flatten the output before passing it to the Dense layer
x = Flatten()(x)

# Dense layer
x = Dense(1, activation='sigmoid')(x)
# it is a binary classification problem, so we are using activation function = 'sigmoid'

model = Model(i,x)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
r = model.fit(x=data_train, y=y_train, epochs=10, validation_data=(data_test, y_test))

# Loss per iteration
import matplotlib.pyplot as plt
plt.plot(r.history['loss'], label='Loss')
plt.plot(r.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Accuracy per iteration
plt.plot(r.history['accuracy'], label='Accuracy')
plt.plot(r.history['val_accuracy'], label='Validation accuracy')
plt.legend()
plt.show()


