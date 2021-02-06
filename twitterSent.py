# read the data
# change to list
# tokenize using tensorflow Tokenizer().fit_on_text()
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, Conv1D, GlobalMaxPool1D, MaxPool1D, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.utils import resample

# load data
# data from https://raw.githubusercontent.com/laxmimerit/twitter-data/master/twitter4000.csv
data = pd.read_csv('/home/tim/Datasets/twitter4000.csv')

# data .to_list()
text = data['twitts'].to_list()

y = data.sentiment

# tokenize
token = Tokenizer()
token.fit_on_texts(text)

vocab_size = len(token.word_index) + 1
encoded_text = token.texts_to_sequences(text)

max_length = 120
X = pad_sequences(encoded_text, maxlen=max_length, padding='post')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)

vec_size = 200

model = Sequential()
model.add(Embedding(vocab_size, vec_size, input_length=max_length, input_shape=(X_train.shape[1], )))

model.add(Dropout(0.3))

"""
model.add(Conv1D(32, 8, activation='relu'))
model.add(MaxPool1D(2))
model.add(Dropout(0.5))
"""

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(GlobalMaxPool1D())
model.add(Dropout(0.5))


model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
done = input('save? ')
if done == "yes":
    model.save('/home/tim/trained/tweets2.h5')
else:
    pass
