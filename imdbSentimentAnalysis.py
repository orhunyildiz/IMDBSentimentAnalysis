# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from keras.datasets import imdb                                           # dataset
from keras.preprocessing.sequence import pad_sequences                    # kerasa input olarak verilecek verisetinin boyutunun aynı olması gerek. bunun için düzenliyoruz
from keras.models import Sequential                                       # model
from keras.layers.embeddings import Embedding                             # integerları yoğunluk vektörüne çevirir
from keras.layers import SimpleRNN, Dense, Activation, Dropout            # RNN, flatten, sigmoid

(X_train, Y_train), (X_test, Y_test) = imdb.load_data(path = "imdb.npz",
                                                      num_words = None,   # kelime sayısı (en çok kullanılan None kelime)
                                                      skip_top = 0,       # sık kullanılan kelimeler göz ardı edilecek mi
                                                      maxlen = None,
                                                      seed = 113,         # random
                                                      start_char = 1,
                                                      oov_char = 2,
                                                      index_from = 3)     # -> returning tuple
print("Type: ", type(X_train))                                            # -> np array

# %% EDA

print("Y train values: ", np.unique(Y_train))    # 0 negative 1 positive
print("Y test values: ", np.unique(Y_test))      # 0 negative 1 positive

unique, counts = np.unique(Y_train, return_counts = True)
print("Y train distribution: ", dict(zip(unique, counts)))

unique, counts = np.unique(Y_test, return_counts = True)
print("Y test distribution: ", dict(zip(unique, counts)))


plt.figure()
sns.countplot(Y_train)
plt.xlabel("Classes")
plt.ylabel("Frequency")
plt.title("Y train")
plt.show()

plt.figure()
sns.countplot(Y_test)
plt.xlabel("Classes")
plt.ylabel("Frequency")
plt.title("Y test")
plt.show()

d = X_train[0]
print(d)
print(len(d))

review_len_train = []
review_len_test = []

for i, ii in zip(X_train, X_test): # yorumların uzunlukları
    review_len_train.append(len(i))
    review_len_test.append(len(ii))

sns.distplot(review_len_train, hist_kws = {"alpha": 0.3})
sns.distplot(review_len_test, hist_kws = {"alpha": 0.3})

print("Train/Test mean: ", np.mean(review_len_train))
print("Train/Test median: ", np.median(review_len_train))
print("Train/Test mode: ", stats.mode(review_len_train))


# number of words

word_index = imdb.get_word_index()
print(type(word_index))
print(len(word_index))

for keys, values in word_index.items(): # dict: values -> numbers, keys -> words
    if values == 111:
        print(keys)
        
        
def whatItSays(index = 24):
    reverse_index = dict([(value, key) for (key, value) in word_index.items()])            # values to keys transformation
    decode_review = " ".join([reverse_index.get(i - 3, "!") for i in X_train[index]])      # x train indexi içerisinde ünlemleri alıyoruz. concat
    print(decode_review)
    print(Y_train[index])
    return decode_review


decoded_review = whatItSays()

# %% Preprocessing

num_words = 15000
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=num_words)

maxlen = 130
X_train = pad_sequences(X_train, maxlen = maxlen)
X_test = pad_sequences(X_test, maxlen = maxlen)

print(X_train[5])

for i in X_train[0:10]:
    print(len(i))

decoded_review = whatItSays(5)

# %% Construct RNN

rnn = Sequential()
rnn.add(Embedding(num_words, 32, input_length = maxlen))
#rnn.add(Dropout(0.25))
rnn.add(SimpleRNN(16, input_shape = (num_words, maxlen), return_sequences = False, activation = "relu"))
#rnn.add(Dropout(0.25))
rnn.add(Dense(1))
#rnn.add(Dropout(0.25))
rnn.add(Dense(1, activation = "sigmoid"))

print(rnn.summary()) # BU SATIRDAN DOLAYI HATA ALIYORUM. BURAYI COMMENTE ALDIĞIMDA FARKLI BİR HATA ALIYORUM.
rnn.compile(loss = "binary_crossentropy", optimizer = "rmsprop", metrics = ["accuracy"])

# %% Training

history = rnn.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = 5, batch_size = 128, verbose = 1)

# %% Evaluating

score = rnn.evaluate(X_test, Y_test)
print("Accuracy: %", score[1] * 100)

plt.figure()
plt.plot(history.history["accuracy"], label = "Train")
plt.plot(history.history["val_accuracy"], label = "Test")
plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history["loss"], label = "Train")
plt.plot(history.history["val_loss"], label = "Test")
plt.title("Loss")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()

# %% Classification Report

from sklearn.metrics import classification_report, accuracy_score

def full_report(model, x, y_true, batch_size = 128):
    y_pred = model.predict_classes(x, batch_size = batch_size)
    print("Accuracy: " + str(accuracy_score(y_true, y_pred)))
    print("")
    print("Classification Report")
    print(classification_report(y_true, y_pred, digits = 5))
    
full_report(rnn, X_test, Y_test)