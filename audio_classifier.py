# To solve this assignmentm, I made use of this gihub repo:
# https://github.com/jaron/deep-listening

import os
import pandas as pd
import numpy as np
import librosa
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, Adagrad
from keras.callbacks import EarlyStopping
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
from keras.utils import np_utils
from keras.regularizers import l2


def get_file_paths(root_dir):
    res = list()
    try:
        for dirname, subdirlist, filelist in os.walk(root_dir):
            for fname in filelist:
                class_name = fname.split(".")[0].split("-")[1]
                fpath = os.path.join(dirname, fname)
                res.append((fpath, class_name))
    except:
        pass
    return res


root_dir = './data'

all_labeled_paths = get_file_paths(root_dir)

np.random.seed(42)
np.random.shuffle(all_labeled_paths)
split_point = int(len(all_labeled_paths) * 0.8)
labeled_train_paths = all_labeled_paths[:split_point]
labeled_test_paths = all_labeled_paths[split_point:]
print(len(labeled_train_paths))
print(len(labeled_test_paths))

"""

feature extraction¶
I used librosa library to extract various features from wave files. the features are log sacled mel-spectrograms with their deltas which come from two windows and sxtract_features methods.
I had to install libav-tools to solve the backend error librosa was throwing

"""


def windows(data, window_size):
    start = 0
    while start < len(data):
        yield int(start), int(start + window_size)
        start += (window_size / 2)


def extract_features(labeled_paths, file_ext="*.wav", bands=60, frames=41):
    window_size = 512 * (frames - 1)
    log_specgrams = []
    labels = []
    for fname, label in labeled_paths:
        print(fname)
        sound_clip, s = librosa.load(fname)
        for (start, end) in windows(sound_clip, window_size):
            if (len(sound_clip[start:end]) == int(window_size)):
                signal = sound_clip[start:end]
                melspec = librosa.feature.melspectrogram(signal, n_mels=bands)
                logspec = librosa.amplitude_to_db(melspec)
                logspec = logspec.T.flatten()[:, np.newaxis].T
                log_specgrams.append(logspec)
                labels.append(label)

    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams), bands, frames, 1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])

    return np.array(features), np.array(labels, dtype=np.int)


def one_hot_encode(labels):
    encoded_labels = pd.get_dummies(labels)

    return encoded_labels


train_features, train_labels = extract_features(labeled_train_paths)
train_labels = one_hot_encode(train_labels)
test_features, test_labels = extract_features(labeled_test_paths)

"""

train a convolutional neural network
I trained a convolutional neural network (CNN)with two Convolution2D layers, the first with 96 output filters, the second with 64 .
I also used dropout layers to prevent overfitting and a Max Pooling laye
"""

tf.set_random_seed(0)
np.random.seed(0)

frames = 41
bands = 60
feature_size = bands * frames  # 60x41
num_labels = 10
num_channels = 2


def evaluate(model):
    y_prob = model.predict_proba(test_features, verbose=0)
    y_pred = y_prob.argmax(axis=1)
    accuracy = accuracy_score(test_labels, y_pred)
    print("accuracy score: {}".format(accuracy))


def build_model():
    model = Sequential()
    f_size = 1

    # first layer has 48 convolution filters
    model.add(Convolution2D(48, f_size, strides=f_size, kernel_initializer='normal', padding='same',
                            input_shape=(bands, frames, num_channels)))
    model.add(Convolution2D(48, f_size, strides=f_size, kernel_initializer='normal', padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    # next layer has 96 convolution filters
    model.add(Convolution2D(96, f_size, strides=f_size, kernel_initializer='normal', padding='same'))
    model.add(Convolution2D(96, f_size, strides=f_size, kernel_initializer='normal', padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    # flatten output into a single dimension
    # Keras will do shape inference automatically
    model.add(Flatten())

    # then a fully connected NN layer
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # finally, an output layer with one node per class
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))
    print(model.summary())
    # use the Adam optimiser
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

    return model


model = build_model()
model.fit(train_features, train_labels, epochs=30, batch_size=64)
evaluate(model)
# I got the accuracy score of 0.592 for the above model
