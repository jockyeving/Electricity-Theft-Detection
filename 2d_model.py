# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 23:48:28 2021

@author: jocky
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
from sklearn import metrics
from keras import backend as K
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.experimental import WideDeepModel, LinearModel
from tensorflow.keras.callbacks import TensorBoard
import keras
import time


import load_module

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


#Configuration
convolutional_layers = 2
filters = 40
kernel_shape = (2,2)
undersampling = True
test_size = 0.22
epochs=2

x_2d_train, x_2d_test, y_train, y_test = load_module.load_data(undersampling,test_size)
x_2d_train  = x_2d_train.reshape(list(x_2d_train.shape) + [1])
x_2d_test  = x_2d_test.reshape(list(x_2d_test.shape) + [1])


## Logging
#%load_ext tensorboard
#log_dir='deep_cnn-{}'.format(int(time.time()))
#tensorboard_callback = TensorBoard(log_dir='logs/{}'.format(log_dir), histogram_freq=1)
#conda prompt(tf-gpu->) tensorboard --logdir=logs/     (change dir to current!)


log_dir='deep_cnn-{}'.format(int(time.time()))
deep_callback = TensorBoard(log_dir='logs/{}'.format(log_dir), histogram_freq=1)




dnn_model = keras.Sequential()
#for n in range(convolutional_layers):
    #dnn_model.add(layers.Conv2D(filters=filters, kernel_size=kernel_shape, activation='relu'))
dnn_model.add(layers.Conv2D(filters=filters, kernel_size=kernel_shape, input_shape=x_2d_train.shape[1:], activation='relu'))
dnn_model.add(layers.MaxPooling2D(pool_size=(2,2)))
dnn_model.add(layers.Conv2D(filters=filters*2, kernel_size=kernel_shape, activation='relu'))
dnn_model.add(layers.MaxPooling2D(pool_size=(2,2)))

dnn_model.add(layers.Flatten())
dnn_model.add(layers.Dense(units=35,activation='relu'))
dnn_model.add(layers.Dense(units=1,activation='sigmoid'))

dnn_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['AUC', f1])
#dnn_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=[tf.keras.metrics.AUC(),tfa.metrics.F1Score(num_classes=2,threshold=0.5)])
dnn_model.fit(x_2d_train, y_train, epochs=epochs,validation_data=(x_2d_test, y_test),callbacks = [deep_callback])
dnn_model.summary()






#probs = dnn_model.predict(x_2d_test)

#max_array = []
#auc_array = []
#f1_array = []
#probs_array = []
#for i in range(20):
#    dnn_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['AUC', f1])
#    dnn_model.fit(x_2d_train, y_train, epochs=epochs,validation_data=(x_2d_test, y_test),callbacks = [deep_callback])
#    probs = dnn_model.predict(x_2d_test)
#    probs_array.append(probs)
#    f1_scores = []
#    for j in range(1,50):
#        threshold = j/50
#        predictions = []
#        for i in range(len(probs)):
#            predictions.append(bool(probs[i]>threshold))
#        f1_scores.append(sklearn.metrics.f1_score(y_test,predictions))
#    maximum = max(f1_scores)
#    print(f"\nMaximum f1_score: {maximum}")
#    max_array.append(maximum)
#    f1_array.append(f1_scores)
#    auc_array.append(sklearn.metrics.roc_auc_score(y_test,probs,average = "macro"))
#maximum = max(max_array)
#maximum_auc = max(auc_array)
#print(f"\nMaximum f1_score: {maximum}")
#print(f"\nMaximum AUC_score: {maximum_auc}")
#log_dir='wide_deep_cnn-{}'.format(int(time.time()))
#wide_deep_callback = TensorBoard(log_dir='logs/{}'.format(log_dir), histogram_freq=1)
#combined_model = WideDeepModel(linear_model, dnn_model,activation = 'sigmoid')
#combined_model.compile(optimizer=['sgd', 'adam'],loss='binary_crossentropy', metrics='AUC')
#combined_model.fit([x_1d_train, x_2d_train], y_train,validation_data=([x_1d_test, x_2d_test], y_test), callbacks = [wide_deep_callback], epochs=30)
#loss, auc = combined_model.evaluate([x_1d_test, x_2d_test], y_test, callbacks = [tensorboard_callback], verbose=1)