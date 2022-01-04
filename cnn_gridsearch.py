# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 17:36:18 2021

@author: jocky
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 23:48:28 2021

@author: jocky
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
from sklearn.metrics import roc_auc_score, f1_score
from keras import backend as K
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.experimental import WideDeepModel, LinearModel
from tensorflow.keras.callbacks import TensorBoard
import keras
import time


import load_module

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


#Configuration
convolutional_layers = 2
filter_array = [25, 30, 35, 40]
test_size_array = [0.18, 0.20, 0.22]
dense_array = [25, 30, 35, 40]

save_dir = 'C:/Users/jocky/OneDrive/Dokumentumok/egyetem/onlab_2/sgcc_data/data/wide_deep/gridsearch_results/'
undersampling = True
epochs=30
run_count = 20

f1_results = []
auc_results = []
labels = []
max_f1s = []
max_auc = []
probs_vector = []

for test_size in test_size_array:
    x_2d_train, x_2d_test, y_train, y_test = load_module.load_data(undersampling,test_size)
    x_2d_train  = x_2d_train.reshape(list(x_2d_train.shape) + [1])
    x_2d_test  = x_2d_test.reshape(list(x_2d_test.shape) + [1])
    for filters in filter_array:
        for dense_size in dense_array:
            for runs in range(run_count):    
                dnn_model_g = keras.Sequential()
                dnn_model_g.add(layers.Conv2D(filters=filters, kernel_size=(2,2), input_shape=x_2d_train.shape[1:], activation='relu'))
                dnn_model_g.add(layers.MaxPooling2D(pool_size=(2,2)))
                dnn_model_g.add(layers.Conv2D(filters=filters*2, kernel_size=(2,2), activation='relu'))
                dnn_model_g.add(layers.MaxPooling2D(pool_size=(2,2)))
                dnn_model_g.add(layers.Flatten())
                dnn_model_g.add(layers.Dense(units=dense_size,activation='relu'))
                dnn_model_g.add(layers.Dense(units=1,activation='sigmoid'))
                
                dnn_model_c = keras.Sequential()
                dnn_model_c.add(layers.Conv2D(filters=filters, kernel_size=(3,3), input_shape=x_2d_train.shape[1:], activation='relu'))
                dnn_model_c.add(layers.MaxPooling2D(pool_size=(2,2)))
                dnn_model_c.add(layers.Flatten())
                dnn_model_c.add(layers.Dense(units=dense_size,activation='relu'))
                dnn_model_c.add(layers.Dense(units=1,activation='sigmoid'))
                
                model_array = [dnn_model_c, dnn_model_g]
                for i in range(len(model_array)):
                    print("\nNext Model")
                    dnn_model = model_array[i]
                    if(i == 0):
                        model_type = 'c'
                    if(i == 1):
                        model_type = 'g'
                    model_label = "{}-model-{}-dense-{}-filter-{}-testsize-{}".format(model_type,dense_size,filters,test_size,int(time.time()))
                    dnn_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['AUC', f1])                            
                    
                    f1_ofmodel = np.zeros(shape=(epochs,100))
                    auc_ofmodel = np.zeros(shape=epochs)
                    probs_ofmodel = []
                    for epoch in range(epochs):
                        curr_epoch = epoch + 1
                        print("Epoch# {}".format(curr_epoch))
                        dnn_model.fit(x_2d_train, y_train, epochs=1,validation_data=(x_2d_test, y_test))
                        probs = dnn_model.predict(x_2d_test)
                        probs_ofmodel.append(probs)
                        auc_ofmodel[epoch] = roc_auc_score(y_test,probs,average = "macro")
                        for threshold in range(100):
                            predictions = []
                            for prob in range(len(probs)):
                                predictions.append(bool(probs[prob]>(threshold/100)))
                            curr_f1 = f1_score(y_test,predictions)
                            f1_ofmodel[epoch,threshold] = curr_f1
                    probs_vector.append(probs_ofmodel)
                    f1_results.append(f1_ofmodel)
                    auc_results.append(auc_ofmodel)
                    labels.append(model_label)
                    max_f1s.append(np.amax(f1_ofmodel))
                    max_auc.append(np.amax(auc_ofmodel))
                    #np.save(save_dir+"{}-model-{}-dense-{}-filter-{}-testsize-{}-f1".format(model_type,dense_size,filters,test_size,int(time.time())),f1_ofmodel)
                    #np.save(save_dir+"{}-model-{}-dense-{}-filter-{}-testsize-{}-auc".format(model_type,dense_size,filters,test_size,int(time.time())),auc_ofmodel)
               


    