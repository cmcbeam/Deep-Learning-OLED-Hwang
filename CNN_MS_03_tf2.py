# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 13:46:03 2020

@author: Hwang
"""

import tensorflow as tf
import os
import tensorflow.keras as keras

import numpy as np
import os.path
import math
import pickle
from os import rename, listdir
import random
import pandas as pd
from sklearn.metrics import accuracy_score
# np.set_printoptions(threshold=100000)
# nowdir = os.getcwd()
# def xavier_init(n_inputs, n_outputs, uniform=True):
#     if uniform:
#         init_range = math.sqrt(6.0 / (n_inputs + n_outputs)/2.0)
#         return tf.random_uniform_initializer(-init_range, init_range)
#     else:
#         stddev = math.sqrt(3.0 / (n_inputs + n_outputs)/2.0)
#         return tf.truncated_normal_initializer(stddev=stddev)

def xavier_init(n_inputs, n_outputs, uniform=True):
    if uniform:
        init_range = math.sqrt(6.0 / (n_inputs + n_outputs)/2.0)
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = math.sqrt(3.0 / (n_inputs + n_outputs)/2.0)
        return tf.compat.v1.truncated_normal_initializer(stddev=stddev)
    
def search(path):
    res = []
    for root, dirs, files in os.walk(path):
        rootpath = os.path.join(os.path.abspath(path), root)
        
        for file in files:
            filepath = os.path.join(rootpath, file)
            
            res.append(filepath)
        #for ress in res:
        #    print ress
    return res

    # prefix = raw_input ('What is the prefix? (enter is none) ')
# hyper parameters
learning_rate = 0.0001
training_epochs = 5
batch_size = 10
dr = 0.7
log_file = open('log.txt', 'w')

#xy2 = np.loadtxt('training_set.txt', unpack=True, dtype='float32')
training_file = open('training_set2.dmp', 'rb')
xy = pickle.load(training_file)

Y_file = open('training_Y2.dmp', 'rb')
y = pickle.load(Y_file)

# decoy_file = open('decoy set.dmp', 'rb')
# decoy_data_ = pickle.load(decoy_file)

# test_file = open('validation set.dmp', 'rb')
# x_test_ = pickle.load(test_file)

# Y_test_file = open('validation Y.dmp', 'rb')
# y_test_ = pickle.load(Y_test_file)



# Y_test_file2 = open('test_Y.dmp', 'rb')
# y_test_2 = pickle.load(Y_test_file2)


# test_name = open('test name_4.dmp','rb')
# test_name_list = pickle.load(test_name)
# training_name = open('training name_4.dmp','rb')
# training_name_list = pickle.load(training_name)

#print(xy[0][-3:])
#x_data = np.transpose(np.array(xy[:-3]))
#y_data = np.transpose(np.array(xy[-3:]))
#print(y_data[0][:3])

#y2_data = np.transpose(xy2[-3:])
#print(y2_data[0])

#for xy_ in xy:
#    x_data_.append(xy_)
    #y_data_.append(xy_[-3:])

x_data = np.array(xy)
y_data = np.array(y)
# x_test = np.array(x_test_)
# y_test = np.array(y_test_)

# y_test = np.array(y_test_2)

# decoy_data = np.array(decoy_data_)


# x_data = np.squeeze(x_data)
# x_test = np.squeeze(x_test)
# x_test2 = np.squeeze(x_test2)


x_data = x_data.reshape(x_data.shape[0], 1 ,2964,1)
# x_test = x_test.reshape(x_test.shape[0], 1 ,2964,1)

# #print(np.shape(x_data))
# #print(np.shape(y_data))
# #print(np.shape(x_test))
# #print(np.shape(y_test))
# #print(np.shape(decoy_data))
# #print(len(xy), len(x_data), len(y_data))
# #print(len(x_data[2]))
# # input place holders
# #print(len(xy))
filter_depth = 300
print(np.shape(x_data))
print(np.shape(y_data))
# print(np.shape(x_test))
# print(np.shape(y_test))
# print(np.shape(decoy_data))
print(len(xy), len(x_data), len(y_data))
print(len(x_data[2]))
# # input place holders
# #print(len(xy))




for i in range(10):
    def create_model():
        model = tf.keras.models.Sequential([
          tf.keras.layers.Conv2D(filters=filter_depth, kernel_size=(1,20), input_shape = (1,2964, 1), strides =(1,5),padding='same'),
          tf.keras.layers.ReLU(),
          tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1,2), padding='same'),
          tf.keras.layers.Conv2D(filters=filter_depth, kernel_size=(1,5),padding='same'),
          tf.keras.layers.ReLU(),
          tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1,2), padding='same'),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(units=500, kernel_initializer=xavier_init(200,500)),
          tf.keras.layers.ReLU(),
          tf.keras.layers.Dense(units=500, kernel_initializer=xavier_init(500,500)),
          tf.keras.layers.ReLU(),
          tf.keras.layers.Dense(units=500, kernel_initializer=xavier_init(500,500)),
          tf.keras.layers.ReLU(),
          tf.keras.layers.Dense(units=11, kernel_initializer=xavier_init(500,11),activation='softmax')
          ])
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Nadam(lr=learning_rate), metrics=['accuracy'])
        return model


    model = create_model()
    model.summary()
    model.fit(x_data, 
              y_data,
                # batch_size = batch_size,
              # validation_batch_size = batch_size,
              epochs=training_epochs)
    model.save(str(i) + '_CNN_model.h5')
    


    train_preds = np.where(model.predict(x_data) > 0.5, 1, 0)
    #test_preds = np.where(model.predict(x_test) > 0.5, 1, 0)
    
    train_accuracy = accuracy_score(y_data, train_preds)
    #test_accuracy = accuracy_score(y_test, test_preds)
    
    print('model ' + str(i), file = log_file)
    print(f'Train Accuracy : {train_accuracy:.4f}', file = log_file)
    #print(f'Test Accuracy  : {test_accuracy:.4f}', file = log_file)
    
    
    #evaluation = model.(x_test, y_test)
    #print('loss: ', evaluation[0])
    #print('Model ' + str(i) + ' accuracy', evaluation[1], file = log_file)
    
    # prediction = model.predict(x_data)
    
    # #pred_df = pd.DataFrame(prediction, columns = ['Decoy', 'Herceptin', 'Remicade', 'Humira' ,'Avastin', 'Herzuma', 'Remsima'])
    # pred_df = pd.DataFrame(prediction, columns = ['10:0', '9:1', '8:2', '7:3', '6:4', '5:5', '4:6', '3:7', '2:8', '1:9', '0:10'])
    
    # pred_df.to_excel(str(i) + '_model_test_.xlsx')
    nowdir = os.getcwd()
    file_list = search(nowdir)
    for file_ in file_list:
        if "_test_set2.dmp" in file_:
            test2_file = open(file_, 'rb')
            test2_df = pd.read_excel(file_[:-5] + '.xlsx')
            files = file_.split('\\')
            filenames = list(test2_df.columns)
            x_test_2 = pickle.load(test2_file)
            x_test = np.array(x_test_2)
            x_test2 = x_test.reshape(x_test.shape[0], 1 ,2964,1)
            prediction = model.predict(x_test2)
            pred_df = pd.DataFrame(prediction, columns = ['10:0', '9:1', '8:2', '7:3', '6:4', '5:5', '4:6', '3:7', '2:8', '1:9', '0:10'], index = filenames[1:])
    
            pred_df.to_excel(str(i) + files[-1][:-4]+'_model_test_.xlsx')
    #print(f'Test Accuracy  : {test_accuracy:.4f}', file = log_file)
    
# print(prediction[50])
# print(y_test2[50])



