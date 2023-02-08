# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 15:54:48 2022

@author: Adam
"""

import os

import sys
from matplotlib import pyplot
import tensorflow
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
import time
import numpy as np

# plot diagnostic learning curves
def summarize_diagnostics(history,filename):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	#filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()


acc_1 = []
acc_2 = []
acc_3 = []
acc_4 = []
acc_5 = []
acc_10 = []
acc_15 = []
acc_20 = []
acc_40 = []
acc_60 = []
acc_80 = []
acc_100 = []
warray_1_l2 = np.empty((0,9), dtype=float)
warray_2_l2 = np.empty((0,9), dtype=float)
warray_3_l2 = np.empty((0,9), dtype=float)
warray_4_l2 = np.empty((0,9), dtype=float)
warray_5_l2 = np.empty((0,9), dtype=float)
warray_10_l2 = np.empty((0,9), dtype=float)
warray_15_l2 = np.empty((0,9), dtype=float)
warray_20 = np.empty((0,9), dtype=float)
warray_40 = np.empty((0,9), dtype=float)
warray_60 = np.empty((0,9), dtype=float)
warray_80 = np.empty((0,9), dtype=float)
warray_100 = np.empty((0,9), dtype=float)
warray_150 = np.empty((0,9), dtype=float)
warray_200 = np.empty((0,9), dtype=float)
warray_20_l2 = np.empty((0,9), dtype=float)
warray_40_l2 = np.empty((0,9), dtype=float)
warray_60_l2 = np.empty((0,9), dtype=float)
warray_80_l2 = np.empty((0,9), dtype=float)
warray_100_l2 = np.empty((0,9), dtype=float)
warray_150_l2 = np.empty((0,9), dtype=float)
warray_200_l2 = np.empty((0,9), dtype=float)
warray_1_l3= np.empty((0,9), dtype=float)


batch_size = 128
num_classes = 10
epochs = 20

# Input image dimensions
img_rows = 28
img_cols = 28

# The data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

x_train = x_train.reshape( ( len(x_train), len(x_train[0]), len(x_train[0][0]), 1 ) )
x_test = x_test.reshape( ( len(x_test), len(x_test[0]), len(x_test[0][0]), 1 ) )

# Transform RGB values into [0,1] range
x_train = x_train / 255
x_test = x_test / 255

print('x_train_shape:', x_train.shape, '\n')
print(len(x_train), 'train samples\n')
print(len(x_test), 'test samples\n')

# One-hot encoding of outputs
y_train = tensorflow.keras.utils.to_categorical( y_train )
y_test = tensorflow.keras.utils.to_categorical( y_test )

for i in range(21):
    print(i)
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), padding = "same",activation='relu', input_shape=x_train[0, :, :, :].shape))
    model.add(layers.MaxPooling2D((2, 2),padding="same"))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # compile model
    model.compile( optimizer = "adam", loss = "categorical_crossentropy", metrics = [ "accuracy" ] ) 
    
    # fit model
    
    acc_1 = []
    acc_2 = []
    acc_3 = []
    acc_4 = []
    acc_5 = []
    acc_10 = []
    acc_15 = []
    acc_20 = []
    acc_40 = []
    acc_60 = []
    acc_100 = []
    acc_200 = []
    
    #test weight retrieval
    weight = model.layers[0].get_weights()[0]
    print("Finished testing weight retrieval");
    
    # 1 Epoch
    epochs = 1
    tic = time.perf_counter()
    history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=128,verbose=0)
    toc = time.perf_counter()
    print(f"Run {i} 1 epochs completed in {toc - tic:0.4f} seconds")
    weight = model.layers[0].get_weights()[0]
    w1 = np.empty((0,9), dtype=float)
    w1 = np.concatenate((np.transpose(weight.reshape((9,64))), w1))
    acc_1.append(model.evaluate( x_test, y_test, verbose=0 )[1])
    
    with open("Results/CNNweights_1.csv", "ab") as f:
      np.savetxt(f, w1, delimiter=",")
    with open("Results/CNNacc_1.csv", "ab") as f:
      np.savetxt(f, np.array(acc_1), delimiter=",")
    
    # 2 Epoch
    epochs = 1
    tic = time.perf_counter()
    history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=128,verbose=0)
    toc = time.perf_counter()
    print(f"Run {i} 2 epochs completed in {toc - tic:0.4f} seconds")
    weight = model.layers[0].get_weights()[0]
    w2 = np.empty((0,9), dtype=float)
    w2 = np.concatenate((np.transpose(weight.reshape((9,64))), w2))
    acc_2.append(model.evaluate( x_test, y_test, verbose=0 )[1])
    with open("Results/CNNweights_2.csv", "ab") as f:
      np.savetxt(f, w2, delimiter=",")
    with open("Results/CNNacc_2.csv", "ab") as f:
      np.savetxt(f, np.array(acc_2), delimiter=",")
    
    # 3 Epoch
    epochs = 1
    tic = time.perf_counter()
    history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=128,verbose=0)
    toc = time.perf_counter()
    print(f"Run {i} 3 epochs completed in {toc - tic:0.4f} seconds")
    weight = model.layers[0].get_weights()[0]
    w3 = np.empty((0,9), dtype=float)
    w3 = np.concatenate((np.transpose(weight.reshape((9,64))), w3))
    acc_3.append(model.evaluate( x_test, y_test, verbose=0 )[1])
    with open("Results/CNNweights_3.csv", "ab") as f:
      np.savetxt(f, w2, delimiter=",")
    with open("Results/CNNacc_3.csv", "ab") as f:
      np.savetxt(f, np.array(acc_3), delimiter=",")
    
    # 4 Epoch
    epochs = 1
    tic = time.perf_counter()
    history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=128,verbose=0)
    toc = time.perf_counter()
    print(f"Run {i} 4 epochs completed in {toc - tic:0.4f} seconds")
    weight = model.layers[0].get_weights()[0]
    w4 = np.empty((0,9), dtype=float)
    w4 = np.concatenate((np.transpose(weight.reshape((9,64))), w4))
    acc_4.append(model.evaluate( x_test, y_test, verbose=0 )[1])
    with open("Results/CNNweights_4.csv", "ab") as f:
      np.savetxt(f, w4, delimiter=",")
    with open("Results/CNNacc_4.csv", "ab") as f:
      np.savetxt(f, np.array(acc_4), delimiter=",")
    
    # 5 Epoch
    epochs = 1
    tic = time.perf_counter()
    history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=128,verbose=0)
    toc = time.perf_counter()
    print(f"Run {i} 5 epochs completed in {toc - tic:0.4f} seconds")
    weight = model.layers[0].get_weights()[0]
    w5 = np.empty((0,9), dtype=float)
    w5 = np.concatenate((np.transpose(weight.reshape((9,64))), w5))
    acc_5.append(model.evaluate( x_test, y_test, verbose=0 )[1])
    with open("Results/CNNweights_5.csv", "ab") as f:
      np.savetxt(f, w5, delimiter=",")
    with open("Results/CNNacc_5.csv", "ab") as f:
      np.savetxt(f, np.array(acc_5), delimiter=",")
    
    # 10 Epoch
    epochs = 5
    tic = time.perf_counter()
    history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=128,verbose=0)
    toc = time.perf_counter()
    print(f"Run {i} 10 epochs completed in {toc - tic:0.4f} seconds")
    weight = model.layers[0].get_weights()[0]
    w10 = np.empty((0,9), dtype=float)
    w10 = np.concatenate((np.transpose(weight.reshape((9,64))), w10))
    acc_10.append(model.evaluate( x_test, y_test, verbose=0 )[1])
    with open("Results/CNNweights_10.csv", "ab") as f:
      np.savetxt(f, w10, delimiter=",")
    with open("Results/CNNacc_10.csv", "ab") as f:
      np.savetxt(f, np.array(acc_10), delimiter=",")
    
    # 15 Epoch
    epochs = 5
    tic = time.perf_counter()
    history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=128,verbose=0)
    toc = time.perf_counter()
    print(f"Run {i} 15 epochs completed in {toc - tic:0.4f} seconds")
    weight = model.layers[0].get_weights()[0]
    w15 = np.empty((0,9), dtype=float)
    w15 = np.concatenate((np.transpose(weight.reshape((9,64))), w15))
    acc_15.append(model.evaluate( x_test, y_test, verbose=0 )[1])
    with open("Results/CNNweights_15.csv", "ab") as f:
      np.savetxt(f, w15, delimiter=",")
    with open("Results/CNNacc_15.csv", "ab") as f:
      np.savetxt(f, np.array(acc_15), delimiter=",")
    
    # 20 Epoch
    epochs = 5
    tic = time.perf_counter()
    history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=128,verbose=0)
    toc = time.perf_counter()
    print(f"Run {i} 20 epochs completed in {toc - tic:0.4f} seconds")
    weight = model.layers[0].get_weights()[0]
    w20 = np.empty((0,9), dtype=float)
    w20 = np.concatenate((np.transpose(weight.reshape((9,64))), w20))
    acc_20.append(model.evaluate( x_test, y_test, verbose=0 )[1])
    with open("Results/CNNweights_20.csv", "ab") as f:
      np.savetxt(f, w20, delimiter=",")
    with open("Results/CNNacc_20.csv", "ab") as f:
      np.savetxt(f, np.array(acc_20), delimiter=",")
    
    # 40 Epoch
    epochs = 20
    tic = time.perf_counter()
    history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=128,verbose=0)
    toc = time.perf_counter()
    print(f"Run {i} 2 epochs completed in {toc - tic:0.4f} seconds")
    weight = model.layers[0].get_weights()[0]
    w40 = np.empty((0,9), dtype=float)
    w40 = np.concatenate((np.transpose(weight.reshape((9,64))), w40))
    acc_40.append(model.evaluate( x_test, y_test, verbose=0 )[1])
    with open("Results/CNNweights_40.csv", "ab") as f:
      np.savetxt(f, w40, delimiter=",")
    with open("Results/CNNacc_40.csv", "ab") as f:
      np.savetxt(f, np.array(acc_40), delimiter=",")
    
    # 60 Epoch
    epochs = 20
    tic = time.perf_counter()
    history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=128,verbose=0)
    toc = time.perf_counter()
    print(f"Run {i} 60 epochs completed in {toc - tic:0.4f} seconds")
    weight = model.layers[0].get_weights()[0]
    w60 = np.empty((0,9), dtype=float)
    w60 = np.concatenate((np.transpose(weight.reshape((9,64))), w60))
    acc_60.append(model.evaluate( x_test, y_test, verbose=0 )[1])
    with open("Results/CNNweights_60.csv", "ab") as f:
      np.savetxt(f, w60, delimiter=",")
    with open("Results/CNNacc_60.csv", "ab") as f:
      np.savetxt(f, np.array(acc_60), delimiter=",")
      
    # 100 Epoch
    epochs = 40
    tic = time.perf_counter()
    history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=128,verbose=0)
    toc = time.perf_counter()
    print(f"Run {i} 100 epochs completed in {toc - tic:0.4f} seconds")
    weight = model.layers[0].get_weights()[0]
    w100 = np.empty((0,9), dtype=float)
    w100 = np.concatenate((np.transpose(weight.reshape((9,64))), w100))
    acc_100.append(model.evaluate( x_test, y_test, verbose=0 )[1])
    with open("Results/CNNweights_100.csv", "ab") as f:
      np.savetxt(f, w100, delimiter=",")
    with open("Results/CNNacc_100.csv", "ab") as f:
      np.savetxt(f, np.array(acc_100), delimiter=",")
      
    # 200 Epoch
    epochs = 100
    tic = time.perf_counter()
    history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=128,verbose=0)
    toc = time.perf_counter()
    print(f"Run {i} 100 epochs completed in {toc - tic:0.4f} seconds")
    weight = model.layers[0].get_weights()[0]
    w200 = np.empty((0,9), dtype=float)
    w200 = np.concatenate((np.transpose(weight.reshape((9,64))), w200))
    acc_200.append(model.evaluate( x_test, y_test, verbose=0 )[1])
    with open("Results/CNNweights_200.csv", "ab") as f:
      np.savetxt(f, w200, delimiter=",")
    with open("Results/CNNacc_200.csv", "ab") as f:
      np.savetxt(f, np.array(acc_200), delimiter=",")
    
    
    
    # learning curves
    #summarize_diagnostics(history,'VGG_drop')
