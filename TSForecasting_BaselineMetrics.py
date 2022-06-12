# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 11:27:40 2022

@author: Mehrin Kiani
From the book Geron Hands-on Machine Learning with Scikit Learn, Keras and TensorFlow
"""
import keras
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf

def generate_time_series (batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace (0,1, n_steps)
    series = 0.5 *np.sin((time-offsets1) * (freq1*10+10)) # wave1
    series += 0.2 *np.sin((time-offsets2) * (freq2*20+20)) # wave2
    series += 0.1 *(np.random.rand(batch_size, n_steps) -0.5) # noise
    return series [..., np.newaxis].astype(np.float32)

# Create the timeseries
n_steps = 50
series = generate_time_series(10000, n_steps +1)
X_train, y_train = series[:7000, :n_steps],series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps],series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps],series[9000:, -1]
    
plt.plot(X_test[0,:])    

# Naive forecasting
y_pred = X_valid[:,-1]
p1 = keras.losses.mean_squared_error (y_valid, y_pred)
with tf.Session() as sess:
    p = np.mean(p1.eval())