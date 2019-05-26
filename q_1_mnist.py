#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

model_2 = Sequential()
input_shape = (x_train.shape[1],x_train.shape[2],1)
model_2.add(Conv2D(32, kernel_size=(3,3), input_shape=input_shape))
model_2.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model_2.add(Flatten())
model_2.add(Dense(128, activation=tf.nn.relu))
model_2.add(Dropout(0.2))
model_2.add(Dense(10,activation=tf.nn.softmax))
model_2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_2.fit(x=x_train,y=y_train, epochs=4)


# In[2]:


model_2.evaluate(x_test, y_test)


# In[ ]:




