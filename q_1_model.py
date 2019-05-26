import glob
import os
from PIL import Image, ImageDraw
import numpy as np

path_out_train = '/home/user/Downloads/sem_6/CS_671_DL/Assignment_2/cla_test/train/'
path_out_test = '/home/user/Downloads/sem_6/CS_671_DL/Assignment_2/cla_test/test/'

x_train = []
y_train = []
x_test = []
y_test = []

for train_file in glob.glob(path_out_train+'*.jpg') :
    img_x_train = Image.open(train_file);
    x_train.append(np.array(img_x_train))
    image_name = str(train_file).split('/')[-1]
    image_name_initials = image_name.split('.')[0]
    class_no = (int(image_name_initials.split('_')[0])*48)+(int(image_name_initials.split('_')[1])*24)+(int(image_name_initials.split('_')[2])*2)+(int(image_name_initials.split('_')[3]))
    y_train.append(class_no)
    img_x_train.close()
    
for test_file in glob.glob(path_out_test+'*.jpg') :
    img_x_test = Image.open(test_file);
    x_test.append(np.array(img_x_test))
    image_name = str(test_file).split('/')[-1]
    image_name_initials = image_name.split('.')[0]
    class_no = (int(image_name_initials.split('_')[0])*48)+(int(image_name_initials.split('_')[1])*24)+(int(image_name_initials.split('_')[2])*2)+(int(image_name_initials.split('_')[3]))
    y_test.append(class_no)
    img_x_test.close()

print(y_test)


print(np.array(x_train).shape,'x_train.shape')
print(np.array(y_train).shape,'y_train.shape')

print(np.array(x_test).shape,'x_test.shape')
print(np.array(y_test).shape,'y_test.shape')

x_train_2 = np.array(x_train).astype('float32')
x_test_2 = np.array(x_test).astype('float32')
y_train_2 = np.array(y_train)
y_test_2 = np.array(y_test)

x_train_2 /= 255
x_test_2 /= 255


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

model = Sequential()
input_shape = (x_train_2.shape[1],x_train_2.shape[2],x_train_2.shape[3])
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(96,activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x=x_train_2,y=y_train_2, epochs=4)

model.evaluate(x_test_2, y_test_2)
