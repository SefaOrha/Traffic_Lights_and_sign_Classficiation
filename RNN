from google.colab import drive

drive.mount("/content/gdrive")

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from keras.models import Sequential 
from keras.layers import Dense, BatchNormalization, Dropout, Flatten,SimpleRNN
from keras.utils.np_utils import to_categorical 
from keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import pickle 
from sklearn.model_selection import train_test_split
import cv2
import random

import os
os.chdir("gdrive/MyDrive")

image_shape=(32,32)
path="Dataset/Data/Trafik/Trafik"
images=[]
labels=[]
for class_name in os.listdir(path):
    for image_path in os.listdir(path+"/"+class_name):
         labels.append(int(class_name))      
         image=cv2.imread(path+"/"+class_name+"/"+image_path,0)
         image=cv2.resize(image,(32,32))
         images.append(image)
images=np.array(images,dtype='float64')
images=images.reshape(images.shape[0],images.shape[1],images.shape[2])
labels=np.array(labels,dtype='float64')
labels = to_categorical(labels, 91)

x_train,X_val_test,y_train,Y_val_test=train_test_split(images,labels,test_size=0.3)
x_val,x_test,y_val,y_test=train_test_split(X_val_test,Y_val_test,test_size=0.5)

data = pd.read_csv("Dataset/labels.csv", encoding="ISO-8859-1")
data

model=Sequential()

model.add(SimpleRNN(64,activation='tanh',input_shape=(32,32),return_sequences=True,use_bias=True,
    kernel_initializer='glorot_uniform',
    recurrent_initializer='orthogonal',))

model.add(SimpleRNN(64,activation='tanh',return_sequences=True))
model.add(SimpleRNN(128,activation='tanh',return_sequences=True))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(91,activation='sigmoid'))
model.compile(loss='categorical_crossentropy',metrics=['acc'])
model.summary()
history=model.fit(x_train,y_train,epochs=100,batch_size=32,validation_data=(x_val,y_val))

model.evaluate(x_test,y_test)
model.save("Save")

img=random.randrange(100)
pred=model.predict(x_test[img].reshape(1,32,32))
plt.imshow(x_test[img])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('RNN MODEL KAYIP')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('RNN MODEL BAŞARI')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
print(data.query("Classid=="+str(np.argmax(pred))))
