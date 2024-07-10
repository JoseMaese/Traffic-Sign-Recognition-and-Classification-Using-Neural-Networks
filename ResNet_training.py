'''
José E. Maese Álvarez. 
TFG: Uso de redes neuronales para identificación de matrículas.
Funciones de ResNet50
'''
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import funciones as fun
import numpy as np
import keras 
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


learn = 0.00001
epocas = 50


BATCH_SIZE = 64
IMG_SIZE = (180, 180)

directory = "Datasets/ClassificationTrafficSigns/"
train_dataset = image_dataset_from_directory(directory = 'Datasets/ClassificationTrafficSigns', 
                                             shuffle=True,
                                             color_mode = 'rgb',
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=0.2,
                                             subset='training',
                                             seed = 42)
                                                 
test_dataset = image_dataset_from_directory(directory = 'Datasets/ClassificationTrafficSigns',
                                            shuffle=True,
                                            batch_size=BATCH_SIZE,
                                            color_mode = 'rgb',
                                            image_size=IMG_SIZE,
                                            validation_split=0.2,
                                            subset='validation',
                                            seed = 42)
                                                 
    
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
data_augmentation = fun.data_augmenter()


model = Sequential()
    
pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                                                 input_shape=(180,180,3),
                                                 pooling='avg',classes=13,
                                                 weights='imagenet')
    
pretrained_model.trainable = False
pretrained_model.summary()
     
model.add(pretrained_model)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(13, activation='softmax'))

opt = keras.optimizers.Adam(learning_rate=learn)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history = model.fit(train_dataset, validation_data=test_dataset, epochs=epocas)
    
model.save("ResNet_Model_local_v2.h5")
    
plt.figure(1)
df_loss_acc = pd.DataFrame(history.history)
df_loss= df_loss_acc[['loss','val_loss']]
df_loss.rename(columns={'loss':'train','val_loss':'validation'},inplace=True)
df_loss.plot(title='Model loss',figsize=(12,8)).set(xlabel='Epoch',ylabel='Loss')
plt.close(1)
    
plt.figure(2)
df_acc= df_loss_acc[['accuracy','val_accuracy']]
df_acc.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True)
df_acc.plot(title='Model Accuracy',figsize=(12,8)).set(xlabel='Epoch',ylabel='Accuracy')
plt.close(2)
    
ResNet_pred = model.predict(test_dataset)
    
resultados_ResNet = df_loss_acc.to_numpy()
            
        
    















