#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
import cv2
from sklearn.preprocessing import OneHotEncoder
import datetime

import PIL

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


import warnings
warnings.filterwarnings("ignore")


# In[2]:


img = cv2.imread(r"C:\Users\kamci\OneDrive\Pulpit\Projekt DL\Nowy folder\images labeled\images labeled\disco\3_0.jpg")


# In[3]:


type(img)


# In[4]:


img.shape


# In[5]:


img.dtype


# In[6]:


fig = plt.figure(figsize=(10,5))
plt.imshow(img)
plt.show()


# In[7]:


import os
import cv2

def load_images_and_labels(categories):
    img_lst = []
    labels = []
    for index, category in enumerate(categories):
        folder_path = f"C:/Users/kamci/OneDrive/Pulpit/Projekt DL/Nowy folder/images labeled/images labeled/{category}"
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_array = cv2.resize(img, (150, 150))
            img_lst.append(img_array)
            labels.append(index)
    return img_lst, labels

categories = ['disco', 'electro', 'folk', 'rap', 'rock']
img_lst, labels = load_images_and_labels(categories)


# In[10]:


#Przygotowanie danych:

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(img_lst, labels, test_size=0.2, random_state=42)

# Przekształcenie obrazów na format zgodny z modelem (np. skalowanie pikseli do zakresu [0, 1])
X_train = np.array(X_train) / 255.0
X_test = np.array(X_test) / 255.0

# Przekształcenie etykiet na postać one-hot encoding
num_classes = len(categories)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)


# In[ ]:


#Zbudowanie modelu sieci neuronowej:
#Zdefiniuj architekturę modelu, wybierając odpowiednie warstwy i parametry.
#Skompiluj model, określając funkcję straty, optymalizator i metryki.
#python


# In[ ]:


from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Zdefiniowanie architektury modelu
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Kompilacja modelu
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[9]:


from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np

# Wczytanie wstępnie wytrenowanego modelu VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Zamrożenie wag wszystkich warstw wstępnie wytrenowanego modelu
for layer in base_model.layers:
    layer.trainable = False

# Dodanie nowych warstw do modelu
model = keras.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))

# Kompilacja modelu
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Konwersja danych na tablice numpy
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Przekształcenie etykiet na kodowanie one-hot
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Definicja liczby epok i rozmiaru partii (batch size)
epochs = 10
batch_size = 32

# Trenowanie modelu
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))


# In[ ]:




