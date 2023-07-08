import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.utils import layer_utils 
import os
import cv2
import random
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf


# Load the trained model
model = load_model('C:/Users/Kasia/Desktop/dl_fun/my_model2.h5')  # replace with your saved model

if model is None:
    st.write('Failed to load the model')

st.title('Music Genre Prediction from Album Covers')

def preprocess_image(image):
    image = image.resize((150, 150))
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image


def load_images_and_labels(categories):
    img_lst=[]
    labels=[]
    for index, category in enumerate(categories):
        for image_name in os.listdir(f'C:/Users/Kasia/Desktop/dl_fun/data/{category}'): #path to the folders with data
            img = cv2.imread(f'C:/Users/Kasia/Desktop/dl_fun/data/{category}/{image_name}') #path to the folders
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

            img_array = cv2.resize(img, (150,150)) 

            img_lst.append(img_array)
            labels.append(index)
    return img_lst, labels

categories = ['disco', 'electro', 'folk', 'rap', 'rock'] 

img_lst, labels = load_images_and_labels(categories)



# Display 16 random images with their labels
random_indices = random.sample(range(0, len(img_lst)), 16)
st.write("## Random Images from Dataset")

for i in range(0, 16, 4):  # Change the step size to adjust the number of images per row
    cols = st.columns(4)  # Create 4 columns
    for j in range(4):
        index = random_indices[i + j]
        cols[j].image(img_lst[index], width=100)  
        cols[j].text(f"Label: {categories[labels[index]]}")

uploaded_file = st.file_uploader("Choose an album cover image...", type=['png', 'jpg'])

images = np.array(img_lst) / 255.0 #normalization
labels = to_categorical(labels)  #one hot encoding
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.4, random_state=27)
y_pred = model.predict(x_test)

# Get the actual labels (instead of one-hot encoding) for y_test and y_pred
y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)


### PRED OF THE NEW IMAGE
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess the image
    image = preprocess_image(image)
    
    
    st.write("Predicting...")
    prediction = model.predict(image)

    # Use softmax to get the scores
    score = tf.nn.softmax(prediction[0])

    categories = ['disco', 'electro', 'folk', 'rap', 'rock'] 

    # Get the name of the predicted class and the confidence score
    predicted_genre = categories[np.argmax(score)]
    confidence_score = 100 * np.max(score)

    st.write(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(predicted_genre, confidence_score)
    )

# Display 15 random images from the test set with their labels
random_indices = np.random.choice(x_test.shape[0], size=15, replace=False)

st.write("## Random Images from Test Dataset with Predicted Labels")
for i in range(0, 15, 3):  # Change the step size to adjust the number of images per row
    cols = st.columns(3)  # Create 3 columns
    for j in range(3):
        idx = random_indices[i + j]
        img = x_test[idx]
        
        true_label = categories[y_test_labels[idx]]
        pred_label = categories[y_pred_labels[idx]]

        if true_label == pred_label:
            cols[j].markdown(f"__True: {true_label}__\n__Predicted: {pred_label}__")  # Text will be bold
        else:
            cols[j].markdown(f"True: {true_label}\nPredicted: {pred_label}")  # Text will be normal

        cols[j].image(img, width=100)  


# Classification report
st.write('## Classification Report')
report = classification_report(y_test_labels, y_pred_labels, target_names=categories, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

from contextlib import redirect_stdout
import io

# SUMMARY - uwaga jest super duze, jak Twojego modelu tez takie bedzie to niekoniecznie musimy zostawiac
#Zostawiam dwie propozycje, krotsza i dluzsza 

trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_variables])
st.write(f"Trainable params: {trainable_params}")

non_trainable_params = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_variables])
st.write(f"Non-trainable params: {non_trainable_params}")

total_params = trainable_params + non_trainable_params
st.write(f"Total params: {total_params}")


#Dluzsza 
stream = io.StringIO()
with redirect_stdout(stream):
    model.summary()
summary_string = stream.getvalue()
stream.close()

st.write('## Model Summary')
st.text(summary_string)

