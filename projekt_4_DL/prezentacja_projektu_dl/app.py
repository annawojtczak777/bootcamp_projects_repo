
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
import os
import cv2
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import PIL
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# Load the trained model
model = load_model(r'C:\Users\toawe\OneDrive\Pulpit\dl\model_1.h5')  

st.set_page_config(page_title='Music Genre Prediction from Album Covers')#,layout="wide")

st.title('App Music Genres Prediction from Album Covers ')

st.image(
        "https://i.imgur.com/c9iy4uf.png",
        use_column_width='auto'
        )

url1 = 'https://i.imgur.com/c9iy4uf.png'
st.write("Photo Source : [Imgur - free image hosting site](%s)" % url1)

st.header('Dataset')
url2 = 'https://www.kaggle.com/datasets/anastasiapetrunia/album-covers-dataset'
st.write("Source : [Kaggle](%s)" % url2)
st.markdown('''
            We worked with images of album covers for the albums released between 2000 & 2022. 
            The data was collected by using Selenium tool from AllMusic website -
            in-depth resource for finding out more about the albums, bands, musicians and songs.
            ''')
st.write('''
The dataset contains 9311 color images of various sizes, maximum 400 x 400 pixels, jpg format.
The images are high resolution and have been classified into 5 classes: rap, folk, hard rock, disco, electronic.
Classes are nicely balanced, each with an average of 1860 photos.
''')

def preprocess_image(image):
    image = image.resize((224, 224))
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image


def load_images_and_labels(categories):
    img_lst=[]
    labels=[]
    for index, category in enumerate(categories):
        for image_name in os.listdir(fr'C:\Users\toawe\OneDrive\Pulpit\dl\archive (9)\images labeled\images labeled\{category}'): #path to the folders with data
            img = cv2.imread(fr'C:\Users\toawe\OneDrive\Pulpit\dl\archive (9)\images labeled\images labeled\{category}\{image_name}') #path to the folders
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

            img_array = cv2.resize(img, (224,224)) 

            img_lst.append(img_array)
            labels.append(index)
    return img_lst, labels

categories = ['disco', 'electro', 'folk', 'rap', 'rock'] 

img_lst, labels = load_images_and_labels(categories)

# Display 16 random images with their labels
random_indices = random.sample(range(0, len(img_lst)), 16)

st.markdown('##### Random Images from Dataset')

for i in range(0, 16, 4):  # Change the step size to adjust the number of images per row
    cols = st.columns(4)  # Create 4 columns
    for j in range(4):
        index = random_indices[i + j]
        cols[j].image(img_lst[index], width=100)  
        cols[j].text(f"Label: {categories[labels[index]]}")

image_output = Image.open(r'C:\Users\toawe\OneDrive\Pulpit\dl\output.png')
st.image(image_output, caption='Number of album covers per categories')


st.header('Short about workflow and fine-tuning our image classification model')
st.markdown('##### Benchmark model')

image_sum_bench_model = Image.open(r'C:\Users\toawe\OneDrive\Pulpit\dl\basic_summary_model.png')
st.image(image_sum_bench_model, caption='Summary of benchmark model')

image_his_bench_model = Image.open(r'C:\Users\toawe\OneDrive\Pulpit\dl\basic_plot_history_model.png')
st.image(image_his_bench_model, caption='History of learn our benchmark model')

image_evaluate_bench_model = Image.open(r'C:\Users\toawe\OneDrive\Pulpit\dl\basic_evaluate_model.png')
st.image(image_evaluate_bench_model, caption='Cassification report of benchmark model')

st.markdown('##### Using data augmentation by applying random transformations')

image_augum = Image.open(r'C:\Users\toawe\OneDrive\Pulpit\dl\augumentation_train_ds.png')
st.image(image_augum , caption='Visualize random image after augmentation', width=600)

st.markdown('##### Using transfer-learning')

image_sum_best_model = Image.open(r'C:\Users\toawe\OneDrive\Pulpit\dl\best_summary_model.png')
st.image(image_sum_best_model, caption='Summary of the best model')

image_his_best_model = Image.open(r'C:\Users\toawe\OneDrive\Pulpit\dl\best_plot_history_model.png')
st.image(image_his_best_model, caption='History of learn the best model')

image_evaluate_best_model = Image.open(r'C:\Users\toawe\OneDrive\Pulpit\dl\best_evaluate_model.png')
st.image(image_evaluate_best_model, caption='Cassification report the best model')


st.write('---')
st.header('Try our model and check genre of your album cover!')

uploaded_file = st.file_uploader("Choose an album cover image...", type=['png', 'jpg'])


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width='never', caption='Uploaded Image.')
    
    # Preprocess the image
    image = preprocess_image(image)
    
    # Make the prediction
    st.write("Predicting...")
    prediction = model.predict(image)

    #categories = ['disco', 'electro', 'folk', 'rap', 'rock'] 

    # Use softmax to get the scores
    score = tf.nn.softmax(prediction[0])

    # Get the name of the predicted class and the confidence score
    predicted_genre = categories[np.argmax(score)]
    confidence_score = 100 * np.max(score)

    st.header(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(predicted_genre, confidence_score)
    )
    