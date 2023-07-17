import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Wczytanie danych
df = pd.read_csv("neo_v2.csv")

# Przetwarzanie danych
df = df.drop(['id', 'name', 'orbiting_body', 'sentry_object'], axis=1)

# Podział na cechy i etykiety
X = df.drop('hazardous', axis=1)
Y = df['hazardous']

# Pobranie nazw cech
feature_names = X.columns.tolist()

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Trenowanie modelu
model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)

# Interaktywne wprowadzanie wartości za pomocą suwaków
st.sidebar.title('Wprowadź wartości cech asteroidy')
est_diameter_min = st.sidebar.slider('Minimalna szacowana średnica (km)', float(df['est_diameter_min'].min()), float(df['est_diameter_min'].max()))
est_diameter_max = st.sidebar.slider('Maksymalna szacowana średnica (km)', float(df['est_diameter_max'].min()), float(df['est_diameter_max'].max()))
relative_velocity = st.sidebar.slider('Prędkość względem Ziemi (km/h)', float(df['relative_velocity'].min()), float(df['relative_velocity'].max()))
miss_distance = st.sidebar.slider('Odległość od Ziemi (km)', float(df['miss_distance'].min()), float(df['miss_distance'].max()))
absolute_magnitude = st.sidebar.slider('Wielkość absolutna', float(df['absolute_magnitude'].min()), float(df['absolute_magnitude'].max()))

# Przygotowanie danych do prognozowania
prediction_input = np.array([[est_diameter_min, est_diameter_max, relative_velocity, miss_distance, absolute_magnitude]])

# Wykonanie prognozy
prediction = model.predict(prediction_input)

# Wyświetlanie wyników
st.title('Prognozowanie klasy asteroidy')
st.write('Przewidywana klasa asteroidy: ', prediction[0])
st.write('Dokładność modelu: ', accuracy_score(y_test, model.predict(X_test)))



#Ta wizualizacja umożliwia interaktywne wprowadzanie wartości dla cech asteroidy za pomocą suwaków. 
#Następnie na podstawie tych wartości prognozowana jest klasa asteroidy (hazardous lub niehazardous). 
#Wynik prognozy oraz wykres lub wizualizacja pozwalają lepiej zrozumieć zależności między wprowadzanymi wartościami a przewidywanymi klasami asteroidy.
