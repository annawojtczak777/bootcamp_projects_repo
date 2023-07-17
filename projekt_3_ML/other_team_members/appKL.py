import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix

# Wczytaj zbiór danych
df = pd.read_csv("neo_v2.csv")

# Przetwarzanie danych
df = df.drop(['id', 'name', 'orbiting_body', 'sentry_object'], axis=1)

# Podział na zbiór treningowy i testowy
X = df.drop('hazardous', axis=1)
Y = df['hazardous']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Trenowanie modelu RandomForestClassifier
model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)

# Generowanie predykcji na zbiorze testowym
y_pred = model.predict(X_test)

# Obliczanie miar klasyfikacji
accuracy = accuracy_score(y_test, y_pred) * 100
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Wyświetlanie raportu klasyfikacji
st.text("Raport klasyfikacji dla modelu RandomForest:")
st.text(classification_report(y_test, y_pred))

# Wizualizacja macierzy pomyłek
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predykcja')
plt.ylabel('Rzeczywistość')
plt.title('Macierz pomyłek')
plt.tight_layout()

# Wyświetlanie dokładności, precyzji i czułości
st.text("Dokładność: {:.2f}%".format(accuracy))
st.text("Precyzja: {:.2f}".format(precision))
st.text("Czułość: {:.2f}".format(recall))

# Wyświetlanie macierzy pomyłek w Streamlit
st.pyplot(plt)

# Wyświetlanie macierzy pomyłek w Streamlit
st.pyplot(plt)
#OPIS 
# W powyższym kodzie wczytujemy zbiór danych "neo_v2.csv", przeprowadzamy przetwarzanie danych przez usunięcie niektórych kolumn, 
# a następnie dzielimy dane na zbiory treningowy i testowy. Trenujemy model RandomForestClassifier, 
# generujemy predykcje na zbiorze testowym i obliczamy miary klasyfikacji, takie jak dokładność, precyzja i czułość.
# Kod zawiera również wizualizację macierzy pomyłek za pomocą heatmapy. 
# Macierz pomyłek przedstawia ogólny obraz działania modelu pod względem poprawnie i 
# niepoprawnie sklasyfikowanych instancji. Dokładność, precyzja i czułość są również wyświetlane.
# Uruchamiając tę aplikację Streamlit, pokazujemy raport klasyfikacji, macierz pomy

#krotszy opis: Ta wizualizacja prezentuje wyniki klasyfikacji modelu RandomForestClassifier dla zbioru danych 
# dotyczącego asteroid. Wykorzystujemy macierz pomyłek, która przedstawia skuteczność modelu w poprawnym i 
# niepoprawnym przewidywaniu klasy asteroidy (hazardous/niehazardous). Raport klasyfikacji dostarcza szczegółowe informacje na temat precyzji, 
# czułości i innych miar oceny modelu. Dodatkowo, podajemy dokładność modelu oraz wykorzystujemy 
# interaktywne komponenty Streamlit, takie jak nagłówki i tekst, aby przedstawić te wyniki w sposób zrozumiały i przejrzysty.