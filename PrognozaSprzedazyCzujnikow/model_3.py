import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

dforg = pd.read_excel('czujniki_glukozy.xlsx', sheet_name='Czujniki')

#print(df.head(5))

dforg.info() #informacje o strukturze danych

print(dforg.describe().T.to_string()) #opis danych liczbowych

duplikaty = dforg.duplicated() #sprawdzenie, czy w zbiorze danych wystepują duplikaty
df_duplikaty = dforg[dforg.duplicated()]
print('\nDuplikaty')
print(df_duplikaty)
df = dforg[dforg['Województwo świadczeniodawcy'] != 'mazowieckie']

print(f'\nUnikalne wartości dla kolumn\n\n{df.nunique()}') #sprawdzenie unikalnych wartości

lista_unikalnych_Nazwa = df['Nazwa'].unique().tolist() #lista placówek, które sprzedały dany czujnik
print(f'\nUnikalne wartości w kolumnie Nazwa:\n{pd.DataFrame(lista_unikalnych_Nazwa).to_string()}')

print(f'\nBraki w zbiorze danych:\n\n{df.isna().sum()}') #sprawdzenie braków w zbiorze danych

print(f"Udział braków w kolumnie Kwota refundacji wynosi {round(df['Kwota refundacji'].isna().sum()/df.shape[0],2)}")

# ------------------ Czyszczenie danych ----------------------

# zmiana wartości "<5" w kolumnie "Liczba wyrobów" na 3 tj. średnią wartość z zakresu 1–4 i zmiana typu danych na liczbowe)
df['Liczba wyrobów'] = df['Liczba wyrobów'].replace('<5', 3).astype(float)

# dodanie nowej kolumny nazwy miesięcy w postaci liczby (styczeń = 1, ..., grudzień = 12)
miesiace_map = {
    'styczeń': 1, 'luty': 2, 'marzec': 3, 'kwiecień': 4, 'maj': 5, 'czerwiec': 6,
    'lipiec': 7, 'sierpień': 8, 'wrzesień': 9, 'październik': 10, 'listopad': 11, 'grudzień': 12
}
df['Miesiąc_num'] = df['Miesiąc'].map(miesiace_map)

"""uzupełnienie braków w kolumnie kwoty refundacji (przefiltrowanie danych pod kątem liczby produktów mniejszej niz 10, uśrednienie kwoty refundacji)
Zbiór danych zawierał braki w kwocie refundacji przy liczbie wyróbów mniejszej niż 5. Aby uzupełnić brakujące dane: 
- tablea została przefiltrowana pod kątem niskiej liczby sprzedanych czujników, 
- dodano nową kolumnę cena_jednostkowa
- cena brakujących refundacji została uśrednion w zależności od kodu produktu
"""
niska_sprzedaz = df[(df['Liczba wyrobów'] > 5) & (df['Liczba wyrobów'] < 10)]
print(f'\nDane przefiltrowane pod kątem niskiej liczby wyróbów w celu ustalenia ceny jednostkowej \n\n{niska_sprzedaz.head(5).to_string()}')
niska_sprzedaz['cena_jednostkowa'] = niska_sprzedaz['Kwota refundacji']/niska_sprzedaz['Liczba wyrobów'] #dodajemy dodatkową kolumnę z ceną jednostkową
print(niska_sprzedaz.head(5))

#--------------------------
#próba uzupełnienia braków w kwocie refundacji na podstawie ustalonej średniej kwoty refundacji jednego czujnika
# print(niska_sprzedaz['cena_jednostkowa'].nunique())
# srednia_cena_jed = round(niska_sprzedaz['cena_jednostkowa'].mean(),2) # średnia kwota refundacji czujnika
# print(f'\nSrednia kwota refundacji jednego czujnika wynosi: {srednia_cena_jed}\n')
# df['Kwota refundacji'].fillna(srednia_cena_jed * df['Liczba wyrobów'], inplace=True)
#---------------------------

rodzaj_czujnika = pd.DataFrame(round(niska_sprzedaz.groupby(['Kod produktu'])['cena_jednostkowa'].mean(),2))
print(rodzaj_czujnika)  #uśredniamy cenę jednostkową refundacji w zależności od kodu produktu
srednie_ceny = rodzaj_czujnika['cena_jednostkowa'].to_dict() # średnie ceny w zalezności od kodu produktu zapisujemy do słownika
print(f'\nSłownik średnich cen jednostkowych: {srednie_ceny}\n')

def uzupelnij_refundacje(row):# Funkcja pomocnicza do wyliczania brakującej kwoty refundacji
    if pd.isna(row['Kwota refundacji']):
        kod = row['Kod produktu']
        if kod in srednie_ceny:
            return srednie_ceny[kod] * row['Liczba wyrobów']
    return row['Kwota refundacji']

df['Kwota refundacji'] = df.apply(uzupelnij_refundacje, axis=1) # uzupełnienie braków przy użyciu funkcji pomocniczej

print(f'Dane po uzupełnieniu braków:\n\n {df.head(10).to_string()}') #podgląd tabeli po uzupełnieniu braków
df = df.sort_values(by='Miesiąc_num')


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# One-hot encoding zmiennych kategorycznych
df_encoded = pd.get_dummies(df, columns=['Województwo świadczeniodawcy','Miesiąc'], drop_first=True)

# Transformacja logarytmiczna zmiennej celu
df_encoded['log_Liczba_wyrobow'] = np.log1p(df_encoded['Liczba wyrobów'])

# Definicja zmiennych X i y
X = df_encoded.drop(columns=['Kwota refundacji','Liczba wyrobów', 'Kod produktu', 'Kod świadczeniodawcy', 'Nazwa','REGON', 'Miesiąc_num', 'log_Liczba_wyrobow'])
y = df_encoded['log_Liczba_wyrobow']

# Podział danych
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trening modelu
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predykcja na zestawie testowym i odwrotna transformacja
y_pred_log = model.predict(X_test) # oszacowanie sprzedaży w skali logarytmicznej
y_pred = np.expm1(y_pred_log) # aby uzyskać realne wartośći, wracamy do oryginalnej skali
y_test_orig = np.expm1(y_test) # transformacja danych testowych do rzeczywistych wartości

# Ewaluacja i ocena modelu

rmse = mean_squared_error(y_test_orig, y_pred, squared=False)
print(f"RMSE: {rmse:.2f}")

mae = mean_absolute_error(y_test_orig, y_pred)
print(f"MAE: {mae:.2f}")

r2 = r2_score(y_test_orig, y_pred)
print(f"R²: {r2:.4f}")


# Tworzenie DataFrame z porównaniem
comparison_df = pd.DataFrame({
    'Rzeczywista liczba wyrobów': y_test_orig,
    'Prognozowana liczba wyrobów': y_pred
}).reset_index(drop=True)
print(comparison_df.head(10))

# Wykres scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Rzeczywista liczba wyrobów', y='Prognozowana liczba wyrobów', data=comparison_df)
plt.plot([0, max(y_test_orig.max(), y_pred.max())], [0, max(y_test_orig.max(), y_pred.max())],
         color='red', linestyle='--', label='Idealna zgodność')
plt.title("Porównanie: Rzeczywista vs Prognozowana liczba wyrobów")
plt.xlabel("Rzeczywista liczba wyrobów")
plt.ylabel("Prognozowana liczba wyrobów")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#----------- Ulepszenie modelu: Strojenie hiperparametrów

from sklearn.model_selection import RandomizedSearchCV

# Definicja siatki parametrów
param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=20,  # liczba kombinacji do przetestowania
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    scoring='neg_mean_squared_error'
)

# Dopasowanie do danych treningowych
random_search.fit(X_train, y_train)

# Najlepszy model
best_model = random_search.best_estimator_
print(f'Najlepsze parametry: \n{random_search.best_params_}')

# Predykcja
y_pred_best_log = best_model.predict(X_test)
y_pred_best = np.expm1(y_pred_best_log)  # odwrotna transformacja log
y_test_orig = np.expm1(y_test)

# Ewaluacja
rmse_best = mean_squared_error(y_test_orig, y_pred_best, squared=False)
mae_best = mean_absolute_error(y_test_orig, y_pred_best)
r2_best = r2_score(y_test_orig, y_pred_best)

print(f"\nStrojenie hiperparametrów - wyniki:")
print(f"RMSE dla rzeczywistej skutecznosci modelu: {rmse_best:.2f}")
print(f"MAE dla rzeczywistej skutecznosci modelu: {mae_best:.2f}")
print(f"R² dla rzeczywistej skutecznosci modelu: {r2_best:.4f}")

# Tworzenie DataFrame z porównaniem
comparison_df = pd.DataFrame({
    'Rzeczywista liczba wyrobów': y_test_orig,
    'Prognozowana liczba wyrobów': y_pred_best
}).reset_index(drop=True)
print(comparison_df.head(10))

# Wykres scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Rzeczywista liczba wyrobów', y='Prognozowana liczba wyrobów', data=comparison_df)
plt.plot([0, max(y_test_orig.max(), y_pred_best.max())], [0, max(y_test_orig.max(), y_pred_best.max())],
         color='red', linestyle='--', label='Idealna zgodność')
plt.title("Porównanie: Rzeczywista vs Prognozowana liczba wyrobów dla rzeczywitej skuteczności modelu")
plt.xlabel("Rzeczywista liczba wyrobów")
plt.ylabel("Prognozowana liczba wyrobów")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Tabela porównawcza
metrics_df = pd.DataFrame({
    'Metryka': ['MAE', 'RMSE', 'R²'],
    'Model bazowy': [round(rmse), round(mae), round(r2,4)],
    'Model strojony': [round(rmse_best), round(mae_best), round(r2_best,4)]
})
print(metrics_df)

# Ważność cech z modelu Random Forest
feature_importances = pd.Series(best_model.feature_importances_, index=X_train.columns)
feature_importances = feature_importances.sort_values(ascending=False).head(15)  # top 15 najważniejszych

# Wykres
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.values, y=feature_importances.index, palette='viridis')
plt.title("Top 15 najważniejszych cech wpływających na liczbę wyrobów")
plt.xlabel("Ważność cechy (feature importance)")
plt.ylabel("Cecha")
plt.grid(True, axis='x')
plt.tight_layout()
plt.show()



