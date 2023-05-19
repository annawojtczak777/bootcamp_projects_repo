import joblib
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score,f1_score

df=pd.read_csv("neo_v2.csv")

df = df.drop(['id','name','orbiting_body','sentry_object'], axis=1)

df = pd.get_dummies(df, columns = ['hazardous'], drop_first= True)
# zmienna targetu przyjmuje dwie wartości więc modyfikujemy w miejscu na binarną

cols  =['est_diameter_max', 'relative_velocity','miss_distance', 'absolute_magnitude']

X_train, X_test, y_train, y_test = train_test_split(df[cols], df['hazardous_True'], test_size=0.3, random_state=42)

rf_class_1 = RandomForestClassifier(max_depth=5, 
                                    min_samples_split=10, 
                                    min_samples_leaf=5,
                                    class_weight='balanced', 
                                    random_state=42)
rf_class_1 =rf_class_1.fit(X_train, y_train)
# predykcja
pred_test = rf_class_1.predict(X_test)

Scores_rf = {'Accuracy':round(accuracy_score(y_test,pred_test),3),
          'F1_score':round(f1_score(y_test,pred_test),3),
          'Recall':round(recall_score(y_test,pred_test),3),
          'Precision':round(precision_score(y_test,pred_test),3)}
df_rf=pd.DataFrame(data=Scores_rf,index=['RandomForestClassifier'])
print(df_rf)
print(confusion_matrix(y_test,pred_test))

joblib.dump(rf_class_1, 'rf_model.sav')