import joblib
import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import time

# read original dataset
df=pd.read_csv(r"C:\Users\toawe\OneDrive\Pulpit\jdszr10-TheChaosMakers\projekt_3_ML\streamlit\clean.csv")

cols=['est_diameter_max','relative_velocity','miss_distance','absolute_magnitude']

# function shows reslut of feature engineering
def exploration():
    st.subheader('Correlation Heatmap')
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0, linewidths=.5, ax=ax)
    st.pyplot(fig)

    st.subheader('Histplot')
    feature_cols = list(df.columns)[0:-1]
    target_var = df.columns[-1]
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10))
    for i, col in enumerate(feature_cols):
        sns.histplot(data = df, x = col, ax = axes[i//2, i%2], hue = target_var, fill = True, kde=True, palette='coolwarm',log_scale=True)
    st.pyplot(fig)

# split data into train and test sets
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(df[cols], df['hazardous_True'], test_size=0.3, random_state=42)

# data standardization
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# create an instance of the random forest classifier
rf_model=RandomForestClassifier(max_depth=5, 
                                min_samples_split=10, 
                                min_samples_leaf=5,
                                class_weight='balanced',
                                random_state=42)

# train the classifier on the training data
start_time = time.time()
rf_model.fit(X_train, y_train)
elapsed_time = time.time() - start_time

#models estimation
def estimation_models():
    # logistic regresion models():
    model_lr = LogisticRegression(random_state=42).fit(X_train, y_train)
    y_pred_model_lr = model_lr.predict(X_test)
    Scores_lr = {'Accuracy':round(accuracy_score(y_test,y_pred_model_lr),3),
          'F1_score':round(f1_score(y_test,y_pred_model_lr),3),
          'Recall':round(recall_score(y_test,y_pred_model_lr),3),
          'Precision':round(precision_score(y_test,y_pred_model_lr),3)}
    df_lr=pd.DataFrame(data=Scores_lr,index=['Logistic_Regresion'])

    # decision tree calssifier model
    tree_1 = DecisionTreeClassifier(random_state=42, max_depth=3,criterion='gini').fit(X_train, y_train) 
    y_pred_tree = tree_1.predict(X_test)       
    Scores_t = {'Accuracy':round(accuracy_score(y_test,y_pred_tree),3),
          'F1_score':round(f1_score(y_test,y_pred_tree),3),
          'Recall':round(recall_score(y_test,y_pred_tree),3),
          'Precision':round(precision_score(y_test,y_pred_tree),3)}
    df_t=pd.DataFrame(data=Scores_t,index=['Decision_Tree_Classifier'])

    # random forest classifier model 
    rf_model=RandomForestClassifier(max_depth=5, 
                                min_samples_split=10, 
                                min_samples_leaf=5,
                                class_weight='balanced',
                                random_state=42).fit(X_train, y_train)
    pred_test = rf_model.predict(X_test)
    scores_rf_model = {'Accuracy':round(accuracy_score(y_test,pred_test),3),
                        'Recall':round(recall_score(y_test,pred_test),3),
                        'Precision':round(precision_score(y_test,pred_test),3),
                        'F1_score':round(f1_score(y_test,pred_test),3)}
    df_rf=pd.DataFrame(data=scores_rf_model,index=['Random_Forest_Classifier'])


    # xgb model
    xgb_cl = xgb.XGBClassifier(n_estimators=100, 
                           max_depth=5, 
                           use_label_encoder=False, 
                           eval_metric='error',
                           random_state=42).fit(X_train, y_train)
    y_pred_xgb = xgb_cl.predict(X_test)
    Scores_xgb = {'Accuracy':round(accuracy_score(y_test, y_pred_xgb),3),
          'F1_score':round(f1_score(y_test, y_pred_xgb),3),
          'Recall':round(recall_score(y_test, y_pred_xgb),3),
          'Precision':round(precision_score(y_test, y_pred_xgb),3)}
    df_xgb=pd.DataFrame(data=Scores_xgb,index=['XGB_Classifier'])

    # KNN model
    knn = KNeighborsClassifier(n_neighbors=28).fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    Scores_KNN = {'Accuracy':round(accuracy_score(y_test, y_pred_knn),3),
          'F1_score':round(f1_score(y_test, y_pred_knn),3),
          'Recall':round(recall_score(y_test, y_pred_knn),3),
          'Precision':round(precision_score(y_test, y_pred_knn),3)}
    df_KNN=pd.DataFrame(data=Scores_xgb,index=['KNN_Classifier'])

    result = pd.concat([df_lr,df_t,df_rf, df_xgb, df_KNN])
    st.table(result)


# function predict on the test set and gets score
def predict_test():

    pred_test = rf_model.predict(X_test)
    pred_train = rf_model.predict(X_train) #predykcja na zbiorze trenigowym 
  
    st.markdown(f'Training time: {elapsed_time:.3f} seconds') #czas
    
    st.markdown('Classification report for train set')
    clsf_report_train = pd.DataFrame(classification_report(y_train,pred_train,output_dict=True))
    st.table(clsf_report_train)

    st.markdown('Classification report for test set')
    clsf_report_test = pd.DataFrame( classification_report(y_test,pred_test,output_dict=True))
    st.table(clsf_report_test)
    
    st.markdown('Feature Importance')
    feature_imp = pd.Series(rf_model.feature_importances_, index=df.columns[0:-1]).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12,5))
    ax.barh(feature_imp.index, feature_imp.values)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    st.pyplot(fig)

    #wizulaizacja dla confusion_matrix
    class_names=[0,1] # name  of classes
    fig, ax = plt.subplots(figsize=(6, 4), dpi=80)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    
    # create heatmap
    sns.heatmap(pd.DataFrame(confusion_matrix(y_test,pred_test)), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('True Values')
    plt.xlabel('Predicted Values')   
    st.pyplot(fig)

def krzywa_ROC():
    
    #sprawdzeie modelu pod kątem krzywej ROC (działa na prawdopodobienstwie (lub score)z modelu, a nie na wartosciach klas - tak jak classification_report)
    #zwsze robimy nie na predict tylko na predict_proba

    from sklearn.metrics import roc_auc_score, roc_curve
    pred_train_proba = rf_model.predict_proba(X_train)[:,1] #wyrzuci prawd. lub score przyporzadkowaia do jedynki lub zera
    pred_test_proba = rf_model.predict_proba(X_test)[:,1]

    st.text('AUC rate train: ')
    st.text(roc_auc_score(y_train,pred_train_proba))
    st.text('AUC rate test: ')
    st.text(roc_auc_score(y_test,pred_test_proba))
    
    roc_auc_score(y_train,pred_train_proba)
    roc_auc_score(y_test,pred_test_proba)
    #krzywa ROC zwraca 3 argumenty: false_positive_rate, true_positive_rate, i treshholds, dla których zostały wiliczone)
    
    st.markdown('Visualization of ROC Curve')
    fpr_train, tpr_train, thresholds =roc_curve(y_train, pred_train_proba)
    fpr_test, tpr_test, thresholds_test =roc_curve(y_test, pred_test_proba)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(fpr_train,tpr_train, label='train')
    ax.plot(fpr_test,tpr_test,label='test')
    ax.plot(np.arange(0,1,0.01),np.arange(0,1,0.01),'--') #odnieisienie do modelu losowego
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    st.pyplot(fig)
   

# save the model to disk
joblib.dump(rf_model, 'rf_model.sav')