import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Load your dataset
@st.cache
def load_data():
    data = pd.read_csv('clean.csv')
    return data

def train_model(df):
    columns = ['est_diameter_max', 'relative_velocity', 'miss_distance','absolute_magnitude']
    X_train, X_test, y_train, y_test = train_test_split(df[columns], df['hazardous_True'], test_size=0.3, random_state = 42)
    model = RandomForestClassifier(class_weight='balanced', max_depth=2, max_leaf_nodes=3, n_estimators=10, random_state=42)
    model = model.fit(X_train, y_train)
    return model, X_test, y_test

def main():
    st.title("Asteroid Hazard Prediction Model")

    df = load_data()

    model, X_test, y_test = train_model(df)

    # Predict function
    def predict(input_data):
        prediction = model.predict(input_data)
        return prediction

    st.subheader("\n Dataset")
    st.write(df)

    st.subheader("\n Correlation Heatmap")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=.5, ax=ax)
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)

    y_preds = model.predict(X_test)

    st.subheader("\n Metrics of the model")
    metrics_df = pd.DataFrame({
        'Metrics': ['Precision for class 0', 'Recall for class 0', 'F1-score for class 0', 'Precision for class 1', 'Recall for class 1', 'F1-score for class 1'],
        'Values': [
            precision_score(y_test, y_preds, pos_label=0),
            recall_score(y_test, y_preds, pos_label=0),
            f1_score(y_test, y_preds, pos_label=0),
            precision_score(y_test, y_preds, pos_label=1),
            recall_score(y_test, y_preds, pos_label=1),
            f1_score(y_test, y_preds, pos_label=1),
        ]
    })
    st.table(metrics_df.set_index('Metrics'))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_preds)
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    st.pyplot(fig)

    st.subheader("\n Feature Importance")
    feature_imp = pd.Series(model.feature_importances_, index=df.columns[:-1]).sort_values(ascending=False)
    fig, ax = plt.subplots()
    ax.barh(feature_imp.index, feature_imp.values, color='skyblue')
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')

    for i in ax.patches:
     ax.text(i.get_width(), i.get_y()+0.5, 
              str(round((i.get_width()), 2)),
                fontsize=10, fontweight='bold',
                color='grey')

    st.pyplot(fig)

    st.subheader("\n Predict your own asteroid!")
    # User input
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.text('est_diameter_max')
        est_diameter_max = st.slider('Max Estimated Diameter in Kilometres', 0.001362 , 84.730541)

    with col2:
        st.text('relative_velocity')
        relative_velocity = st.slider('Velocity Relative to Earth', 203.346433, 236990.128088)

    with col3:
        st.text('miss_distance')
        miss_distance = st.slider('Distance in Kilometres missed', 6.75, 74.8) 

    with col4:
        st.text('absolute_magnitude')
        absolute_magnitude = st.slider('Describes intrinsic luminosity', 9.230000, 33.200000)

    if st.button('Predict type of Hazardous'):
        result = predict(np.array([[est_diameter_max, relative_velocity, miss_distance, absolute_magnitude]]))
        if result[0] == 1:
            st.text("The asteroid is hazardous.")
        else:
            st.text("The asteroid is not hazardous.")

if __name__ == "__main__":
    main()
