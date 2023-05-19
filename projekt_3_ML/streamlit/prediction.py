import joblib
def predict(data):
    rf_class_1 = joblib.load('rf_model.sav')
    return rf_class_1.predict(data)