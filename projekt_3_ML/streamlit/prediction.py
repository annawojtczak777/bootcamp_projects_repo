import joblib
def predict(data):
    rf_model = joblib.load('rf_model.sav')
    return rf_model.predict(data)