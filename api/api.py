import sys
import numpy as np
from sklearn.externals import joblib


def model_prediction(inputs):
    features = np.array(inputs).reshape(-1, 1)
    try:
        with open('static/weather_model.pkl', 'rb') as file:
            model = joblib.load(file)
    except:
        print('Error: Application failed')
        sys.exit(0)
    return model.predict(features)
