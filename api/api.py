import sys
import numpy as np
from sklearn.externals import joblib


def model_prediction(inputs):
    features = np.array(inputs).reshape(-1, 1)
    try:
        model = joblib.load('static/weather_model.pkl')
    except:
        print('Error: Application failed')
        sys.exit(0)
    return model.predict(features)
