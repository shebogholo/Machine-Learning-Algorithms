import sys
import numpy as np
from sklearn.externals import joblib

file_name = 'static/weather_model.pkl'


def model_prediction(inputs):
    features = np.array(inputs).reshape(-1, 1)
    try:
        with open(file_name, 'rb') as file:
            model = joblib.load(file)
    except:
        print('Error: Application failed')
        sys.exit(0)
    return model.predict(features)
