import sys
import numpy as np
from flask import Flask
from sklearn.externals import joblib
from flask import render_template, request

app = Flask(__name__)


def model_prediction(inputs):
    features = np.array(inputs).reshape(-1, 1)
    try:
        model = joblib.load('static/weather_model.pkl')
    except:
        print('Error: Application failed')
        sys.exit(0)
    return model.predict(features)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['post', 'get'])
def predict():
    min_temperature = request.form.get('min_temperature')
    data = np.array(int(min_temperature))
    prediction = model_prediction(data).flatten()[0]
    output = 'The predicted maximum temperature is {:.2f}'.format(prediction)
    return render_template('index.html', feature=data, prediction=output)


if __name__ == '__main__':
    app.run()
