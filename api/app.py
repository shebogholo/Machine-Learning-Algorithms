import sys
import numpy as np
from flask import Flask
from sklearn.externals import joblib
from flask import render_template, request, jsonify, redirect, url_for

app = Flask(__name__)

try:
    model = joblib.load('static/logistic_regression_model.pkl')
except:
    print('Error: Application failed')
    sys.exit(0)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict')
def predict():
    # args = request.args
    # required = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    # difference = set(required).difference(set(args.keys()))
    # if len(difference):
    #     return 'Error: Missing arguments'

    # features = np.array().reshape(1, -1)
    # result = model.predict()[:, 1][0]
    # return redirect(url_for('index'))
    return 'Waziri Shebogholo'
    # return jsonify({'Predicted temperature is: ': result})


if __name__ == '__main__':
    app.run()
