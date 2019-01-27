from flask import Flask, request, jsonify
from sklearn.externals import joblib
import pandas as pd
import json

# Set feature names/order
feats = ['sepal length (cm)',
         'sepal width (cm)',
         'petal length (cm)',
         'petal width (cm)']

# Load model
clf = joblib.load('./data/clf_iris.model')

# Create flask app
app = Flask(__name__)

@app.route('/classes_/', methods=['GET','POST'])
def get_classes():
    classes = list(clf.classes_.astype(str))
    return json.dumps({'classes':classes})

@app.route('/predict/', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.read_json(data['input'])
    df['y_pred'] = clf.predict(df[feats])
    return jsonify({'output':df.to_json()})

@app.route('/predict_proba/', methods=['POST'])
def predict_proba():
    data = request.get_json()
    df = pd.read_json(data['input'])
    y_probs = clf.predict_proba(df[feats])
    # Remember, the output from predict_proba is a column for each class
    for i, label in enumerate(clf.classes_):
        df['y_prob_%s' % label] = y_probs[:,i]
    return jsonify({'output':df.to_json()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)