# flask_api_template

A simple template project for delivering a model as a service via flask. A basic RESTful API, with data being pushed around in JSON format. The model is a RandomForestClassifier from scikit-learn and is trained on the Iris dataset.

## Operation

1. Create flask_api environment on your local machine: `conda env create -f flask_api.yml`
2. Activate conda environment: `(source) activate flask_api`
3. Run app.py: `python app.py`

At this point, you can either access the classes_ route via web browser (e.g., browse to http://localhost:8000/classes_) or use [call_flask_app](https://github.com/pilotneko/flask_api_template/blob/master/call_flask_app.ipynb) to predict/predict_proba/classes_ routes.
