from typing import Tuple

import pandas as pd

import mlflow
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def transform(
    df: pd.DataFrame, **kwargs
):
    print("Starting...")

    mlflow.set_tracking_uri(uri="http://mlflow:5000")
    mlflow.experiment_name("lr_model_mage")
    with mlflow.start_run():
        categorical = ['PULocationID', 'DOLocationID']
        df[categorical] = df[categorical].astype(str)
        train_dicts = df[categorical].to_dict(orient='records')
        print("Computing vectorization...")
        dv = DictVectorizer()
        X_train = dv.fit_transform(train_dicts)
        print("Vectorization computed")
        target = 'duration'
        y_train = df[target].values
        print("Computing model...")
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        mlflow.sklearn.log_model(sk_model=lr,
                            "linear_regression_model",
                            input_example=X_train)
        mlflow.log_artifact(dv, "dict_vectorizer")

        y_pred = lr.predict(X_train)

    print("Model intercept is: ", lr.intercept_)
    return lr, dv
