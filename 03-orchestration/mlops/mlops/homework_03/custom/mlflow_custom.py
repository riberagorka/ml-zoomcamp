import pandas as pd
import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.from sklearn.feature_extraction import DictVectorizer

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@custom
def transform_custom(df: pd.DataFrame, *args, **kwargs
    ):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your custom logic here

    print("Starting...")

    mlflow.set_tracking_uri(uri="http://mlflow:5000")
    mlflow.experiment_name("lr_model_mage")
    with mlflow.start_run():

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


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
