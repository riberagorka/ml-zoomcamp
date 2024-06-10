from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import pickle
import requests


if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

def check_mlflow_server(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print(f"MLflow server is running at {url}")
        else:
            print(f"Received unexpected status code {response.status_code} from {url}")
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to MLflow server: {e}")

@transformer
def transform(df, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    print("Starting the block...")
    mlflow_url = 'http://localhost:5000'
    check_mlflow_server(mlflow_url)

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    train_dicts = df[categorical].to_dict(orient='records')

    vec = DictVectorizer()
    print("Vectorizing...")
    X_train = vec.fit_transform(train_dicts)

    target = 'duration'
    y_train = df[target].values
    print("Starting the model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("model fitted")
     # Log the model using MLflow
    
    mlflow.set_tracking_uri(mlflow_url)
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, "linear_regression_model")
        print("Model logged to MLflow")

    # Create a DictVectorizer instance
    vec = DictVectorizer()

    # Save and log the artifact
    artifact_path = "dict_vectorizer.pkl"
    with open(artifact_path, "wb") as f:
        import pickle
        pickle.dump(vec, f)

    mlflow.log_artifact(artifact_path)
    print("Artifact logged to MLflow")

    return model, vec
