import os
import warnings
import sys
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import logging
import yaml
from mlflow.models.signature import infer_signature


## Logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

mlflow.set_experiment("ElasticHeartDiseaseModel")

## Evaluation Metrics

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    mse = mean_squared_error(actual, pred)

    return rmse, mae, r2, mse



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

with open("config.yaml", "r") as f:
    data = yaml.safe_load(f)
    data_path = data["data_source"]



try:
        data_frame = pd.read_csv(data_path, sep=",")
except Exception as e:
        logger.exception(
            "Unable to get training & test CSV, check your internet connection. Error: %s", e
        )


# Split the data into training and test sets. (0.75, 0.25) split.
train, test = train_test_split(data_frame)

with mlflow.start_run(run_name="ExperElasticHeartDiseaseModel") as run:
    # The predicted column is "quality" which is a scalar from [3, 9]
        train_x = train.drop(["target"], axis=1)
        test_x = test.drop(["target"], axis=1)
        train_y = train[["target"]]
        test_y = test[["target"]]

        alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
        l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5


        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_disease = lr.predict(test_x)

        (rmse, mae, r2, mse) = eval_metrics(test_y, predicted_disease)

        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)


        predictions = lr.predict(train_x)

        mlflow.sklearn.log_model(lr,  artifact_path="model")
        signature = infer_signature(train_x, predictions)
        mlflow.log_metric("mse", mse)

        run_id = run.info.run_uuid
        experiment_id = run.info.experiment_id

        print(f'artifact_uri={mlflow.get_artifact_uri()}')
        print(f'run_id={run_id}')
        print(f'experiment_id={experiment_id}')







