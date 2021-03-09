# Standard lib imports
import warnings
import sys
import pathlib
import logging

# imports
import numpy as np
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn

# Settings
MODEL_DIR = pathlib.Path('models/')
DATA_DIR = pathlib.Path('data/')
SEED = 43

# Logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Evaluation metrics
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# Main function
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    np.random.seed(SEED)

    # Get training data
    try:
        train_data = np.load(DATA_DIR.joinpath('train_no_log.npy'))
        test_data = np.load(DATA_DIR.joinpath('test_no_log.npy'))
    except Exception as e:
        logger.exception(f'Unable to load data. Error: {e}')

    # Separate targets and features
    x_train = train_data[:, 1:]
    y_train = train_data[:, 0].reshape(-1, 1)
    x_test = test_data[:, 1:]
    y_test = test_data[:, 0].reshape(-1, 1)


    # Normalize X
    scaler_x = MinMaxScaler()
    scaler_x.fit(x_train)
    x_train = scaler_x.transform(x_train)
    x_test = scaler_x.transform(x_test)

    # Normalize y
    scaler_y = MinMaxScaler()
    scaler_y.fit(y_train.reshape(-1, 1))
    y_train = scaler_y.transform(y_train)

    # Command line arguments
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    # Run mlflow process
    with mlflow.start_run():
        # Instantiate model
        rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1,
                                   random_state=SEED, criterion='mse',
                                   verbose=2)
        # Fit model
        rf.fit(x_train, y_train)

        # Predict on test set
        y_pred = scaler_y.inverse_transform(rf.predict(x_test).reshape(-1, 1))

        # Evaluate metrics
        (rmse, mae, r2) = eval_metrics(y_test, y_pred)

        # Print metrics
        print(f'Random Forest model: (n_estimators={n_estimators}')
        print(f'  RMSE: {rmse}')
        print(f'  MAE: {mae}')
        print(f'  R2: {r2}')

        # Log params and metrics
        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('r2', r2)
        mlflow.log_metric('mae', mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != 'file':

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(rf, 'model', registered_model_name='RandomForestSOC')
        else:
            mlflow.sklearn.log_model(rf, 'model')

        model_name = 'rf_1'
        # Save model to disk
        mlflow.sklearn.save_model(rf, MODEL_DIR.joinpath(f'{model_name}'),
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
