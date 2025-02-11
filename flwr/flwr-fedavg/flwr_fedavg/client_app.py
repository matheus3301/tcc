"""flwr-clusterization: A Flower / TensorFlow app."""

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
import os
import pandas as pd
from datetime import datetime

from flwr_fedavg.task import load_data, load_model
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(
        self, client_id, model, data, epochs, batch_size, verbose
    ):
        self.model = model
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.current_round = 0
        self.train_history = []
        self.eval_history = []
        
        # Create results directory with timestamp
        self.results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                                      "results", "flwr", "fedavg", "tmp")

        # if os.path.exists(self.results_dir):
        #     import shutil
        #     shutil.rmtree(self.results_dir)
        # os.makedirs(self.results_dir)
        
        # Get client ID from context when available
        self.client_id = client_id
        
        # Create client directory
        self.client_dir = os.path.join(self.results_dir, f"client_{self.client_id}")
        os.makedirs(self.client_dir, exist_ok=True)

        self.client_sample_dir = os.path.join(self.client_dir, 'samples')
        os.makedirs(self.client_sample_dir, exist_ok=True)

    def fit(self, parameters, config):
        self.current_round = config["current_round"]
        
        # Update client_id if available in config
        self.client_dir = os.path.join(self.results_dir, f"client_{self.client_id}")
        os.makedirs(self.client_dir, exist_ok=True)
            
        self.model.set_weights(parameters)
        history = self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            validation_data=(self.x_val, self.y_val),
        )

        # Store training metrics for this round
        train_metrics = {
            "round": self.current_round,
            "loss": float(history.history['loss'][-1]),
            "rmse": float(history.history['rmse'][-1]),
            "val_loss": float(history.history['val_loss'][-1]),
            "val_rmse": float(history.history['val_rmse'][-1])
        }
        
        self.train_history.append(train_metrics)
        
        # Save training metrics to CSV
        train_metrics_file = os.path.join(self.client_dir, 'train_metrics_history.csv')
        if os.path.exists(train_metrics_file):
            pd.DataFrame([train_metrics]).to_csv(train_metrics_file, mode='a', header=False, index=False)
        else:
            pd.DataFrame([train_metrics]).to_csv(train_metrics_file, index=False)

        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.current_round = config["current_round"]
        self.model.set_weights(parameters)
        loss, rmse = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        
        # Store evaluation metrics for this round
        eval_metrics = {
            "round": self.current_round,
            "test_loss": loss,
            "test_rmse": rmse,
        }
        
        self.eval_history.append(eval_metrics)
        
        # Save evaluation metrics to CSV
        eval_metrics_file = os.path.join(self.client_dir, 'eval_metrics_history.csv')
        if os.path.exists(eval_metrics_file):
            pd.DataFrame([eval_metrics]).to_csv(eval_metrics_file, mode='a', header=False, index=False)
        else:
            pd.DataFrame([eval_metrics]).to_csv(eval_metrics_file, index=False)


        if self.current_round % 10 == 0:
            # Use test data for predictions
            input_data = self.x_test
            expected_output = self.y_test
            generated_output = self.model.predict(input_data, batch_size=input_data.shape[0])
            
            # Save sample predictions
            with open(os.path.join(self.client_sample_dir, f'input_round_{self.current_round}.json'), 'w') as f:
                json.dump(input_data.tolist(), f, cls=NumpyEncoder)
            
            with open(os.path.join(self.client_sample_dir, f'expected_output_round_{self.current_round}.json'), 'w') as f:
                json.dump(expected_output.tolist(), f, cls=NumpyEncoder)
            
            with open(os.path.join(self.client_sample_dir, f'output_round_{self.current_round}.json'), 'w') as f:
                json.dump(generated_output.tolist(), f, cls=NumpyEncoder)
            
        return loss, len(self.x_test), {"rmse": rmse}


def client_fn(context: Context):
    # Load model and data
    net = load_model()

    partition_id = context.node_config["partition-id"]
    # num_partitions = context.node_config["num-partitions"]
    data = load_data(partition_id)
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")

    # Return Client instance
    return FlowerClient(
        partition_id, net, data, epochs, batch_size, verbose
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)
