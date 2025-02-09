import sys
import os
# import ray
# ray.init(address='auto', runtime_env={"working_dir": "./"})
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics
import numpy as np
from datetime import datetime
import pandas as pd
import tensorflow as tf
from models.bilstm import BiLSTM
from tensorflow.keras.regularizers import L1L2
from helpers.load_data import load_data
import json
from helpers.numpy_serializer import NumpyEncoder

# Create base results directory if it doesn't exist
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "fedavg+interpolation")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Create timestamped directory for this run
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(RESULTS_DIR, TIMESTAMP)
os.makedirs(RUN_DIR, exist_ok=True)
# os.makedirs(os.path.join(RUN_DIR, 'sample'), exist_ok=True)

# Values closer to 1.0 will give more weight to the local model
# Values closer to 0.0 will give more weight to the federated model
INTERPOLATION_WEIGHT = 0.6
# Configuration
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 1  # Local epochs per round
N_NEURONS = 128
NUM_ROUNDS = 100
NUM_CLIENTS = 2
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "mimic2_dataset.json")

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregates metrics weighted by number of samples."""
    accuracies = [num_examples * m["rmse"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"rmse": sum(accuracies) / sum(examples)}

class BiLSTMClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.interpolation_weight = INTERPOLATION_WEIGHT  # Weight for interpolation between local and federated models
        
        # Create client-specific sample directory
        self.client_sample_dir = os.path.join(RUN_DIR, f'client_{self.client_id}', 'sample')
        os.makedirs(self.client_sample_dir, exist_ok=True)
        
        # Load data for this client
        (self.x_train, self.y_train), (self.x_val, self.y_val), (self.x_test, self.y_test) = load_data(
            client_id=self.client_id,
            path=DATA_PATH,
            segment_len=None
        )
        
        # Reshape input data
        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], 1)
        self.x_val = self.x_val.reshape(self.x_val.shape[0], self.x_val.shape[1], 1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], 1)
        
        # Create regularizer
        regularizer = L1L2(l1=0.0001, l2=0.0001)
        
        # Create federated model (participates in federation)
        self.federated_model = BiLSTM(
            input_shape=(self.x_train.shape[1], 1),
            n_neurons=N_NEURONS,
            regularizer=regularizer,
            learning_rate=LEARNING_RATE,
        )
        self.federated_model.compile()
        
        # Create local model (trains only on local data)
        self.local_model = BiLSTM(
            input_shape=(self.x_train.shape[1], 1),
            n_neurons=N_NEURONS,
            regularizer=regularizer,
            learning_rate=LEARNING_RATE,
        )
        self.local_model.compile()

    def get_parameters(self, config):
        # Only share federated model parameters
        weights = self.federated_model.get_model().get_weights()
        return [np.array(w) for w in weights]

    def set_parameters(self, parameters):
        # Only update federated model parameters
        self.federated_model.get_model().set_weights(parameters)

    def interpolate_predictions(self, x_data):
        """Combine predictions from local and federated models using interpolation."""
        local_pred = self.local_model.get_model().predict(x_data, batch_size=x_data.shape[0])
        federated_pred = self.federated_model.get_model().predict(x_data, batch_size=x_data.shape[0])
        # Ensure predictions have the same shape as target data by squeezing the last dimension
        local_pred = np.squeeze(local_pred, axis=-1)
        federated_pred = np.squeeze(federated_pred, axis=-1)
        return (self.interpolation_weight * local_pred + 
                (1 - self.interpolation_weight) * federated_pred)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # Train federated model
        self.federated_model.get_model().fit(
            self.x_train,
            self.y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(self.x_val, self.y_val),
            verbose=0
        )
        
        # Train local model independently
        self.local_model.get_model().fit(
            self.x_train,
            self.y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(self.x_val, self.y_val),
            verbose=0
        )
        
        # Get current round from config
        current_round = config.get("current_round", 0)
        
        # Every 10 rounds, save sample predictions
        if current_round % 10 == 0 or current_round == NUM_ROUNDS:
            # Use test data for predictions
            input_data = self.x_test[0:5]
            expected_output = self.y_test[0:5]
            generated_output = self.interpolate_predictions(input_data)
            
            # Save sample predictions
            with open(os.path.join(self.client_sample_dir, f'input_round_{current_round}.json'), 'w') as f:
                json.dump(input_data.tolist(), f, cls=NumpyEncoder)
            
            with open(os.path.join(self.client_sample_dir, f'expected_output_round_{current_round}.json'), 'w') as f:
                json.dump(expected_output.tolist(), f, cls=NumpyEncoder)
            
            with open(os.path.join(self.client_sample_dir, f'output_round_{current_round}.json'), 'w') as f:
                json.dump(generated_output.tolist(), f, cls=NumpyEncoder)
        
        return self.get_parameters(config), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        # Make predictions using interpolated model
        predictions = self.interpolate_predictions(self.x_test)
        
        # Calculate RMSE manually since we're using interpolated predictions
        mse = np.mean(np.square(predictions - self.y_test))
        rmse = np.sqrt(mse)
        
        # Also calculate individual RMSEs for local and federated models for comparison
        local_pred = np.squeeze(self.local_model.get_model().predict(self.x_test, batch_size=self.x_test.shape[0]), axis=-1)
        federated_pred = np.squeeze(self.federated_model.get_model().predict(self.x_test, batch_size=self.x_test.shape[0]), axis=-1)
        
        local_rmse = np.sqrt(np.mean(np.square(local_pred - self.y_test)))
        federated_rmse = np.sqrt(np.mean(np.square(federated_pred - self.y_test)))
        
        return mse, len(self.x_test), {
            "rmse": rmse,
            "local_rmse": local_rmse,
            "federated_rmse": federated_rmse
        }

def client_fn(cid: str) -> fl.client.Client:
    """Creates a Flower client representing a single organization."""
    return BiLSTMClient(client_id=int(cid))

def get_evaluate_fn():
    """Returns an evaluation function for server-side evaluation."""
    
    # Load test data
    _, _, (x_test, y_test) = load_data(
        client_id=0,  # Use first client's test data for server-side evaluation
        path=DATA_PATH,
        segment_len=None
    )
    
    # Reshape test data
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    
    # Create model
    regularizer = L1L2(l1=0.0001, l2=0.0001)
    model = BiLSTM(
        input_shape=(x_test.shape[1], 1),
        n_neurons=N_NEURONS,
        regularizer=regularizer,
        learning_rate=LEARNING_RATE,
    )
    model.compile()
    
    # Store metrics across rounds
    metrics_history = []
    
    # The `evaluate` function will be called after every round
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]):
        model.get_model().set_weights(parameters)  # Update model with the latest parameters
        loss, rmse = model.get_model().evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
        
        # Store metrics
        metrics_dict = {
            "round": server_round,
            "loss": float(loss),
            "rmse": float(rmse)
        }
        metrics_history.append(metrics_dict)
        
        # Save metrics history to CSV
        pd.DataFrame(metrics_history).to_csv(
            os.path.join(RUN_DIR, 'history.csv'),
            index=False
        )
        
        return loss, {"rmse": rmse}
    
    return evaluate

def main():
    # Create strategy with custom configuration
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=get_evaluate_fn(),
        on_fit_config_fn=lambda server_round: {"current_round": server_round}  # Pass current round to clients
    )

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

if __name__ == "__main__":
    main() 