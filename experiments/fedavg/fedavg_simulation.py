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
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "fedavg")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Create timestamped directory for this run
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(RESULTS_DIR, TIMESTAMP)
os.makedirs(RUN_DIR, exist_ok=True)
# os.makedirs(os.path.join(RUN_DIR, 'sample'), exist_ok=True)

# Configuration
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 1  # Local epochs per round
N_NEURONS = 128
NUM_ROUNDS = 250
NUM_CLIENTS = 3
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "mimic2_dataset.json")

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregates metrics weighted by number of samples."""
    accuracies = [num_examples * m["rmse"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"rmse": sum(accuracies) / sum(examples)}

class BiLSTMClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        
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
        
        # Create model
        regularizer = L1L2(l1=0.0001, l2=0.0001)
        self.model = BiLSTM(
            input_shape=(self.x_train.shape[1], 1),
            n_neurons=N_NEURONS,
            regularizer=regularizer,
            learning_rate=LEARNING_RATE,
        )
        self.model.compile()

    def get_parameters(self, config):
        weights = self.model.get_model().get_weights()
        return [np.array(w) for w in weights]

    def set_parameters(self, parameters):
        self.model.get_model().set_weights(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        history = self.model.get_model().fit(
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
            generated_output = self.model.get_model().predict(input_data, batch_size=input_data.shape[0])
            
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
        
        loss, rmse = self.model.get_model().evaluate(
            self.x_test,
            self.y_test,
            batch_size=BATCH_SIZE,
            verbose=0
        )
        
        return loss, len(self.x_test), {"rmse": rmse}

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