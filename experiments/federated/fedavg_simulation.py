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

# Configuration
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 1  # Local epochs per round
N_NEURONS = 128
NUM_ROUNDS = 50
NUM_CLIENTS = 2
DATA_PATH = "../data/mimic2_dataset.json"

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregates metrics weighted by number of samples."""
    accuracies = [num_examples * m["rmse"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"rmse": sum(accuracies) / sum(examples)}

class BiLSTMClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        
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
        return [np.array(layer.get_weights()) for layer in self.model.get_model().layers]

    def set_parameters(self, parameters):
        for layer, weights in zip(self.model.get_model().layers, parameters):
            layer.set_weights(weights)

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
    
    # The `evaluate` function will be called after every round
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]):
        model.get_model().set_weights(parameters)  # Update model with the latest parameters
        loss, rmse = model.get_model().evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
        
        # Log metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_dict = {
            "round": server_round,
            "loss": loss,
            "rmse": rmse
        }
        
        # Save metrics to CSV
        pd.DataFrame([metrics_dict]).to_csv(
            f'results/federated/simulation_metrics_{timestamp}_round_{server_round}.csv',
            index=False,
            mode='a',
            header=not pd.io.common.file_exists(f'results/federated/simulation_metrics_{timestamp}_round_{server_round}.csv')
        )
        
        return loss, {"rmse": rmse}
    
    return evaluate

def main():
    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=get_evaluate_fn(),  # Pass the evaluation function
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