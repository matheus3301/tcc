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
import psutil
import time
import humanize

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
EPOCHS = 10  # Local epochs per round
N_NEURONS = 128
NUM_ROUNDS = 50
NUM_CLIENTS = 10
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "mimic2_dataset.json")

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregates metrics weighted by number of samples."""
    accuracies = [num_examples * m["rmse"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"rmse": sum(accuracies) / sum(examples)}

def get_gpu_memory():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            return "No GPU available"
        return tf.config.experimental.get_memory_info('GPU:0')
    except:
        return "GPU memory info not available"

class BiLSTMClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        
        # Create client-specific directories
        self.client_dir = os.path.join(RUN_DIR, f'client_{self.client_id}')
        self.client_sample_dir = os.path.join(self.client_dir, 'sample')
        os.makedirs(self.client_sample_dir, exist_ok=True)
        
        # Initialize performance tracking
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss
        self.metrics_history = []
        self.training_times = []
        
        # Load data for this client
        data_load_start = time.time()
        (self.x_train, self.y_train), (self.x_val, self.y_val), (self.x_test, self.y_test) = load_data(
            client_id=self.client_id,
            path=DATA_PATH,
            segment_len=None
        )
        self.data_load_time = time.time() - data_load_start
        
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
        
        # Track training time and memory
        start_time = time.time()
        start_memory = self.process.memory_info().rss
        
        history = self.model.get_model().fit(
            self.x_train,
            self.y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(self.x_val, self.y_val),
            verbose=0
        )
        
        # Calculate metrics
        training_time = time.time() - start_time
        current_memory = self.process.memory_info().rss
        memory_increase = current_memory - start_memory
        
        # Get current round from config
        current_round = config.get("current_round", 0)
        
        # Store metrics for this round
        round_metrics = {
            "round": current_round,
            "training_time_seconds": training_time,
            "memory_usage_bytes": current_memory,
            "memory_usage_formatted": humanize.naturalsize(current_memory),
            "memory_increase_bytes": memory_increase,
            "memory_increase_formatted": humanize.naturalsize(memory_increase),
            "gpu_memory_info": str(get_gpu_memory()),
            "train_loss": float(history.history['loss'][-1]),
            "val_loss": float(history.history['val_loss'][-1]),
            "train_rmse": float(history.history['rmse'][-1]),
            "val_rmse": float(history.history['val_rmse'][-1])
        }
        
        self.metrics_history.append(round_metrics)
        self.training_times.append(training_time)
        
        # Save metrics to CSV
        pd.DataFrame(self.metrics_history).to_csv(
            os.path.join(self.client_dir, 'metrics_history.csv'),
            index=False,
            mode='a',
            header=not os.path.exists(os.path.join(self.client_dir, 'metrics_history.csv'))
        )
        
        # Every 10 rounds, save sample predictions
        if current_round % 10 == 0 or current_round == NUM_ROUNDS:
            # Use test data for predictions
            input_data = self.x_test
            expected_output = self.y_test
            generated_output = self.model.get_model().predict(input_data, batch_size=input_data.shape[0])
            
            # Save sample predictions
            with open(os.path.join(self.client_sample_dir, f'input_round_{current_round}.json'), 'w') as f:
                json.dump(input_data.tolist(), f, cls=NumpyEncoder)
            
            with open(os.path.join(self.client_sample_dir, f'expected_output_round_{current_round}.json'), 'w') as f:
                json.dump(expected_output.tolist(), f, cls=NumpyEncoder)
            
            with open(os.path.join(self.client_sample_dir, f'output_round_{current_round}.json'), 'w') as f:
                json.dump(generated_output.tolist(), f, cls=NumpyEncoder)
        
        return self.get_parameters(config), len(self.x_train), round_metrics

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
    data_load_start = time.time()
    _, _, (x_test, y_test) = load_data(
        client_id=0,  # Use first client's test data for server-side evaluation
        path=DATA_PATH,
        segment_len=None
    )
    data_load_time = time.time() - data_load_start
    
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
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # The `evaluate` function will be called after every round
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]):
        # Track evaluation time and memory
        start_time = time.time()
        start_memory = process.memory_info().rss
        
        model.get_model().set_weights(parameters)  # Update model with the latest parameters
        loss, rmse = model.get_model().evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
        
        # Calculate metrics
        eval_time = time.time() - start_time
        current_memory = process.memory_info().rss
        memory_increase = current_memory - start_memory
        
        # Store metrics
        metrics_dict = {
            "round": server_round,
            "loss": float(loss),
            "rmse": float(rmse),
            "evaluation_time_seconds": eval_time,
            "memory_usage_bytes": current_memory,
            "memory_usage_formatted": humanize.naturalsize(current_memory),
            "memory_increase_bytes": memory_increase,
            "memory_increase_formatted": humanize.naturalsize(memory_increase),
            "gpu_memory_info": str(get_gpu_memory())
        }
        metrics_history.append(metrics_dict)
        
        # Save metrics history to CSV
        pd.DataFrame(metrics_history).to_csv(
            os.path.join(RUN_DIR, 'server_metrics.csv'),
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