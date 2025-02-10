import sys
import os
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
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "fedper")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Create timestamped directory for this run
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(RESULTS_DIR, TIMESTAMP)
os.makedirs(RUN_DIR, exist_ok=True)

# Configuration
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10  # Local epochs per round
N_NEURONS = 128
NUM_ROUNDS = 50
NUM_CLIENTS = 10
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "mimic2_dataset.json")

# Values closer to 1.0 will give more weight to the local model
# Values closer to 0.0 will give more weight to the federated model
INTERPOLATION_WEIGHT = 0.8

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregates metrics weighted by number of samples."""
    accuracies = [num_examples * m["rmse"] for num_examples, m in metrics]
    loss = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"rmse": sum(accuracies) / sum(examples), "loss": sum(loss) / sum(examples)}

def get_gpu_memory():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            return "No GPU available"
        return tf.config.experimental.get_memory_info('GPU:0')
    except:
        return "GPU memory info not available"

class BiLSTMClientFedPer(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.interpolation_weight = INTERPOLATION_WEIGHT
        
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
        
        # Create regularizer
        regularizer = L1L2(l1=0.0001, l2=0.0001)
        
        # Create federated model
        self.federated_model = BiLSTM(
            input_shape=(self.x_train.shape[1], 1),
            n_neurons=N_NEURONS,
            regularizer=regularizer,
            learning_rate=LEARNING_RATE,
        )
        self.federated_model.compile()
        
        # Create local model
        self.local_model = BiLSTM(
            input_shape=(self.x_train.shape[1], 1),
            n_neurons=N_NEURONS,
            regularizer=regularizer,
            learning_rate=LEARNING_RATE,
        )
        self.local_model.compile()

    def get_parameters(self, config):
        # Get all model weights from federated model
        weights = self.federated_model.get_model().get_weights()
        # Return only BiLSTM layers (excluding the Dense layer)
        return weights[:-2]  # Last 2 weights are Dense layer's weights and biases

    def set_parameters(self, parameters):
        # Get current federated model weights
        current_weights = self.federated_model.get_model().get_weights()
        # Keep the Dense layer weights (personalized)
        personalized_weights = current_weights[-2:]  # Dense layer's weights and biases
        # Combine shared parameters with personalized weights
        new_weights = parameters + personalized_weights
        # Set the combined weights
        self.federated_model.get_model().set_weights(new_weights)

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
        
        # Track training time and memory
        start_time = time.time()
        start_memory = self.process.memory_info().rss
        
        # Train federated model
        fed_history = self.federated_model.get_model().fit(
            self.x_train,
            self.y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(self.x_val, self.y_val),
            verbose=0
        )
        
        # Train local model independently
        local_history = self.local_model.get_model().fit(
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
        
        # Calculate interpolated predictions
        interpolated_train_rmse = float(fed_history.history['loss'][-1]) * self.interpolation_weight + float(local_history.history['loss'][-1]) * (1 - self.interpolation_weight)
        interpolated_train_loss = float(fed_history.history['loss'][-1]) * self.interpolation_weight + float(local_history.history['loss'][-1]) * (1 - self.interpolation_weight)
        interpolated_val_rmse = float(fed_history.history['val_loss'][-1]) * self.interpolation_weight + float(local_history.history['val_loss'][-1]) * (1 - self.interpolation_weight)
        interpolated_val_loss = float(fed_history.history['val_loss'][-1]) * self.interpolation_weight + float(local_history.history['val_loss'][-1]) * (1 - self.interpolation_weight)

        # Store metrics for this round
        round_metrics = {
            "round": current_round,
            "training_time_seconds": training_time,
            "memory_usage_bytes": current_memory,
            "memory_usage_formatted": humanize.naturalsize(current_memory),
            "memory_increase_bytes": memory_increase,
            "memory_increase_formatted": humanize.naturalsize(memory_increase),
            "gpu_memory_info": str(get_gpu_memory()),
            "fed_train_loss": float(fed_history.history['loss'][-1]),
            "fed_val_loss": float(fed_history.history['val_loss'][-1]),
            "fed_train_rmse": float(fed_history.history['rmse'][-1]),
            "fed_val_rmse": float(fed_history.history['val_rmse'][-1]),
            "local_train_loss": float(local_history.history['loss'][-1]),
            "local_val_loss": float(local_history.history['val_loss'][-1]),
            "local_train_rmse": float(local_history.history['rmse'][-1]),
            "local_val_rmse": float(local_history.history['val_rmse'][-1]),
            "interpolated_train_rmse": float(interpolated_train_rmse),
            "interpolated_train_loss": float(interpolated_train_loss),
            "interpolated_val_rmse": float(interpolated_val_rmse),
            "interpolated_val_loss": float(interpolated_val_loss)
        }
        
        self.metrics_history.append(round_metrics)
        self.training_times.append(training_time)
        
        # Save metrics to CSV
        metrics_file = os.path.join(self.client_dir, 'metrics_history.csv')
        if os.path.exists(metrics_file):
            pd.DataFrame([self.metrics_history[-1]]).to_csv(metrics_file, mode='a', header=False, index=False)
        else:
            pd.DataFrame(self.metrics_history).to_csv(metrics_file, index=False)
        
        # Every 10 rounds, save sample predictions
        if current_round % 10 == 0 or current_round == NUM_ROUNDS:
            # Use test data for predictions
            input_data = self.x_test
            expected_output = self.y_test
            generated_output = self.interpolate_predictions(input_data)
            
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
    return BiLSTMClientFedPer(client_id=int(cid))

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
        
        # For server-side evaluation in FedPer, we'll use the shared BiLSTM layers
        # but initialize a random Dense layer since it's personalized per client
        current_weights = model.get_model().get_weights()
        personalized_weights = current_weights[-2:]  # Keep current Dense layer
        new_weights = parameters + personalized_weights
        model.get_model().set_weights(new_weights)
        
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
