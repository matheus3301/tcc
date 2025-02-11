import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics, Parameters, FitRes
from flwr.server.client_proxy import ClientProxy
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
from sklearn.cluster import KMeans

# Create base results directory if it doesn't exist
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "fedavg+clusterization")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Create timestamped directory for this run
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(RESULTS_DIR, TIMESTAMP)
os.makedirs(RUN_DIR, exist_ok=True)

# Configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 15
N_NEURONS = 64 
NUM_ROUNDS = 30
NUM_CLIENTS = 5
N_CLUSTERS = 2
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

class ClusteringFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cluster_history = []

    def _parameters_to_ndarrays(self, parameters: Parameters) -> List[np.ndarray]:
        """Convert Parameters to a list of NumPy arrays."""
        return [
            np.frombuffer(param, dtype=np.float32).copy()
            for param in parameters.tensors
        ]

    def _ndarrays_to_parameters(self, ndarrays: List[np.ndarray]) -> Parameters:
        """Convert a list of NumPy arrays to Parameters."""
        tensors = [
            ndarray.astype(np.float32).tobytes()
            for ndarray in ndarrays
        ]
        return Parameters(tensors=tensors, tensor_type="numpy.ndarray")

    def _flatten_weights(self, weights: List[np.ndarray]) -> np.ndarray:
        """Flatten a list of NumPy arrays into a single 1D array."""
        return np.concatenate([w.flatten() for w in weights])

    def _unflatten_weights(self, flattened: np.ndarray, shapes: List[tuple]) -> List[np.ndarray]:
        """Restore a flattened array back to its original shapes."""
        result = []
        idx = 0
        for shape in shapes:
            size = np.prod(shape)
            result.append(flattened[idx:idx + size].reshape(shape))
            idx += size
        return result

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:
        """Aggregate model weights using KMeans clustering."""
        if not results:
            return None, {}
        
        # Convert Parameters to NumPy arrays and extract weights and num_examples
        weights_results = []
        metrics_list = []
        for _, fit_res in results:
            try:
                weights = self._parameters_to_ndarrays(fit_res.parameters)
                weights_results.append((weights, fit_res.num_examples))
                metrics_list.append((fit_res.num_examples, fit_res.metrics))
            except Exception as e:
                print(f"Error converting parameters for a client: {str(e)}")
                continue

        if not weights_results:
            return None, {}

        # Get shapes from the first client's weights for later unflattening
        original_shapes = [w.shape for w in weights_results[0][0]]

        # Prepare data for clustering by flattening weights
        try:
            X = np.array([self._flatten_weights(w) for w, _ in weights_results])
            
            # Normalize the weights to prevent clustering issues
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0)
            X_std[X_std == 0] = 1  # Prevent division by zero
            X_normalized = (X - X_mean) / X_std
            
            # Remove any NaN values that might have been created
            X_normalized = np.nan_to_num(X_normalized)
        except Exception as e:
            print(f"Error flattening weights for clustering: {str(e)}")
            return None, {}
        
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=min(N_CLUSTERS, len(X)), random_state=42)
        cluster_labels = kmeans.fit_predict(X_normalized)

        # Save clustering information
        cluster_info = {
            "round": server_round,
            "cluster_labels": cluster_labels.tolist(),
            "n_clients_per_cluster": np.bincount(cluster_labels).tolist(),
            "weights_stats": {
                "original_mean": float(np.mean(X)),
                "original_std": float(np.std(X)),
                "normalized_mean": float(np.mean(X_normalized)),
                "normalized_std": float(np.std(X_normalized))
            }
        }
        self.cluster_history.append(cluster_info)

        # Print diagnostic information
        print(f"\nRound {server_round} clustering stats:")
        print(f"Number of clients per cluster: {np.bincount(cluster_labels).tolist()}")
        print(f"Original weights stats - Mean: {np.mean(X):.2e}, Std: {np.std(X):.2e}")
        print(f"Normalized weights stats - Mean: {np.mean(X_normalized):.2e}, Std: {np.std(X_normalized):.2e}\n")

        # Save cluster history
        with open(os.path.join(RUN_DIR, 'cluster_history.json'), 'w') as f:
            json.dump(self.cluster_history, f)

        # Aggregate within clusters
        cluster_weights = []
        cluster_importances = []

        for i in range(kmeans.n_clusters):
            # Get indices of clients in this cluster
            cluster_indices = np.where(cluster_labels == i)[0]
            
            if len(cluster_indices) > 0:
                # Extract weights and num_examples for this cluster
                cluster_weights_results = [weights_results[idx] for idx in cluster_indices]
                
                try:
                    # Weighted average within cluster using float64 for higher precision
                    weights_sum = np.sum(
                        [self._flatten_weights(w).astype(np.float64) * float(n) for w, n in cluster_weights_results],
                        axis=0
                    )
                    examples_sum = sum(n for _, n in cluster_weights_results)
                    
                    cluster_weights.append(weights_sum / examples_sum)
                    cluster_importances.append(examples_sum)
                except Exception as e:
                    print(f"Error aggregating weights for cluster {i}: {str(e)}")
                    continue

        if not cluster_weights:
            return None, {}

        try:
            # Final aggregation across clusters (weighted by cluster sizes)
            total_examples = sum(cluster_importances)
            aggregated_weights = np.sum(
                [w * (n / total_examples) for w, n in zip(cluster_weights, cluster_importances)],
                axis=0
            )

            # Unflatten weights back to original shapes
            final_weights = self._unflatten_weights(aggregated_weights, original_shapes)

            # Convert back to Parameters
            parameters_aggregated = self._ndarrays_to_parameters(final_weights)
            
            # Aggregate metrics
            metrics_aggregated = {}
            if metrics_list:
                metrics_aggregated = weighted_average(metrics_list)
            
            # Clear memory
            gc.collect()
            
            return parameters_aggregated, metrics_aggregated
        except Exception as e:
            print(f"Error in final aggregation: {str(e)}")
            return None, {}

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
    strategy = ClusteringFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=get_evaluate_fn(),
        on_fit_config_fn=lambda server_round: {"current_round": server_round}
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
