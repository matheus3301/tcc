import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
from models.bilstm import BiLSTM
from tensorflow.keras.regularizers import L1L2
from helpers.load_data import load_all_data
from datetime import datetime, timedelta
import pandas as pd
from helpers.numpy_serializer import NumpyEncoder
import json
import numpy as np
import psutil
import time
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import humanize

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

BATCH_SIZE=64
LEARNING_RATE=0.001
EPOCHS=250
N_NEURONS=128
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "mimic2_dataset.json")

class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_times = []
        self.start_time = None
        
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.start_time
        self.epoch_times.append(epoch_time)
        
def get_model_size(model):
    # Save model to a temporary file to get its size
    temp_path = 'temp_model'
    model.save(temp_path)
    size_bytes = sum(os.path.getsize(os.path.join(temp_path, f)) for f in os.listdir(temp_path))
    import shutil
    shutil.rmtree(temp_path)
    return size_bytes

def get_gpu_memory():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            return "No GPU available"
        return tf.config.experimental.get_memory_info('GPU:0')
    except:
        return "GPU memory info not available"

def main():
    logger.info("Starting the experiment")
    
    # Initialize metrics dictionary
    metrics = {
        'memory': [],
        'timestamps': [],
        'start_time': time.time()
    }
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss

    # Create timestamped directory for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join('results', 'centralized', timestamp)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'sample'), exist_ok=True)

    # Load data
    load_start_time = time.time()
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_all_data(DATA_PATH)
    data_load_time = time.time() - load_start_time
    
    regularizer = L1L2(l1=0.0001, l2=0.0001)
    model = BiLSTM(
            input_shape=(x_train.shape[1], 1),
            n_neurons=N_NEURONS,
            regularizer=regularizer,
            learning_rate=LEARNING_RATE,
        )
    model.compile()
    
    # Get model information
    trainable_params = np.sum([np.prod(v.get_shape()) for v in model.get_model().trainable_variables])
    non_trainable_params = np.sum([np.prod(v.get_shape()) for v in model.get_model().non_trainable_variables])
    
    # Initialize callback
    metrics_callback = MetricsCallback()
    
    # Training start time
    train_start_time = time.time()
    
    history = model.get_model().fit(
            x_train.reshape(x_train.shape[0], x_train.shape[1], 1),
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=True,
            verbose=True,
            validation_data=(x_test, y_test),
            callbacks=[metrics_callback]
        )
    
    # Calculate training time
    total_training_time = time.time() - train_start_time
    
    # Get peak memory usage
    peak_memory = process.memory_info().rss
    memory_increase = peak_memory - initial_memory
    
    # Get model size
    model_size_bytes = get_model_size(model.get_model())
    
    # Save the model history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(results_dir, 'history.csv'), index=False)
    
    # Calculate evaluation metrics
    train_result = model.get_model().predict(x_train, batch_size=1)
    loss, rmse = model.get_model().evaluate(x_train, y_train, batch_size=1, verbose=0)
    
    # Compile all metrics
    performance_metrics = {
        "fit_rmse": float(rmse),
        "fit_loss": float(loss),
        "training_time_seconds": total_training_time,
        "data_loading_time_seconds": data_load_time,
        "peak_memory_bytes": peak_memory,
        "peak_memory_formatted": humanize.naturalsize(peak_memory),
        "memory_increase_bytes": memory_increase,
        "memory_increase_formatted": humanize.naturalsize(memory_increase),
        "model_size_bytes": model_size_bytes,
        "model_size_formatted": humanize.naturalsize(model_size_bytes),
        "trainable_parameters": int(trainable_params),
        "non_trainable_parameters": int(non_trainable_params),
        "total_parameters": int(trainable_params + non_trainable_params),
        "average_epoch_time": np.mean(metrics_callback.epoch_times),
        "gpu_memory_info": str(get_gpu_memory()),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "n_neurons": N_NEURONS
    }
    
    # Save detailed metrics
    pd.DataFrame([performance_metrics]).to_csv(os.path.join(results_dir, 'performance_metrics.csv'), index=False)
    
    # Save epoch times
    pd.DataFrame({
        'epoch': range(1, len(metrics_callback.epoch_times) + 1),
        'time_seconds': metrics_callback.epoch_times
    }).to_csv(os.path.join(results_dir, 'epoch_times.csv'), index=False)

    input_data = x_test
    expected_output = y_test

    # Use the model to generate the output
    generated_output = model.get_model().predict(input_data, batch_size=input_data.shape[0])
    generated_output = np.reshape(generated_output, (generated_output.shape[0], generated_output.shape[1]))

    print(f"input_data: {input_data.shape}")
    print(f"expected_output: {expected_output.shape}")
    print(f"generated_output: {generated_output.shape}")

    # Save the input, expected output, and generated output in three json files
    with open(os.path.join(results_dir, 'sample', 'input.json'), 'w') as f:
        json.dump(input_data.tolist(), f, cls=NumpyEncoder)

    with open(os.path.join(results_dir, 'sample', 'expected_output.json'), 'w') as f:
        json.dump(expected_output.tolist(), f, cls=NumpyEncoder)

    with open(os.path.join(results_dir, 'sample', 'output.json'), 'w') as f:
        json.dump(generated_output.tolist(), f, cls=NumpyEncoder)

if __name__ == "__main__":
    main()

