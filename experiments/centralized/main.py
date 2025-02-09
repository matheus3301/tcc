import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
from models.bilstm import BiLSTM
from tensorflow.keras.regularizers import L1L2
from helpers.load_data import load_all_data
from datetime import datetime
import pandas as pd
from helpers.numpy_serializer import NumpyEncoder
import json
import numpy as np


logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

BATCH_SIZE=64
LEARNING_RATE=0.001
EPOCHS=250
N_NEURONS=128
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "mimic2_dataset.json")

def main():
    logger.info("Starting the experiment")

    # Create timestamped directory for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join('results', 'centralized', timestamp)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'sample'), exist_ok=True)

    # Load data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_all_data(DATA_PATH)
    
    regularizer = L1L2(l1=0.0001, l2=0.0001)
    model = BiLSTM(
            # input_shape=(x_train.shape[0], x_train.shape[1]),
            input_shape=(x_train.shape[1], 1),
            n_neurons=N_NEURONS,
            regularizer=regularizer,
            learning_rate=LEARNING_RATE,
        )
    model.compile()

    history = model.get_model().fit(
            x_train.reshape(x_train.shape[0], x_train.shape[1], 1),
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=True,
            verbose=True,
            validation_data=(x_test, y_test)
        )

    # Save the model history
    pd.DataFrame(history.history).to_csv(os.path.join(results_dir, 'history.csv'), index=False)

    train_result = model.get_model().predict(x_train, batch_size=1)

    # Calculate evaluation metric
    loss, rmse = model.get_model().evaluate(
        x_train, y_train, batch_size=1, verbose=0
    )
    results = {
        "fit_rmse": float(rmse),
        "fit_loss": float(loss),
    }

    pd.DataFrame(results, index=[0]).to_csv(os.path.join(results_dir, 'results.csv'), index=False)

    input_data = x_test[0:5]
    expected_output = y_test[0:5]

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

