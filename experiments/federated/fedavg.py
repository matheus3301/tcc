import flwr as fl
import tensorflow as tf
from models.bilstm import BiLSTM
from tensorflow.keras.regularizers import L1L2
from helpers.load_data import load_data
import numpy as np

# Configuration
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 1  # Local epochs per round
N_NEURONS = 128
DATA_PATH = "../data/mimic2_dataset.json"

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
        
        # Return updated model parameters and number of training examples
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

def client_fn(client_id):
    return BiLSTMClient(client_id)

def main():
    # Start Flower client
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client_fn(fl.common.DEVICE.index)
    )

if __name__ == "__main__":
    main()
