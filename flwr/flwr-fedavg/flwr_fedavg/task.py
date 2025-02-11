"""flwr-clusterization: A Flower / TensorFlow app."""

import os
import json
import numpy as np

import keras
from keras import layers
from keras.regularizers import L1L2

# from flwr_datasets import FederatedDataset
# from flwr_datasets.partitioner import IidPartitioner


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def load_datafile(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def load_model():
    # Define a simple CNN for CIFAR-10 and set Adam optimizer
    # model = keras.Sequential(
    #     [
    #         keras.Input(shape=(32, 32, 3)),
    #         layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    #         layers.MaxPooling2D(pool_size=(2, 2)),
    #         layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    #         layers.MaxPooling2D(pool_size=(2, 2)),
    #         layers.Flatten(),
    #         layers.Dropout(0.5),
    #         layers.Dense(10, activation="softmax"),
    #     ]
    # )
    # model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    # return model
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(125,1)))
    model.add(
        keras.layers.Bidirectional(
            keras.layers.LSTM(
                128, 
                return_sequences=True, 
                kernel_regularizer=L1L2(l1=0.0001, l2=0.0001)
            )
        )
    )
    model.add(keras.layers.Dense(1))

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer, 
        loss="mean_squared_error", 
        metrics=[keras.metrics.RootMeanSquaredError(name="rmse")]
    )
    return model


def load_data(partition_id):
    data = load_datafile("../../data/mimic2_dataset.json")
    client_id = int(partition_id)
    assert client_id < len(data) and client_id >= 0, "Invalid Client ID"
    x_train, y_train = data[client_id]['train']['input_series'], data[client_id]['train']['output_series']
    x_val, y_val = data[client_id]['validation']['input_series'], data[client_id]['validation']['output_series']
    x_test, y_test = data[client_id]['test']['input_series'], data[client_id]['test']['output_series']
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    return x_train, y_train, x_test, y_test


    # Download and partition dataset
    # Only initialize `FederatedDataset` once
    # global fds
    # if fds is None:
    #     partitioner = IidPartitioner(num_partitions=num_partitions)
    #     fds = FederatedDataset(
    #         dataset="uoft-cs/cifar10",
    #         partitioners={"train": partitioner},
    #     )
    # partition = fds.load_partition(partition_id, "train")
    # partition.set_format("numpy")

    # # Divide data on each node: 80% train, 20% test
    # partition = partition.train_test_split(test_size=0.2)
    # x_train, y_train = partition["train"]["img"] / 255.0, partition["train"]["label"]
    # x_test, y_test = partition["test"]["img"] / 255.0, partition["test"]["label"]
    # return x_train, y_train, x_test, y_test
