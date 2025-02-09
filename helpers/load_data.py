import numpy as np
import logging
import json

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

def fit_format(data):
    data1 = np.array(data)
    result = data1.reshape(data1.shape[0], data1.shape[1], 1)
    return result

def load_datafile(path: str):
    with open(path, 'r') as f:
        return json.load(f)

def load_data(client_id, path, segment_len, *args, **kwargs):
    """Load federated dataset partition based on client ID.

    Args:
        data_sampling_percentage (float): Percentage of the dataset to use for training.
        client_id (int): Unique ID for the client.
        total_clients (int): Total number of clients.

    Returns:
        Tuple of arrays: `(x_train, y_train), (x_test, y_test)`.
    """

    # Download and partition dataset
    data = load_datafile(path)
    logger.info("[LOAD DATA]Available client data: %d", len(data))
    client_id = int(client_id)
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
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def load_all_data(path, *args, **kwargs):
    """Load complete dataset for all patients/clients.

    Args:
        path (str): Path to the data file.
        segment_len (int): Length of data segments.

    Returns:
        Tuple of arrays containing concatenated data from all clients:
        `(x_train, y_train), (x_val, y_val), (x_test, y_test)`.
    """
    
    data = load_datafile(path)
    
    # Initialize empty lists to store data from all clients
    x_train, y_train = [], []
    x_val, y_val = [], []
    x_test, y_test = [], []
    
    # Iterate through all clients and collect their data
    for client_data in data:
        x_train, y_train = client_data['train']['input_series'], client_data['train']['output_series']
        x_val, y_val = client_data['validation']['input_series'], client_data['validation']['output_series']
        x_test, y_test = client_data['test']['input_series'], client_data['test']['output_series']
        
        x_train.extend(x_train)
        y_train.extend(y_train)
        x_val.extend(x_val)
        y_val.extend(y_val)
        x_test.extend(x_test)
        y_test.extend(y_test)
    
    # Convert lists to numpy arrays
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
