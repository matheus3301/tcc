"""flwr-clusterization: A Flower / TensorFlow app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from flwr_fedavg.task import load_model


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Get parameters to initialize global model
    parameters = ndarrays_to_parameters(load_model().get_weights())

    # Define strategy
    strategy = strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

def fit_config(server_round: int):
    return {
        "current_round": server_round,
    }

def evaluate_config(server_round: int): 
    return {
        "current_round": server_round,
    }

# Create ServerApp
app = ServerApp(server_fn=server_fn)
