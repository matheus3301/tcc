import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics
import numpy as np
from datetime import datetime
import pandas as pd

# Training configuration
NUM_ROUNDS = 50
MIN_CLIENTS = 2  # Minimum number of clients to start training
MIN_AVAILABLE_CLIENTS = 2  # Minimum number of available clients required

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregates metrics weighted by number of samples."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["rmse"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Aggregate and return custom metric (weighted average)
    return {"rmse": sum(accuracies) / sum(examples)}

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)

        # Log the aggregated metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_dict = {
            "round": server_round,
            "loss": loss_aggregated,
            "rmse": metrics_aggregated["rmse"] if metrics_aggregated else None
        }
        
        # Save metrics to CSV
        pd.DataFrame([metrics_dict]).to_csv(
            f'results/federated/metrics_{timestamp}_round_{server_round}.csv',
            index=False,
            mode='a',
            header=not pd.io.common.file_exists(f'results/federated/metrics_{timestamp}_round_{server_round}.csv')
        )

        return loss_aggregated

def main():
    # Create strategy
    strategy = SaveModelStrategy(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
        min_fit_clients=MIN_CLIENTS,  # Never sample less than MIN_CLIENTS clients for training
        min_evaluate_clients=MIN_CLIENTS,  # Never sample less than MIN_CLIENTS clients for evaluation
        min_available_clients=MIN_AVAILABLE_CLIENTS,  # Wait until MIN_AVAILABLE_CLIENTS are available
        evaluate_metrics_aggregation_fn=weighted_average,  # Custom aggregation function for metrics
    )

    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

if __name__ == "__main__":
    main() 