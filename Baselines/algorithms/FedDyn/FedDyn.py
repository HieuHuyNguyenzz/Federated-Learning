from FL import FedAvg
from FL import aggregate
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
import numpy as np
from functools import reduce
from typing import List, Tuple, Dict, Union, Optional
from utils import load_config

config = load_config()

class FedDyn(FedAvg):
    def __init__(
        self,
        num_rounds: int,
        num_clients: int,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.0,
        min_fit_clients: int = 0,
        min_evaluate_clients: int = 0,
        min_available_clients: int = 0,
        learning_rate: float = 0.1,
        decay_rate: float = 0.995,
        alpha: float = 0.01,
        current_parameters: Optional[Parameters] = None,
        evaluate_fn=None,
        test_dataset=None,
    ) -> None:
        super().__init__(
            num_rounds=num_rounds,
            num_clients=num_clients,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            learning_rate=learning_rate,
            decay_rate=decay_rate,
            algorithm="FedDyn",
            current_parameters=current_parameters,
            evaluate_fn=evaluate_fn,
            test_dataset=test_dataset,
        )
        self.alpha = alpha

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using FedDyn algorithm."""
        # Extract client updates
        client_updates = []
        num_examples_total = 0

        for _, fit_res in results:
            local_weights = parameters_to_ndarrays(fit_res.parameters)
            num_examples = fit_res.num_examples
            delta = [w - w0 for w, w0 in zip(local_weights, self.previous_parameters)]
            client_updates.append((delta, num_examples))
            num_examples_total += num_examples

        # Compute average update direction (FedDyn h_t update)
        avg_update = [
            sum(delta[i] * num_examples for delta, num_examples in client_updates) / num_examples_total
            for i in range(len(self.previous_parameters))
        ]

        # Update server model (FedDyn step)
        self.ht = [h - self.alpha * update for h, update in zip(self.ht, avg_update)]
        new_weights = [
            sum(local_weights[i] * num_examples for (_, fit_res) in results for i, local_weights in enumerate(parameters_to_ndarrays(fit_res.parameters))) / num_examples_total
            for i in range(len(self.previous_parameters))
        ]
        self.current_parameters = ndarrays_to_parameters([
            new_weights[i] - (1 / self.alpha) * self.ht[i] for i in range(len(new_weights))
        ])

        # Compute metrics
        losses = [fit_res.num_examples * fit_res.metrics["loss"] for _, fit_res in results]
        corrects = [round(fit_res.num_examples * fit_res.metrics["accuracy"]) for _, fit_res in results]
        examples = [fit_res.num_examples for _, fit_res in results]
        loss = sum(losses) / sum(examples)
        accuracy = sum(corrects) / sum(examples)

        self.result["round"].append(server_round)
        self.result["train_loss"].append(loss)
        self.result["train_accuracy"].append(accuracy)

        return self.current_parameters, {}
        