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

class FedImp(FedAvg):
    def __init__(
        self,
        num_rounds: int,
        num_clients: int,
        entropies: List[float],
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.0,
        min_fit_clients: int = 0,
        min_evaluate_clients: int = 0,
        min_available_clients: int = 0,
        learning_rate: float = 0.1,
        decay_rate: float = 0.995,
        tempurature: float = 0.7,
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
            algorithm="FedImp",
            current_parameters=current_parameters,
            evaluate_fn=evaluate_fn,
            test_dataset=test_dataset,
        )
        self.tempurature = tempurature
        self.entropies = entropies

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        weights_results = [(parameters_to_ndarrays(fit_res.parameters),
                            fit_res.num_examples * np.exp(self.entropies[int(fit_res.metrics["id"])]/self.temperature))
                            for _, fit_res in results]
        print([fit_res.metrics["id"] for _, fit_res in results])
        self.current_parameters = ndarrays_to_parameters(aggregate(weights_results))
        metrics_aggregated = {}

        losses = [fit_res.num_examples * fit_res.metrics["loss"] for _, fit_res in results]
        corrects = [round(fit_res.num_examples * fit_res.metrics["accuracy"]) for _, fit_res in results]
        examples = [fit_res.num_examples for _, fit_res in results]
        loss = sum(losses) / sum(examples)
        accuracy = sum(corrects) / sum(examples)
        print(f"train_loss: {loss} - train_acc: {accuracy}")

        self.result["round"].append(server_round)
        self.result["train_loss"].append(loss)
        self.result["train_accuracy"].append(accuracy)

        return self.current_parameters, metrics_aggregated
    