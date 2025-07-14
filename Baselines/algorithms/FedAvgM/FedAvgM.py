from FL import FedAvg
from FL import aggregate
from flwr.common import (
    FitIns,
    Parameters,
    NDArrays,
    FitRes,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
    )
from flwr.server.client_proxy import ClientProxy
from typing import List, Tuple, Dict, Union, Optional
from utils import load_config

config = load_config()

class FedAvgM(FedAvg):
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
        server_learning_rate: float = 1.0,
        server_momentum: float = 0.0,
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
            algorithm="FedAvgM",
            current_parameters=current_parameters,
            evaluate_fn=evaluate_fn,
            test_dataset=test_dataset,
        )
        self.server_momentum = server_momentum
        self.server_learning_rate = server_learning_rate
        self.server_opt: bool = (self.server_momentum != 0.0) or (
            self.server_learning_rate != 1.0
        )
        self.momentum_vector: Optional[NDArrays] = None

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        weights_results = [(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results]
        fedavg_result = aggregate(weights_results)

        if self.server_opt:
            initial_weights = parameters_to_ndarrays(self.current_parameters)
            pseudo_gradient: NDArrays = [
                x - y
                for x, y in zip(
                    parameters_to_ndarrays(self.current_parameters), fedavg_result
                )
            ]
            if self.server_momentum > 0.0:
                if server_round > 1:
                    assert (
                        self.momentum_vector
                    ), "Momentum should have been created on round 1."
                    self.momentum_vector = [
                        self.server_momentum * x + y
                        for x, y in zip(self.momentum_vector, pseudo_gradient)
                    ]
                else:
                    self.momentum_vector = pseudo_gradient

                # No nesterov for now
                pseudo_gradient = self.momentum_vector

            # SGD
            fedavg_result = [
                x - self.server_learning_rate * y
                for x, y in zip(initial_weights, pseudo_gradient)
            ]
            # Update current weights
            self.initial_parameters = ndarrays_to_parameters(fedavg_result)

        self.current_parameters = ndarrays_to_parameters(fedavg_result)
        metrics_aggregated = {}

        losses = [fit_res.num_examples * fit_res.metrics["loss"] for _, fit_res in results]
        corrects = [round(fit_res.num_examples * fit_res.metrics["accuracy"]) for _, fit_res in results]
        examples = [fit_res.num_examples for _, fit_res in results]
        loss = sum(losses) / sum(examples)
        accuracy = sum(corrects) / sum(examples)

        self.result["round"].append(server_round)
        self.result["train_loss"].append(loss)
        self.result["train_accuracy"].append(accuracy)
        print(f"train_loss: {loss} - train_acc: {accuracy}")
        return self.current_parameters, metrics_aggregated