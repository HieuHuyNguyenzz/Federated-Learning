from FL import FedAvg
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
import numpy as np
from typing import List, Tuple, Dict, Union, Optional
from utils import load_config

config = load_config()

class FedAdp(FedAvg):
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
        alpha: int = 5,
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
            algorithm="FedAdp",
            current_parameters=current_parameters,
            evaluate_fn=evaluate_fn,
            test_dataset=test_dataset,
        )
        self.alpha = alpha
        self.current_angles = [None] * num_clients

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results."""
        weights_results = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        num_examples = [fit_res.num_examples for _, fit_res in results]
        ids = [int(fit_res.metrics["id"]) for _, fit_res in results]

        local_updates = np.array(weights_results, dtype=object) - np.array(parameters_to_ndarrays(self.current_parameters), dtype=object)

        local_gradients = -local_updates/self.learning_rate
        self.learning_rate *= self.decay_rate

        global_gradient = np.sum(np.array(num_examples).reshape(len(num_examples), 1) * local_gradients, axis=0) / sum(num_examples)

        local_grad_vectors = [np.concatenate([arr for arr in local_gradient], axis = None)
                              for local_gradient in local_gradients]

        global_grad_vector = np.concatenate([arr for arr in global_gradient], axis = None)

        instant_angles = np.arccos([np.dot(local_grad_vector, global_grad_vector) / (np.linalg.norm(local_grad_vector) * np.linalg.norm(global_grad_vector))
                          for local_grad_vector in local_grad_vectors])
        
        if server_round == 1:
            smoothed_angles = instant_angles
        else:
            pre_angles = [self.current_angles[i] for i in ids]
            smoothed_angles = [(server_round-1)/server_round * x + 1/server_round * y if x is not None else y
                               for x, y in zip(pre_angles, instant_angles)]
  

        for id, i in zip(ids, range(len(ids))):
            self.current_angles[id] = smoothed_angles[i]

        maps = self.alpha*(1-np.exp(-np.exp(-self.alpha*(np.array(smoothed_angles)-1))))

        weights = num_examples * np.exp(maps) / sum(num_examples * np.exp(maps))

        parameters_aggregated = np.sum(weights.reshape(len(weights), 1) * np.array(weights_results, dtype=object), axis=0)

        self.current_parameters = ndarrays_to_parameters(parameters_aggregated)
        metrics_aggregated = {}

        losses = [fit_res.num_examples * fit_res.metrics["loss"] for _, fit_res in results]
        corrects = [round(fit_res.num_examples * fit_res.metrics["accuracy"]) for _, fit_res in results]
        loss = sum(losses) / sum(num_examples)
        accuracy = sum(corrects) / sum(num_examples)
        print(f"train_loss: {loss} - train_acc: {accuracy}")

        self.result["round"].append(server_round)
        self.result["train_loss"].append(loss)
        self.result["train_accuracy"].append(accuracy)

        return self.current_parameters, metrics_aggregated
    