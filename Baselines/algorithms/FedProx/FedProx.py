from FL import FedAvg
from flwr.common import (
    FitIns,
    Parameters,
    )
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from typing import List, Tuple, Optional
from utils import load_config

config = load_config()

class FedProx(FedAvg):
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
        proximal_mu: float = 1.0,
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
            algorithm="FedProx",
            current_parameters=current_parameters,
            evaluate_fn=evaluate_fn,
            test_dataset=test_dataset,
        )
        self.proximal_mu = proximal_mu

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        config = {
            "learning_rate": self.learning_rate,
            "proximal_mu": self.proximal_mu
        }
        self.learning_rate *= self.decay_rate
        fit_ins = FitIns(parameters, config)
        return [(client, fit_ins) for client in clients]