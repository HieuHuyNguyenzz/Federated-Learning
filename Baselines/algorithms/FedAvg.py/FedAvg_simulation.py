from FL import ClientManager, FedAvg, FlowerClient
import flwr as fl
import torch
import random
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
import numpy as np
from flwr.common import ndarrays_to_parameters
from . import utils
seed_value = 42
random.seed(seed_value)
torch.manual_seed(seed_value)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def FedAvg_simulation(
    num_rounds: int,
    trainset,
    testset,
    num_clients: int = 10,
    fraction_fit: float = 0.1,
    model = None,
    client_resource = {"num_gpus": 1, "num_cpus": 0.1},
    learning_rate: float = 0.01,
    decay_rate: float = 1,
):
    """Run a federated learning simulation using FedAvg."""
    
    client_manager = ClientManager()
    current_parameters = ndarrays_to_parameters(utils.get_parameters(model))

    # Create a FedAvg strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        initial_parameters=current_parameters,
        learning_rate = learning_rate,

    )
    
    # Start the simulation
    fl.simulation.start_simulation(
        client_manager=client_manager,
        num_rounds=num_rounds,
        num_clients = num_clients,
        config = fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resource=client_resource,
    )

    print("Simulation completed.")