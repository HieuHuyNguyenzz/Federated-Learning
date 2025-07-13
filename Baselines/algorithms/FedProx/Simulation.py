from FL import ClientManager
import flwr as fl
import torch
from torch.utils.data import Subset
import random
from flwr.common import ndarrays_to_parameters, Context
from utils import load_data, load_config, get_parameters, data_partition, to_dataloader
from models.get_model import get_model
from .FedProxClient import FedProxClient
from .FedProx import FedProx

class Simulation:
    def __init__(self, config_path: str = None, proximal_mu: float = 1.0):
        self.config = load_config(config_path) if config_path else load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.proximal_mu = proximal_mu
        self.trainloaders = []
        self.valloaders = []  # Placeholder in case val loaders are needed later
        self._set_seed(42)

    def _set_seed(self, seed_value: int):
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)

    def _prepare_data(self):
        trainset, _ = load_data(self.config['dataset']['name'])
        client_dict = data_partition(trainset, self.config['flwr']['num_clients'], self.config['dataset']['alpha'])

        for i in range(self.config['flwr']['num_clients']):
            client_subset = Subset(trainset, client_dict[i])
            self.trainloaders.append(to_dataloader(client_subset))

    def _create_client_fn(self):
        def client_fn(context: Context) -> FedProxClient:
            cid = int(context.node_config["partition-id"])
            model = get_model(
                self.config['model']['name'],
                self.config['model']['input_shape'],
                self.config['model']['num_classes']
            ).to(self.device)
            trainloader = self.trainloaders[cid]
            return FedProxClient(cid=cid, net=model, trainloader=trainloader).to_client()

        return client_fn

    def run(self):
        self._prepare_data()

        client_manager = ClientManager()
        model = get_model(
            self.config['model']['name'],
            self.config['model']['input_shape'],
            self.config['model']['num_classes']
        )
        initial_params = ndarrays_to_parameters(get_parameters(model))

        strategy = FedProx(
            num_rounds=self.config['flwr']['num_rounds'],
            num_clients=self.config['flwr']['num_clients'],
            fraction_fit=self.config['flwr']['fraction_fit'],
            proximal_mu=self.proximal_mu,
            current_parameters=initial_params,
            learning_rate=self.config['training']['learning_rate'],
            decay_rate=self.config['training']['decay_rate'],
        )

        fl.simulation.start_simulation(
            client_fn=self._create_client_fn(),
            client_manager=client_manager,
            num_clients=self.config['flwr']['num_clients'],
            config=fl.server.ServerConfig(num_rounds=self.config['flwr']['num_rounds']),
            strategy=strategy,
            client_resources=self.config['flwr']['client_resource']
        )

        print("Simulation completed.")

