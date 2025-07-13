import torch
from utils import set_parameters, get_parameters
from FL import FlowerClient
from .train import train

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FedProxClient(FlowerClient):
    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = train(
            self.net, 
            self.trainloader, 
            learning_rate=config["learning_rate"],
            proximal_mu=config.get("proximal_mu", 0.0)
        )
        return get_parameters(self.net), len(self.trainloader.sampler), {
            "loss": loss, 
            "accuracy": accuracy, 
            "id": self.cid
        }