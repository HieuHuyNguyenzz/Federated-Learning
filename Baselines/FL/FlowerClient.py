
import torch
import flwr as fl
from utils import test, train, set_parameters, get_parameters

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader=None):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = train(self.net, self.trainloader, learning_rate=config["learning_rate"])
        return get_parameters(self.net), len(self.trainloader.sampler), {"loss": loss, "accuracy": accuracy, "id": self.cid}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return loss, len(self.valloader.sampler), {"accuracy": accuracy}

