
import torch
import flwr as fl
from . import utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return utils.get_parameters(self.net)

    def fit(self, parameters, config):
        utils.set_parameters(self.net, parameters)
        loss, accuracy = utils.train(self.net, self.trainloader, learning_rate=config["learning_rate"])
        return utils.get_parameters(self.net), len(self.trainloader.sampler), {"loss": loss, "accuracy": accuracy, "id": self.cid}

    def evaluate(self, parameters, config):
        utils.set_parameters(self.net, parameters)
        loss, accuracy = utils.test(self.net, self.valloader)
        return loss, len(self.valloader.sampler), {"accuracy": accuracy}
