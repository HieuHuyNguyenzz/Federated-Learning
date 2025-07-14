import torch
import flwr as fl
from utils import test, set_parameters, get_parameters
from copy import deepcopy
from train_FedDyn import train_feddyn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FedDynClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader=None):
        self.cid = cid
        self.net = net.to(DEVICE)
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    # fit.py

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        
        # Lấy mô hình toàn cục từ server (dưới dạng parameters)
        global_model = deepcopy(self.net)
        set_parameters(global_model, parameters)
        
        loss, accuracy = train_feddyn(
            self.net,
            self.trainloader,
            learning_rate=config["learning_rate"],
            alpha=config["alpha"],
            global_model=global_model
        )

        return get_parameters(self.net), len(self.trainloader.sampler), {
            "loss": loss,
            "accuracy": accuracy,
            "id": self.cid
        }


    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        if self.valloader is None:
            return float("nan"), 0, {"accuracy": float("nan")}
        loss, accuracy = test(self.net, self.valloader)
        return loss, len(self.valloader.sampler), {"accuracy": accuracy}