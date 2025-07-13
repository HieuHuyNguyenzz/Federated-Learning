import torch
import flwr as fl
import sys
from algorithms import FedAvg, FedProx, FedAdp
from utils import load_config

config_path="config.yaml"

def main(args):
    config = load_config(path=config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} using PyTorch {torch.__version__} and Flower {fl.__version__}")

    algo = config["flwr"]["algo"]
    if algo == "FedAvg":
        simulation = FedAvg.Simulation(config_path=config_path)
    elif algo == "FedProx":
        simulation = FedProx.Simulation(config_path=config_path, proximal_mu=2.0)
    elif algo == "FedAdp":
        simulation = FedAdp.Simulation(config_path=config_path)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")
    
    simulation.run()
    print(f"{algo} simulation completed successfully.")

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
