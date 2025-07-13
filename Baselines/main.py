import torch
import flwr as fl
import sys
from algorithms.FedAvg import FedAvgSimulation
from utils import load_config

def main(args):
    config = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} using PyTorch {torch.__version__} and Flower {fl.__version__}")

    algo = config["flwr"]["algo"]
    if algo == "FedAvg":
        simulation = FedAvgSimulation()
    elif algo == "FedProx":
        raise NotImplementedError("FedProx simulation is not yet implemented.")
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")
    
    simulation.run()
    print(f"{algo} simulation completed successfully.")

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
