import torch
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
import numpy as np
import flwr as fl
from flwr.common import ndarrays_to_parameters


from models import 
import utils
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")