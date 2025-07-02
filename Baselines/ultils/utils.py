import numpy as np
from collections import OrderedDict
from typing import List, Dict
import random
import math
import torch
import yaml

seed_value = 42
random.seed(seed_value)
torch.manual_seed(seed_value)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def renormalize(dist: torch.tensor, labels: List[int], label: int):
    idx = labels.index(label)
    dist[idx] = 0
    dist /= sum(dist)
    dist = torch.concat((dist[:idx], dist[idx+1:]))
    return dist

def compute_entropy(counts: Dict):
    entropy = 0.0
    counts = list(counts.values())
    counts = [0 if value is None else value for value in counts]
    for value in counts:
        entropy += -value/sum(counts) * math.log(value/sum(counts), len(counts)) if value != 0 else 0
    return entropy

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict)

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
