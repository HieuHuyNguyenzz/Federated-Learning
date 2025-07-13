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

def renormalize(dist: torch.Tensor, labels: List[int], label: int):
    idx = labels.index(label)
    dist[idx] = 0
    dist /= dist.sum()
    return torch.cat((dist[:idx], dist[idx+1:]))

def compute_entropy(counts: Dict):
    counts = [v or 0 for v in counts.values()]
    total = sum(counts)
    if total == 0:
        return 0.0
    return -sum(
        (v / total) * math.log(v / total, len(counts)) if v > 0 else 0
        for v in counts
    )

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict)

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
