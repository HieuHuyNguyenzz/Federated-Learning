import numpy as np
import random
import torch
from torch.distributions import Dirichlet
from collections import Counter
from utils.utils import renormalize

seed_value = 42
np.random.seed(seed_value)

def data_partition(trainset, num_clients: int, alpha: float):
    """
    Chia trainset thành num_clients phần theo phân phối không đồng đều dựa trên alpha (Dirichlet).

    Args:
        trainset: Dataset (PyTorch Dataset có .targets)
        num_clients (int): Số lượng client
        alpha (float): Tham số của phân phối Dirichlet

    Returns:
        dict: client_id -> list các index mẫu dữ liệu thuộc về client đó
    """
    classes = trainset.classes
    client_size = int(len(trainset)/num_clients)
    label_size = int(len(trainset)/len(classes))
    data = list(map(lambda x: (trainset[x][1], x), range(len(trainset))))
    data.sort()
    data = list(map(lambda x: data[x][1], range(len(data))))
    data = [data[i*label_size:(i+1)*label_size] for i in range(len(classes))]

    ids = [[] for _ in range(num_clients)]
    label_dist = []
    labels = list(range(len(classes)))

    for i in range(num_clients):
        concentration = torch.ones(len(labels))*alpha
        dist = Dirichlet(concentration).sample()
        for _ in range(client_size):
            label = random.choices(labels, dist)[0]
            id = random.choices(data[label])[0]
            ids[i].append(id)
            data[label].remove(id)

            if len(data[label]) == 0:
                dist = renormalize(dist, labels, label)
                labels.remove(label)

        counter = Counter(list(map(lambda x: trainset[x][1], ids[i])))
        label_dist.append({classes[i]: counter.get(i) for i in range(len(classes))})

    return ids, label_dist
