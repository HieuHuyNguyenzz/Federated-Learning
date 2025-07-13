import numpy as np
from collections import defaultdict

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
    labels = np.array(trainset.targets)
    num_classes = len(np.unique(labels))
    data_indices = [np.where(labels == i)[0] for i in range(num_classes)]

    client_dict = defaultdict(list)

    for class_id in range(num_classes):
        class_indices = data_indices[class_id]
        np.random.shuffle(class_indices)

        proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]

        split_indices = np.split(class_indices, proportions)
        for client_id, client_indices in enumerate(split_indices):
            client_dict[client_id].extend(client_indices.tolist())

    return client_dict
