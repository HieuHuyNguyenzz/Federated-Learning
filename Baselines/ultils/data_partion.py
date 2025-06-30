import numpy as np
from collections import defaultdict

seed_value = 42
np.random.seed(seed_value)

def data_partition(dataset, num_clients, alpha = 100):
    """
    Partition the dataset into `num_clients` parts.
    
    Args:
        data (list): The dataset to be partitioned.
        num_clients (int): The number of clients to partition the data for.
        alpha (int, optional): A parameter for partitioning strategy. Defaults to 100.
        
    Returns:
        list: A list containing the partitioned data for each client.
    """
    # Assume dataset is a tuple (data, labels)
    data, labels = dataset
    labels = np.array(labels)
    num_classes = np.max(labels) + 1

    # Create a list for each client
    idxs = np.arange(len(labels))
    partitioned_data = [[] for _ in range(num_clients)]

    # For each class, partition its indices to clients using Dirichlet distribution
    for c in range(num_classes):
        idxs_c = idxs[labels == c]
        np.random.shuffle(idxs_c)
        proportions = np.random.dirichlet([alpha] * num_clients)
        # Scale proportions so that sum equals the number of indices
        proportions = (np.cumsum(proportions) * len(idxs_c)).astype(int)[:-1]
        split_idxs = np.split(idxs_c, proportions)
        for i, idx in enumerate(split_idxs):
            partitioned_data[i].extend(idx.tolist())

    # Shuffle each client's data
    for i in range(num_clients):
        np.random.shuffle(partitioned_data[i])

    return partitioned_data