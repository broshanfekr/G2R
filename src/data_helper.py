import numpy as np
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Coauthor, Amazon
import torch_geometric.transforms as T
import matplotlib.pyplot as plt
from torch_geometric.data import Data

import utils


def load_dataset(path, dataset_name, split_type):
    if dataset_name == "Cora":
        dataset = Planetoid(path, dataset_name, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.num_classes = dataset.num_classes
        if split_type == "random":
            data = utils.random_planetoid_splits(data, dataset.num_classes, lcc_mask=None)

    elif dataset_name == "CiteSeer":
        dataset = Planetoid(path, dataset_name, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.num_classes = dataset.num_classes
        if split_type == "random":
            data = utils.random_planetoid_splits(data, dataset.num_classes, lcc_mask=None)

    elif dataset_name == "PubMed":
        dataset = Planetoid(path, dataset_name, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.num_classes = dataset.num_classes
        if split_type == "random":
            data = utils.random_planetoid_splits(data, dataset.num_classes, lcc_mask=None)

    elif dataset_name == "CoraFull":
        dataset = CitationFull(path, "cora")
        data = dataset[0]
        data.num_classes = dataset.num_classes
        data = utils.random_coauthor_amazon_splits(data, dataset.num_classes, lcc_mask=None)

    elif dataset_name == "Photo":
        dataset = Amazon(path, dataset_name, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.num_classes = dataset.num_classes
        data = utils.random_coauthor_amazon_splits(data, dataset.num_classes, lcc_mask=None)

    elif dataset_name == "Computers":
        dataset = Amazon(path, dataset_name, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.num_classes = dataset.num_classes
        data = utils.random_coauthor_amazon_splits(data, dataset.num_classes, lcc_mask=None)

    elif dataset_name == "CS":
        dataset = Coauthor(path, dataset_name, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.num_classes = dataset.num_classes
        data = utils.random_coauthor_amazon_splits(data, dataset.num_classes, lcc_mask=None)

    elif dataset_name == "Physics":
        dataset = Coauthor(path, dataset_name, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.num_classes = dataset.num_classes
        data = utils.random_coauthor_amazon_splits(data, dataset.num_classes, lcc_mask=None)
    elif dataset_name == "Synthetic":
        pass
    
    else:
        print("Input dataset name!!")
        raise NotImplementedError
    
    return data


def generate_subspace_syntetich_data(ni, d, sigma, tetha):
    """
    Generate random data from 3 subspaces
    input: Ni = number of samples on each subspace
           d  = intrinsic dimension of each subspace
           sigma= level of noise
           tetha= angle between subspaces
    output: X: generated data
            labels: label of generated data
    """
    # generate base for each subspace
    base_subspaces = [np.concatenate([np.eye(d), np.zeros([d, d])])]
    base_subspaces.append(np.concatenate([np.cos(tetha) * np.eye(d), np.sin(tetha)*np.eye(d)]))
    base_subspaces.append(np.concatenate([np.cos(tetha) * np.eye(d), -np.sin(tetha)*np.eye(d)]))

    labels = []
    X = []
    for i in range(3):
        temp = np.random.randn(d, ni)
        X.append(base_subspaces[i] @ temp)  # + (-1)^(kk)*(kk-1);)
        labels.extend(ni*[i])
    X = np.concatenate(X, axis=1)
    labels = np.asarray(labels)

    noise = np.random.randn(*X.shape)
    noise = noise / np.sqrt(np.sum(noise ** 2, axis=0)).reshape(1, -1)

    X = X + sigma*noise
    X = X.T

    return X, labels


if __name__ == "__main__":
    X, labels = generate_subspace_syntetich_data(100, 1, 0.1, 2*np.pi/3)




    colors = ["red", "blue", 'green']
    for i, c in enumerate(colors):
        plt.plot(X[labels == i, 0], X[labels == i, 1], '.', alpha=0.5, c=c)
    plt.axis('equal')
    plt.grid()
    plt.show()
