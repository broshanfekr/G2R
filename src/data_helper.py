import numpy as np
import torch
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Coauthor, Amazon
import torch_geometric.transforms as T
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import networkx as nx
from sklearn.model_selection import train_test_split
import copy

import utils


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)


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
        in_prob = 0.7
        out_prob = 0.01
        ni = 30
        cluster_num = 2
        angle = np.pi/4  #2*np.pi/3
        intrinsic_dim = 1
        noise_sigma = 0.1

        G, X, labels = generate_subspace_graph_syntetich_data(ni, intrinsic_dim, noise_sigma, angle, in_prob, out_prob, cluster_num)

        data_index = np.arange(labels.shape[0])
        train_index, test_index = train_test_split(data_index, test_size=0.3, random_state=42)
        val_index, test_index = train_test_split(test_index, test_size=0.33, random_state=42)
        train_mask = sample_mask(train_index, labels.shape[0])
        val_mask = sample_mask(val_index, labels.shape[0])
        test_mask = sample_mask(test_index, labels.shape[0])
        pos = copy.deepcopy(X)

        edge_index = torch.tensor([[*e] for e in G.edges], dtype=torch.long).T
        X = np.float32(X)
        X = torch.from_numpy(X)
        labels = torch.from_numpy(labels)
        pos = torch.from_numpy(pos)

        data = Data(x=X, y=labels, edge_index=edge_index, pos=pos)
        data.num_classes = 3
        data.train_mask = torch.from_numpy(train_mask)
        data.val_mask = torch.from_numpy(val_mask)
        data.test_mask = torch.from_numpy(test_mask)

    else:
        print("Input dataset name!!")
        raise NotImplementedError
    
    return data


def generate_subspace_graph_syntetich_data(ni, d, sigma, tetha, in_prob, out_prob, cluster_num):
    """
    Generate random data from 3 subspaces
    input: Ni = number of samples on each subspace
           d  = intrinsic dimension of each subspace
           sigma = level of noise
           tetha = angle between subspaces
           in_prob =
           out_prob =
    output: X: generated data
            labels: label of generated data
    """

    sizes = cluster_num * [ni]
    temp_p = [in_prob] + (cluster_num-1) * [out_prob]
    probs = [temp_p]
    for i in range(1, cluster_num):
        temp_p = temp_p[:]
        temp_p[i] = in_prob
        temp_p[i-1] = out_prob
        probs.append(temp_p)

    # probs = [[in_prob, out_prob, out_prob], [out_prob, in_prob, out_prob], [out_prob, out_prob, in_prob]]

    G = nx.stochastic_block_model(sizes, probs, seed=0)

    # generate base for each subspace
    base_subspaces = [np.concatenate([np.eye(d), np.zeros([d, d])])]
    base_subspaces.append(np.concatenate([np.cos(tetha) * np.eye(d), np.sin(tetha)*np.eye(d)]))
    if cluster_num == 3:
        base_subspaces.append(np.concatenate([np.cos(tetha) * np.eye(d), -np.sin(tetha)*np.eye(d)]))

    labels = []
    X = []
    for i in range(cluster_num):
        temp = np.random.randn(d, ni)
        X.append(base_subspaces[i] @ temp)  # + (-1)^(kk)*(kk-1);)
        labels.extend(ni*[i])
    X = np.concatenate(X, axis=1)
    labels = np.asarray(labels)

    noise = np.random.randn(*X.shape)
    noise = noise / np.sqrt(np.sum(noise ** 2, axis=0)).reshape(1, -1)

    X = X + sigma*noise
    X = X.T

    return G, X, labels


if __name__ == "__main__":
    in_prob = 0.3
    out_prob = 0.01
    ni = 30
    G, X, labels = generate_subspace_graph_syntetich_data(ni, 1, 0.1, 2*np.pi/3, in_prob, out_prob)

    data_index = np.arange(labels.shape[0])
    train_index, test_index = train_test_split(data_index, test_size=0.3, random_state=42)
    val_index, test_index = train_test_split(test_index, test_size=0.33, random_state=42)
    train_mask = sample_mask(train_index, labels.shape[0])
    val_mask = sample_mask(val_index, labels.shape[0])
    test_mask = sample_mask(test_index, labels.shape[0])
    # pos = nx.spectral_layout(G)
    pos = X

    edge_index = torch.tensor([[*e] for e in G.edges], dtype=torch.long).T
    X = torch.from_numpy(X)
    labels = torch.from_numpy(labels)
    pos = torch.from_numpy(pos)

    data = Data(x=X, y=labels, edge_index=edge_index, pos=pos)

    # pos = nx.spectral_layout(G)
    pos = X
    fig, (ax1, ax2) = plt.subplots(1, 2)

    nx.draw(G, pos=pos.numpy(),
            ax=ax1,
            with_labels=True,
            font_weight='bold',
            font_color='black',
            font_size=10,
            node_color=labels,
            edge_color='gray',
            linewidths=1,
            alpha=0.7)

    plt.show()
