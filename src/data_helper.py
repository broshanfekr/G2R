from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Coauthor, Amazon
import torch_geometric.transforms as T

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
    else:
        print("Input dataset name!!")
        raise NotImplementedError
    
    return data