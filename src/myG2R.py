import numpy as np
from torch_geometric.utils import to_dense_adj
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Coauthor, Amazon
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
import torch
import argparse
import os

from eval import label_classification
from mymodel import GCNConv, Encoder

from utils import random_coauthor_amazon_splits, random_planetoid_splits, normalize_adj_row, normalize_adj, build_tsne_representation_fig

from loss import GeometricMaximalCodingRateReduction as MaximalCodingRateReduction


def train(model, data, A, A_hat, MaximalCodingRateReduction):
    x = data.x
    y = data.y
    num_classes = data.num_classes

    model.train()
    optimizer.zero_grad()
    z = model(x, A_hat)

    loss = MaximalCodingRateReduction(z, A)

    loss.backward()
    optimizer.step()

    return loss.item(), z


def test(model, data, A_hat, train_mask=None, test_mask=None):
    x = data.x
    y = data.y
    num_classes = data.num_classes

    model.eval()

    z = model(x, A_hat)

    res = label_classification(z, y, train_mask=train_mask, test_mask=test_mask)
    return res, z


def build_graph(y):
    y = y.numpy()
    num_nodes = y.shape[0]
    A = np.zeros([num_nodes, num_nodes])
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if y[i] == y[j]:
                A[i, j] = 1
    
    A = A + A.T
    return A



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_hidden', type=int, default=1024)
    parser.add_argument('--num_out', type=int, default=512)
    parser.add_argument('--gam1', type=float, default=0.5)
    parser.add_argument('--gam2', type=float, default=0.5)
    parser.add_argument('--eps', type=float, default=0.05)
    parser.add_argument('--activation', type=str, default="relu")
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_node_batch', type=int, default=768)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--input_dir', type=str, default="../data")
    parser.add_argument('--seed', type=int, default=21415)
    parser.add_argument('--round', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default="G2R")
    parser.add_argument('--split', type=str, default="fixed")
    parser.add_argument('--cuda_idx', type=str, default="cuda:0")
    args = parser.parse_args()

    print("Args:{}".format(args))

    seed = args.seed
    learning_rate = args.learning_rate
    num_hidden = args.num_hidden
    num_out = args.num_out
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[args.activation]
    base_model = GCNConv
    num_epochs = args.num_epochs
    weight_decay = args.weight_decay

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(3)


    path = os.path.join(args.input_dir)

    if args.dataset == "Cora":
        dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.num_classes = dataset.num_classes
        if args.split == "random":
            data = random_planetoid_splits(data, dataset.num_classes, lcc_mask=None)
    
    elif args.dataset == "CiteSeer":
        dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
        # dataset = Planetoid(path, args.dataset)
        data = dataset[0]
        data.num_classes = dataset.num_classes
        if args.split == "random":
            data = random_planetoid_splits(data, dataset.num_classes, lcc_mask=None)
    
    elif args.dataset == "PubMed":
        dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
        # dataset = Planetoid(path, args.dataset)
        data = dataset[0]
        data.num_classes = dataset.num_classes
        if args.split == "random":
            data = random_planetoid_splits(data, dataset.num_classes, lcc_mask=None)
    
    elif args.dataset == "CoraFull":
        dataset = CitationFull(path, "cora")
        data = dataset[0]
        data.num_classes = dataset.num_classes
        data = random_coauthor_amazon_splits(data, dataset.num_classes, lcc_mask=None)
    
    elif args.dataset == "Photo":
        dataset = Amazon(path, args.dataset, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.num_classes = dataset.num_classes
        data = random_coauthor_amazon_splits(data, dataset.num_classes, lcc_mask=None)
    
    elif args.dataset == "Computers":
        dataset = Amazon(path, args.dataset, transform=T.NormalizeFeatures())
        # dataset = Amazon(path, args.dataset)
        data = dataset[0]
        data.num_classes = dataset.num_classes
        data = random_coauthor_amazon_splits(data, dataset.num_classes, lcc_mask=None)
    
    elif args.dataset == "CS":
        dataset = Coauthor(path, args.dataset, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.num_classes = dataset.num_classes
        data = random_coauthor_amazon_splits(data, dataset.num_classes, lcc_mask=None)
    
    elif args.dataset == "Physics":
        dataset = Coauthor(path, args.dataset, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.num_classes = dataset.num_classes
        data = random_coauthor_amazon_splits(data, dataset.num_classes, lcc_mask=None)
    else:
        print("Input dataset name!!")
        raise NotImplementedError
    
    print("Dataset:", args.dataset)
    print("Number of Nodes:", data.x.shape[0])
    print("Number of Nodes Features:", data.x.shape[1])
    print("Number of Edges:", data.edge_index.shape[1])


    node_num = data.x.shape[0]
    num_features = data.x.shape[1]

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(args.cuda_idx if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    A = to_dense_adj(data.edge_index)[0].cpu()
    A = normalize_adj_row(A)
    A = torch.from_numpy(A.todense())

    A_hat = to_dense_adj(data.edge_index)[0].cpu()
    A_hat = A_hat + torch.eye(A_hat.shape[0])
    A_hat = normalize_adj(A_hat)
    A_hat = torch.from_numpy(A_hat.todense()).to(device)


    model = Encoder(in_channels=num_features,out_channels=num_out, hidden_channels=num_hidden, 
                      activation=activation,base_model=base_model, k=args.num_layers).to(device)
    
    # model = Model(encoder=encoder).to(device)
    coding_rate_loss = MaximalCodingRateReduction(device=device, num_node_batch=args.num_node_batch, 
                                                  gam1=args.gam1, gam2=args.gam2, eps=args.eps).to(device)
    # coding_rate_loss = MaximalCodingRateReduction(gam1=args.gam1, gam2=args.gam2, eps=args.eps).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(coding_rate_loss.parameters()), lr=learning_rate, weight_decay=weight_decay)


    for epoch in range(1, num_epochs + 1):
        loss, z = train(model, data, A, A_hat, coding_rate_loss)
        train_res, z = test(model, data, A_hat, train_mask=data.train_mask.cpu().numpy(), test_mask=data.train_mask.cpu().numpy())
        val_res, z = test(model, data, A_hat, train_mask=data.train_mask.cpu().numpy(), test_mask=data.val_mask.cpu().numpy())
        test_res, z = test(model, data, A_hat, train_mask=data.train_mask.cpu().numpy(), test_mask=data.test_mask.cpu().numpy())
        print("Epoch: {:03d}, train_acc: {:.4f}, val_acc: {:.4f}, test_acc:{:.4f}".format(epoch, train_res["acc"], val_res["acc"], test_res["acc"] ))

    x = z.detach().cpu().numpy()
    build_tsne_representation_fig(x, target=data.y.cpu().numpy(), path="susegar.png")
    