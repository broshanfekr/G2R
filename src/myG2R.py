import numpy as np
from torch_geometric.utils import to_dense_adj

import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import argparse

from eval import label_classification
from mymodel import GCNConv, Encoder

import utils
import data_helper
from loss import GeometricMaximalCodingRateReduction as MaximalCodingRateReduction


def my_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_hidden', type=int, default=1024)
    parser.add_argument('--num_out', type=int, default=512)
    parser.add_argument('--gam1', type=float, default=0.5)
    parser.add_argument('--gam2', type=float, default=0.5)
    parser.add_argument('--eps', type=float, default=0.05)
    parser.add_argument('--activation', type=str, default="relu")
    parser.add_argument('--has_bias', type=bool, default=True)
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
    return args


def train(model, data, A_hat, MaximalCodingRateReduction):
    x = data.x
    y = data.y
    num_classes = data.num_classes

    model.train()
    
    optimizer.zero_grad()
    z = model(x, A_hat)
    loss = MaximalCodingRateReduction(z, A_hat)
    loss.backward()
    optimizer.step()

    return loss.item(), z


def test(model, data, A_hat, train_mask, val_mask, test_mask):
    x = data.x
    y = data.y
    num_classes = data.num_classes

    model.eval()

    z = model(x, A_hat)

    train_res = label_classification(z, y, train_mask=train_mask, test_mask=train_mask)
    val_res = label_classification(z, y, train_mask=train_mask, test_mask=val_mask)
    test_res = label_classification(z, y, train_mask=train_mask, test_mask=test_mask)

    return train_res, val_res, test_res, z


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
    args = my_arg_parser()
    print("Args:{}".format(args))

    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[args.activation]
    base_model = GCNConv

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(3)

    path = os.path.join(args.input_dir)
    data = data_helper.load_dataset(path=path, dataset_name=args.dataset, split_type=args.split)
    
    print("Dataset:", args.dataset)
    print("Number of Nodes:", data.x.shape[0])
    print("Number of Nodes Features:", data.x.shape[1])
    print("Number of Edges:", data.edge_index.shape[1])

    node_num = data.x.shape[0]
    num_features = data.x.shape[1]

    device = torch.device(args.cuda_idx if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    A_hat = to_dense_adj(data.edge_index)[0].cpu()
    A_hat = A_hat + torch.eye(A_hat.shape[0])
    A_hat = utils.normalize_adj(A_hat)
    A_hat = torch.from_numpy(A_hat.todense()).to(device)

    model = Encoder(in_channels=num_features,out_channels=args.num_out, hidden_channels=args.num_hidden, 
                      activation=activation,base_model=base_model, dropout=args.dropout, k=args.num_layers).to(device)
    
    coding_rate_loss = MaximalCodingRateReduction(device=device, num_node_batch=args.num_node_batch, 
                                                  gam1=args.gam1, gam2=args.gam2, eps=args.eps).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)

    for epoch in range(1, args.num_epochs + 1):
        loss, z = train(model, data, A_hat, coding_rate_loss)
        train_res, val_res, test_res, z = test(model, data, A_hat, 
                                               train_mask=data.train_mask.cpu().numpy(),
                                               val_mask=data.val_mask.cpu().numpy(),
                                               test_mask=data.test_mask.cpu().numpy())
        print("Epoch: {:03d}, train_acc: {:.4f}, val_acc: {:.4f}, test_acc:{:.4f}".format(epoch, train_res["acc"], val_res["acc"], test_res["acc"] ))

    x = z.detach().cpu().numpy()
    utils.build_tsne_representation_fig(x, target=data.y.cpu().numpy(), path="susegar.png")
    