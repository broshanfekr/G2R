import numpy as np
import torch
import torch.nn.functional as F

from itertools import combinations


def one_hot(labels_int, n_classes):
    """Turn labels into one hot vector of K classes. """
    labels_onehot = torch.zeros(size=(len(labels_int), n_classes)).float()
    for i, y in enumerate(labels_int):
        labels_onehot[i, y] = 1.
    return labels_onehot


def label_to_membership(targets, num_classes=None):
    """Generate a true membership matrix, and assign value to current Pi.

    Parameters:
        targets (np.ndarray): matrix with one hot labels

    Return:
        Pi: membership matirx, shape (num_classes, num_samples, num_samples)

    """
    targets = one_hot(targets, num_classes)
    num_samples, num_classes = targets.shape
    Pi = np.zeros(shape=(num_classes, num_samples, num_samples))
    for j in range(len(targets)):
        k = np.argmax(targets[j])
        Pi[k, j, j] = 1.
    return Pi


class OrigMaximalCodingRateReduction(torch.nn.Module):
    def __init__(self, device, num_node_batch, gam1=1.0, gam2=1.0, eps=0.01):
        super(OrigMaximalCodingRateReduction, self).__init__()
        self.gam1 = gam1
        self.gam2 = gam2
        self.eps = eps
        self.device = device
        self.num_node_batch = num_node_batch

    def compute_discrimn_loss_empirical(self, W):
        """Empirical Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p).to(self.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + self.gam1 * scalar * W.matmul(W.T))
        return logdet / 2.

    def compute_compress_loss_empirical(self, W, Pi):
        """Empirical Compressive Loss."""
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p).to(self.device)
        compress_loss = 0.
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
            compress_loss += log_det * trPi / m
        return compress_loss / 2.

    def compute_discrimn_loss_theoretical(self, W):
        """Theoretical Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p).to(self.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.

    def compute_compress_loss_theoretical(self, W, Pi):
        """Theoretical Compressive Loss."""
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p).to(self.device)
        compress_loss = 0.
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
            compress_loss += trPi / (2 * m) * log_det
        return compress_loss

    def forward(self, X, Y, num_classes=None):
        if num_classes is None:
            num_classes = Y.max() + 1
        W = X.T
        Pi = label_to_membership(Y.cpu().numpy(), num_classes)
        Pi = torch.tensor(Pi, dtype=torch.float32).to(self.device)

        discrimn_loss_empi = self.compute_discrimn_loss_empirical(W)
        compress_loss_empi = self.compute_compress_loss_empirical(W, Pi)
        # discrimn_loss_theo = self.compute_discrimn_loss_theoretical(W)
        # compress_loss_theo = self.compute_compress_loss_theoretical(W, Pi)
 
        total_loss_empi = self.gam2 * -discrimn_loss_empi + compress_loss_empi
        return total_loss_empi


class GeometricMaximalCodingRateReduction(torch.nn.Module):
    # This function is based on https://github.com/ryanchankh/mcr2/blob/master/loss.py
    def __init__(self, device, num_node_batch, gam1=1.0, gam2=1.0, eps=0.01):
        super(GeometricMaximalCodingRateReduction, self).__init__()
        self.gam1 = gam1
        self.gam2 = gam2
        self.eps = eps
        self.device = device
        self.num_node_batch = num_node_batch

    def compute_discrimn_loss_empirical(self, W):
        """Empirical Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p).to(self.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + self.gam1 * scalar * W.matmul(W.T))
        return logdet / 2.

    def compute_compress_loss_empirical_all(self, W, Pi):
        """Empirical Compressive Loss."""
        p, m = W.shape
        k, _ = Pi.shape
        sum_trPi = torch.sum(Pi)
        d_bar = sum_trPi/m
        # d_bar = sum_trPi/k

        I = torch.eye(p).to(self.device)
        compress_loss = 0.
        for j in range(k):
            trPi = torch.sum(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            a = W.T * Pi[j].view(-1, 1)
            a = a.T
            log_det = torch.logdet(I + scalar * a.matmul(W.T))
            compress_loss += log_det * trPi / m

        compress_loss = compress_loss / (2*d_bar)
        return compress_loss

    def forward(self, X, A):
        i = np.random.randint(A.shape[0], size=self.num_node_batch)
        A = A[i, ::]
        W = X.T
        Pi = A

        discrimn_loss_empi = self.compute_discrimn_loss_empirical(W)
        compress_loss_empi = self.compute_compress_loss_empirical_all(W, Pi)
        total_loss_empi = - self.gam2 * discrimn_loss_empi + compress_loss_empi
        return total_loss_empi
    

class GraphLearningLoss(torch.nn.Module):

    def __init__(self, device, num_nodes, losslr1=1.0, losslr2=1.0):
        super(GraphLearningLoss, self).__init__()
        self.losslr1 = losslr1
        self.losslr2 = losslr2
        self.num_nodes = num_nodes
        self.device = device

    def forward(self, S, x):
        L = torch.eye(self.num_nodes).to(self.device) - S
        t1 = torch.mm(x.transpose(1, 0), L)
        t1 = torch.trace(torch.mm(t1, x))

        t2 = torch.trace(torch.mm(S.transpose(1, 0), S))

        loss = (t1 * self.losslr1) + (t2 * self.losslr2)

        return loss
    

class MyGeometricMaximalCodingRateReduction(torch.nn.Module):
    # This function is based on https://github.com/ryanchankh/mcr2/blob/master/loss.py
    def __init__(self, device, num_node_batch, gam1=1.0, gam2=1.0, eps=0.01):
        super(MyGeometricMaximalCodingRateReduction, self).__init__()
        self.gam1 = gam1
        self.gam2 = gam2
        self.eps = eps
        self.device = device
        self.num_node_batch = num_node_batch

    def compute_discrimn_loss_empirical(self, W):
        """Empirical Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p).to(self.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + self.gam1 * scalar * W.matmul(W.T))
        return logdet / 2.

    def compute_compress_loss_empirical_all(self, W, Pi):
        """Empirical Compressive Loss."""
        p, m = W.shape
        k, _ = Pi.shape
        sum_trPi = torch.sum(Pi)
        d_bar = sum_trPi/m
        # d_bar = sum_trPi/k

        I = torch.eye(p).to(self.device)
        compress_loss = 0.
        for j in range(k):
            trPi = torch.sum(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            a = W.T * Pi[j].view(-1, 1)
            a = a.T
            log_det = torch.logdet(I + scalar * a.matmul(W.T))
            compress_loss += log_det * trPi / m

        compress_loss = compress_loss / (2*d_bar)
        return compress_loss

    def forward(self, X, A):
        i = np.random.randint(A.shape[0], size=self.num_node_batch)
        A = A[i, ::]
        W = X.T
        Pi = A

        discrimn_loss_empi = self.compute_discrimn_loss_empirical(W)
        compress_loss_empi = self.compute_compress_loss_empirical_all(W, Pi)
        total_loss_empi = - self.gam2 * discrimn_loss_empi + compress_loss_empi
        return total_loss_empi
