import torch.nn as nn
import torch.nn.functional as F


def l1norm(X, dim, eps=1e-8):
    return F.normalize(X, 1, dim=dim, eps=eps)


def l2norm(X, dim=-1, eps=1e-8):
    return F.normalize(X, 2, dim=dim, eps=eps)


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    return F.cosine_similarity(x1, x2, dim=dim, eps=eps)


def init_weight(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Embedding):
        nn.init.uniform_(module.weight, -0.1, 0.1)
    elif isinstance(module, nn.BatchNorm1d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
