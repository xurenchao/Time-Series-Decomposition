import torch
import torch.nn as nn
import numpy as np
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

USE_GPU = True


def to_gpu(x, USE_GPU=USE_GPU):
    if USE_GPU:
        return x.cuda()
    else:
        return x
def clip_grads(net, low=-10, high=10):
    """Gradient clipping to the range [low, high]."""
    parameters = [param for param in net.parameters()
                  if param.grad is not None]
    for p in parameters:
        p.grad.data.clamp_(low, high)

def picp(target, lower_bound, upper_bound):
    return np.mean((target>=lower_bound) * (target<=upper_bound))

def mpiw(target, lower_bound, upper_bound, norm=0):
    R = max(target)-min(target)
    v = np.mean(upper_bound-lower_bound)
    if norm:
        return v/R
    else:
        return v

def rpiw(target, lower_bound, upper_bound, norm=0):
    R = max(target)-min(target)
    v = (np.mean((upper_bound-lower_bound)**2))**.5
    if norm:
        return v/R
    else:
        return v

def cwc(target, lower_bound, upper_bound, alpha=0.05, yita=50):
    NMPIW = mpiw(target, lower_bound, upper_bound, 1)
    PICP = picp(target, lower_bound, upper_bound)
    gama = (PICP < (1-alpha))
    v = NMPIW * (1 + gama * math.exp(yita * (1-alpha-PICP)))
    return v


class MyLoss(nn.Module):
    def __init__(self, lmd=0.3, alpha=0.05, s=50):
        super(MyLoss, self).__init__()
        self.lmd = lmd
        self.alpha = alpha
        self.s = s

    def forward(self, pred, truth):
        """
        pred.shape = (B, D, H*2) or (D, H*2)
        truth.shape = (B, D, H) or (D, H)
        """
        target = torch.reshape(truth, (-1, truth.shape[-1]))
        bound = torch.reshape(pred, (-1, pred.shape[-1]))
        M, H = target.shape
        N = M * H
        lower_bound, upper_bound= bound[:, :H], bound[:, H:]

        K = ((target >= lower_bound) * (upper_bound >= target)).float()
        C = torch.sum(K)
        mpiw_capt = torch.sum((upper_bound - lower_bound) * K) / C
        K_soft = torch.sigmoid(self.s * (target - lower_bound)) *\
                 torch.sigmoid(self.s * (upper_bound - target))
        picp_soft = torch.sum(K_soft) / N
        gap = 1 - self.alpha - picp_soft
        if gap>0:
            return mpiw_capt + self.lmd * N * gap**2 / self.alpha / (1 - self.alpha)
        else:
            return mpiw_capt
