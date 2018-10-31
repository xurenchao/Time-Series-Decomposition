import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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