import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

USE_GPU = True


def to_gpu(x, USE_GPU=USE_GPU):
    if USE_GPU:
        return x.cuda()
    else:
        return x
