import torch
from torch.autograd import Variable
# import config


USE_GPU = True

if torch.cuda.is_available() and USE_GPU:
    def to_var(x, requires_grad=False, volatile=False, gpu=None):
        x = x.cuda(gpu)
        return Variable(x, requires_grad=requires_grad, volatile=volatile)
else:
    def to_var(x, requires_grad=False, volatile=False, gpu=None):
        return Variable(x, requires_grad=requires_grad, volatile=volatile)
