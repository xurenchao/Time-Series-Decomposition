import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
import torch.optim as optim
from tool import to_gpu
import numpy as np


class BinaryFunction(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.round(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables[0]
        grad_input = to_gpu(torch.ones_like(input.data))
        return grad_input

class GatedRNN(nn.Module):
    def __init__(self, seq_dim, hidden_size):
        super(GatedRNN, self).__init__()
        self.seq_dim = seq_dim
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(seq_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, seq_dim)
        self.gate = nn.Linear(seq_dim, 1)
        self.binary = BinaryFunction.apply

    def init_hidden_state(self):
        return to_gpu(torch.zeros(1, self.hidden_size))

    def init_gate_state(self):
        return to_gpu(torch.ones(1, 1))

    def get_deviation(self, y, y_pred):
        assert y.dim() == y_pred.dim()
        dev = torch.abs(y - y_pred)
        return dev

    def get_gate_state(self, y, y_pred):
        dev = self.get_deviation(y, y_pred)
        _u = F.sigmoid(self.gate(dev)).squeeze(0)
        u = self.binary(_u)
        return u

    def forward(self, x):
        hidden = self.init_hidden_state()
        u = self.init_gate_state()
        outputs = []
        xlist = x.split(1, dim=1)
        l = len(xlist)
        idx = 0
        for input_t in xlist:
            hidden = u * F.tanh(self.fc1(input_t.squeeze(1)) + self.fc2(hidden)) + (1 - u) * hidden
            output = self.fc3(hidden)
            outputs += [output]

            idx += 1
            if idx < l:
                u = self.get_gate_state(xlist[idx].squeeze(1), output)
            
        outputs = torch.stack(outputs, 1).squeeze(2)  # 不理解
        return outputs

    def forecast(self, x):
        x = x.unsqueeze(0)  # non-batch
        hidden = self.init_hidden_state()
        u = self.init_gate_state()
        outputs = []
        xlist = x.split(1, dim=1)
        l = len(xlist)
        idx = 0
        for input_t in xlist:
            hidden = u * F.tanh(self.fc1(input_t.squeeze(1)) + self.fc2(hidden)) + (1 - u) * hidden
            output = self.fc3(hidden)
            outputs += [output]

            idx += 1
            if idx < l:
                u = self.get_gate_state(xlist[idx].squeeze(1), output)

        outputs = torch.stack(outputs, 1).squeeze(2)  # 不理解
        return outputs[0]

    # def self_forecast(self, x, y, step):
    #     x = x.unsqueeze(0)  # non-batch
    #     y = y.unsqueeze(0)
    #     hidden = self.init_hidden_state()
    #     u = self.init_gate_state()
    #     outputs = []
    #     xlist = x.split(1, dim=1)
    #     ylist = y.split(1, dim=1)
    #     idx=0
    #     for input_t in xlist:
    #         output, hidden = self.rnn(input_t.squeeze(dim=1), hidden, u)
    #         u = self.get_gate_state(ylist[idx].squeeze(dim=1), output)
    #         idx += 1
    #     for i in range(step - 1):  # if we should predict the future
    #         output, hidden = self.rnn(output, hidden, u)
    #         u = self.get_gate_state(ylist[idx].squeeze(dim=1), output)
    #         outputs += [output.squeeze(1)]
    #         idx += 1
    #     outputs = torch.stack(outputs, 1).squeeze(2)
    #     return outputs[0]
