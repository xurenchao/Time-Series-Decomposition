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
        x = ctx.saved_variables[0]
        x_grad = None

        if ctx.needs_input_grad[0]:
            x_grad = grad_output.clone()

        return x_grad


class StackGatedRNN(nn.Module):
    def __init__(self, seq_dim, hidden_size):
        super(StackGatedRNN, self).__init__()
        self.seq_dim = seq_dim
        self.hidden_size = hidden_size
        self.l1_fc1 = nn.Linear(seq_dim, hidden_size)
        self.l1_fc2 = nn.Linear(hidden_size, hidden_size)
        self.l2_fc1 = nn.Linear(hidden_size, hidden_size)
        self.l2_fc2 = nn.Linear(hidden_size, hidden_size)
        self.l1_h2o = nn.Linear(hidden_size, seq_dim)
        self.l2_h2o = nn.Linear(hidden_size, seq_dim)
        self.h2g = nn.Linear(hidden_size * 2, hidden_size)
        self.binary = BinaryFunction.apply

    def init_hidden_state(self):
        return to_gpu(torch.zeros(1, self.hidden_size))

    def init_gate_state(self):
        return to_gpu(torch.ones(1, 1))

    def get_gate_state(self, hidden1, hidden2):
        s = torch.cat((hidden1, hidden2), 1)
        _u = F.sigmoid(self.h2g(s)).squeeze(0)
        u = self.binary(_u)
        # u = _u
        return u

    def forward(self, x):
        l1_hidden = self.init_hidden_state()
        l2_hidden = self.init_hidden_state()
        u = self.init_gate_state()
        outputs = []
        for input_t in x.split(1, dim=1):
            l1_hidden = F.tanh(self.l1_fc1(input_t.squeeze(1)) + self.l1_fc2(l1_hidden))
            l2_hidden = u * F.tanh(self.l2_fc1(l1_hidden) + self.l2_fc2(l2_hidden)) + (1 - u) * l2_hidden
            output = self.l1_h2o(l1_hidden) + self.l2_h2o(l2_hidden)
            u = self.get_gate_state(l1_hidden, l2_hidden)
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

    def forecast(self, x):
        x = x.unsqueeze(0)  # non-batch
        l1_hidden = self.init_hidden_state()
        l2_hidden = self.init_hidden_state()
        u = self.init_gate_state()
        outputs = []
        outputs1 = []
        outputs2 = []
        for input_t in x.split(1, dim=1):
            l1_hidden = F.tanh(self.l1_fc1(input_t.squeeze(1)) + self.l1_fc2(l1_hidden))
            l2_hidden = u * F.tanh(self.l2_fc1(l1_hidden) + self.l2_fc2(l2_hidden)) + (1 - u) * l2_hidden
            output1 = self.l1_h2o(l1_hidden)
            output2 = self.l2_h2o(l2_hidden)
            output = output1 + output2

            u = self.get_gate_state(l1_hidden, l2_hidden)
            outputs += [output]
            outputs1 += [output1]
            outputs2 += [output2]
        outputs = torch.stack(outputs, 1).squeeze(2)
        outputs1 = torch.stack(outputs1, 1).squeeze(2)
        outputs2 = torch.stack(outputs2, 1).squeeze(2)
        return outputs[0], outputs1[0], outputs2[0]

    def self_forecast(self, x, step):
        x = x.unsqueeze(0)  # non-batch
        l1_hidden = self.init_hidden_state()
        l2_hidden = self.init_hidden_state()
        u = self.init_gate_state()
        outputs = []
        for input_t in x.split(1, dim=1):
            l1_hidden = F.tanh(self.l1_fc1(input_t.squeeze(1)) + self.l1_fc2(l1_hidden))
            l2_hidden = F.tanh(self.l2_fc1(l1_hidden) + self.l2_fc2(l2_hidden))
            output = self.l1_h2o(l1_hidden) + self.l2_h2o(l2_hidden)
            u = self.get_gate_state(l1_hidden, l2_hidden)
            outputs += [output]
        for i in range(step - 1):  # if we should predict the future
            l1_hidden = F.tanh(self.l1_fc1(output) + self.l1_fc2(l1_hidden))
            l2_hidden = u * F.tanh(self.l2_fc1(l1_hidden) + self.l2_fc2(l2_hidden)) + (1 - u) * l2_hidden
            output = self.l1_h2o(l1_hidden) + self.l2_h2o(l2_hidden)
            u = self.get_gate_state(l1_hidden, l2_hidden)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs[0]


