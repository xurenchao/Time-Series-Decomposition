import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from tool import to_gpu
import numpy as numpy
import types


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.cell = nn.RNNCell(input_dim, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_dim)

    def init_hidden_state(self, batch_size):
        return to_gpu(torch.zeros(batch_size, self.hidden_size))

    def forward(self, x):
        outputs = []
        hidden = self.init_hidden_state(x.shape[0])
        for input_t in x.split(1, dim=1):
            hidden = self.cell(input_t.squeeze(1), hidden)
            output = self.h2o(hidden)
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

    def forecast(self, x, hidden=None):
        x = x.unsqueeze(0)
        outputs = []
        hiddens = []
        if not torch.is_tensor(hidden):
            hidden = self.init_hidden_state(1)
        for input_t in x.split(1, dim=1):
            hidden = self.cell(input_t.squeeze(1), hidden)
            output = self.h2o(hidden)
            outputs += [output]
            hiddens += [hidden]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs[0], hiddens

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.cell = nn.LSTMCell(input_dim, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_dim)

    def init_hidden_state(self, batch_size):
        return to_gpu(torch.zeros(batch_size, self.hidden_size))

    def forward(self, x):
        outputs = []
        hidden = self.init_hidden_state(x.shape[0])
        cell = self.init_hidden_state(x.shape[0])
        for input_t in x.split(1, dim=1):
            hidden, cell = self.cell(input_t.squeeze(1), (hidden, cell))
            output = self.h2o(hidden)
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

    def forecast(self, x, state=None):
        x = x.unsqueeze(0)
        outputs = []
        states = []
        if state==None:
            hidden = self.init_hidden_state(1)
            cell = self.init_hidden_state(1)
        else:
            hidden, cell = state
        for input_t in x.split(1, dim=1):
            hidden, cell = self.cell(input_t.squeeze(1), (hidden, cell))
            output = self.h2o(hidden)
            outputs += [output]
            states += [(hidden, cell)]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs[0], states

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.cell = nn.GRUCell(input_dim, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_dim)

    def init_hidden_state(self, batch_size):
        return to_gpu(torch.zeros(batch_size, self.hidden_size))

    def forward(self, x):
        outputs = []
        hidden = self.init_hidden_state(x.shape[0])
        for input_t in x.split(1, dim=1):
            hidden = self.cell(input_t.squeeze(1), hidden)
            output = self.h2o(hidden)
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

    def forecast(self, x, hidden=None):
        x = x.unsqueeze(0)
        outputs = []
        hiddens = []
        if not torch.is_tensor(hidden):
            hidden = self.init_hidden_state(1)
        for input_t in x.split(1, dim=1):
            hidden = self.cell(input_t.squeeze(1), hidden)
            output = self.h2o(hidden)
            outputs += [output]
            hiddens += [hidden]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs[0], hiddens


# Attention Models
class GRU_a(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim):
        super(GRU_a, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.cell = nn.GRUCell(input_dim, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_dim)
        self.W1 = nn.Linear(hidden_size, input_dim)
        self.W2 = nn.Linear(input_dim, input_dim)

    def init_hidden_state(self, batch_size):
        return to_gpu(torch.zeros(batch_size, self.hidden_size))

    def init_attention(self, batch_size, n=0):
        if n == 0:
            return to_gpu(torch.zeros(batch_size, self.input_dim))
        else:
            return to_gpu(torch.ones(batch_size, self.input_dim))

    def forward(self, x):
        batch_size = x.shape[0]
        outputs = []
        hidden = self.init_hidden_state(batch_size)
        a1 = self.init_attention(batch_size, 1)
        a2 = self.init_attention(batch_size, 0)
        for input_t in x.split(1, dim=1):
            h = input_t.squeeze(1)
            h1, h2 = h[:,:24], h[:,24:]
            c = a1 * h1 + a2 * h2
            hidden = self.cell(c, hidden)
            output = torch.tanh(self.h2o(hidden))
            outputs += [output]

            e1 = torch.exp(torch.tanh(self.W1(hidden)+self.W2(h1)))
            e2 = torch.exp(torch.tanh(self.W1(hidden)+self.W2(h2)))
            a1 = e1/(e1+e2)
            a2 = e2/(e1+e2)

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

    def forecast(self, x, step):
        T = 7
        outputs = []
        hidden = self.init_hidden_state(1)
        a1 = self.init_attention(1, 1)
        a2 = self.init_attention(1, 0)
        for input_t in x.split(1, dim=0):
            h = input_t.squeeze(1)
            h1, h2 = h[:,:24], h[:,24:]
            c = a1 * h1 + a2 * h2
            hidden = self.cell(c, hidden)
            output = torch.tanh(self.h2o(hidden))
            outputs += [output]

            e1 = torch.exp(torch.tanh(self.W1(hidden)+self.W2(h1)))
            e2 = torch.exp(torch.tanh(self.W1(hidden)+self.W2(h2)))
            a1 = e1/(e1+e2)
            a2 = e2/(e1+e2)

        keep = torch.clone(x[-T:, 24:])
        h1 = output
        h2 = keep[0]
        for i in range(step):  # if we should predict the future
            c = a1 * h1 + a2 * h2
            hidden = self.cell(c, hidden)
            output = torch.tanh(self.h2o(hidden))
            outputs += [output]

            e1 = torch.exp(torch.tanh(self.W1(hidden)+self.W2(h1)))
            e2 = torch.exp(torch.tanh(self.W1(hidden)+self.W2(h2)))
            a1 = e1/(e1+e2)
            a2 = e2/(e1+e2)

            h2 = keep[(i + 1) % T]


        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs[0]
