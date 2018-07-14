import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.optim as optim
from tool import to_gpu
import numpy as np
import types


class FS_RNN(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, cell_type='GRU', k=4, dropout=0, batch_first=True):
        super(FS_RNN, self).__init__()
        self.W = n_input // 24
        self.n_hidden = n_hidden
        self.batch_first = batch_first
        layers = []
        if cell_type == "GRU":
            cell = nn.GRU
        elif cell_type == "RNN":
            cell = nn.RNN

        for i in range(k):
            if i == 0:
                c = cell(n_input, n_hidden, dropout=dropout)
            else:
                c = cell(n_hidden, n_hidden, dropout=dropout)
            layers.append(c)
        self.fastcells = nn.Sequential(*layers)
        print(self.fastcells)
        self.slowcell = cell(n_hidden, n_hidden, dropout=dropout)
        self.h2o1 = nn.Linear(n_hidden, n_output)
        self.h2o2 = nn.Linear(n_hidden, n_output)

    def fs_rnn(self, inputs, hiddens_f=None, hiddens_s=None):
        if self.batch_first:
            inputs = inputs.transpose(0, 1)
        batch_size = inputs.shape[0]

        for i, cell in enumerate(self.fastcells):
            if i == 0:
                _, h_f = cell(inputs) if hiddens_f is None else cell(inputs, hiddens_f)
                _, h_s = self.slowcell(h_f) if hiddens_s is None else self.slowcell(h_f, hiddens_s)

            elif i == 1:
                _, h_f = cell(h_s, h_f)
            else:
                _, h_f = cell(torch.zeros_like(h_f), h_f)

        return h_f, h_s

    # # DailyDataset
    # def forward(self, x):
    #     b, d, h = x.shape
    #     h_f, h_s = self.fs_rnn(x.reshape(b, 1, d * h))
    #     output = F.tanh(self.h2o1(h_f)) + F.tanh(self.h2o2(h_s))

    #     return output[0]

    # def forecast(self, x):
    #     # x = x.unsqueeze(0)
    #     d, _ = x.shape
    #     outputs = []
    #     outputs1 = []
    #     outputs2 = []
    #     for i in range(d - self.W + 1):
    #         h_f, h_s = self.fs_rnn(x[i:i + self.W, :].reshape(1, 1, self.W * 24))
    #         output1 = F.tanh(self.h2o1(h_f))
    #         output2 = F.tanh(self.h2o2(h_s))
    #         outputs += [output1 + output2]
    #         outputs1 += [output1]
    #         outputs2 += [output2]

    #     outputs = torch.stack(outputs, 1).squeeze(2)[0]
    #     outputs1 = torch.stack(outputs1, 1).squeeze(2)[0]
    #     outputs2 = torch.stack(outputs2, 1).squeeze(2)[0]
    #     return outputs, outputs1, outputs2
    
    # PeriodDataset, P=1
    def forward(self, x):
        outputs = []
        h_f = h_s = None
        for input_t in x.split(1, dim=1):
            h_f, h_s = self.fs_rnn(input_t, h_f, h_s)
            output = F.tanh(self.h2o1(h_f[0]))# + F.tanh(self.h2o2(h_s[0]))
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

    def forecast(self, x):
        x = x.unsqueeze(0)  # non-batch
        h_f = h_s = None
        outputs = []
        outputs1 = []
        outputs2 = []
        for input_t in x.split(1, dim=1):
            h_f, h_s = self.fs_rnn(input_t, h_f, h_s)
            output1 = F.tanh(self.h2o1(h_f[0]))
            # output2 = F.tanh(self.h2o2(h_s[0]))
            output = output1# + output2

            outputs += [output]
            # outputs1 += [output1]
            # outputs2 += [output2]
        outputs = torch.stack(outputs, 1).squeeze(2)
        # outputs1 = torch.stack(outputs1, 1).squeeze(2)
        # outputs2 = torch.stack(outputs2, 1).squeeze(2)
        return outputs#[0], outputs1[0], outputs2[0]

    # def self_forecast(self, x, step):
    #     x = x.unsqueeze(0)  # non-batch
    #     h_f = h_s = None
    #     outputs = []
    #     for input_t in x.split(1, dim=1):
    #         h_f, h_s = self.fs_rnn(input_t, h_f, h_s)
    #         output = F.tanh(self.h2o1(h_f[0])) + F.tanh(self.h2o2(h_s[0]))
    #         outputs += [output]
    #     for i in range(step - 1):  # if we should predict the future
    #         h_f, h_s = self.fs_rnn(output, h_f, h_s)
    #         output = F.tanh(self.h2o1(h_f[0])) + F.tanh(self.h2o2(h_s[0]))
    #         outputs += [output]
    #     outputs = torch.stack(outputs, 1).squeeze(2)
    #     return outputs[0]