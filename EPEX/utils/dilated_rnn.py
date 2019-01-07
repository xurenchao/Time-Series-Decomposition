import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
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


class DRNN(nn.Module): 
    def __init__(self, input_dim, output_dim, hidden_size, cell='RNN', dilation=[1,2,7], hard_gate=False):
        super(DRNN, self).__init__()
        if cell=='RNN' or cell=='rnn':
            Cell = nn.RNNCell
        elif cell=='GRU' or cell=='gru':
            Cell = nn.GRUCell
        self.hidden_size = hidden_size
        self.layer1 = Cell(input_dim, hidden_size)
        self.layer2 = Cell(hidden_size, hidden_size)
        self.layer3 = Cell(hidden_size, hidden_size)
        self.h2o1 = nn.Linear(hidden_size, output_dim)
        self.h2o2 = nn.Linear(hidden_size, output_dim)
        self.h2o3 = nn.Linear(hidden_size, output_dim)
        self.dilation = dilation
        # self.gate1 = nn.Linear(hidden_size * 2, hidden_size)
        # self.gate2 = nn.Linear(hidden_size * 2, hidden_size)
        # self.binary = BinaryFunction.apply
        # self.hard_gate = hard_gate

    def get_gate_state(self, hidden1, hidden2, gate):
        s = torch.cat((hidden1, hidden2), 1)
        u = torch.sigmoid(gate(s)).squeeze(0)
        if self.hard_gate:
            u = self.binary(u)
        return u

    def init_hidden_state(self, batch_size):
        return to_gpu(torch.zeros(batch_size, self.hidden_size))

    # def init_gate_state(self, batch_size):
    #     return to_gpu(torch.ones(batch_size, self.hidden_size))

    def forward(self, x):
        T = 7
        batch_size = x.shape[0]
        h0 = self.init_hidden_state(batch_size)
        cache1=[]
        cache2=[]
        cache3=[]
        for _ in range(self.dilation[0]):
            cache1.append(h0)
        for _ in range(self.dilation[1]):
            cache2.append(h0)
        for _ in range(self.dilation[2]):
            cache3.append(h0)

        outputs = []
        s_cached = []
        res_loss = 0
        seasonal_loss = 0
        smooth_loss = 0

        for input_t in x.split(1, dim=1):
            h1, h2, h3 = cache1.pop(0), cache2.pop(0), cache3.pop(0)

            hidden1 = self.layer1(input_t.squeeze(1), h1)
            hidden2 = self.layer2(hidden1, h2)
            hidden3 = self.layer3(hidden2, h3)
            cache1.append(hidden1)
            cache2.append(hidden2)
            cache3.append(hidden3)

            output1 = torch.tanh(self.h2o1(hidden1))
            output2 = torch.tanh(self.h2o2(hidden2))
            output3 = torch.tanh(self.h2o3(hidden3))
            output = output1 + output2 + output3
            
            res_loss += F.mse_loss(output1, to_gpu(torch.zeros_like(output1)))

            if len(s_cached) == T:
                s_w = s_cached.pop(0)
                seasonal_loss += F.mse_loss(output2, s_w)

            var = torch.std(output3, dim=1)
            smooth_loss += F.mse_loss(var, to_gpu(torch.zeros_like(var)))

            s_cached.append(output2)

            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs, res_loss, seasonal_loss, smooth_loss

    def forecast(self, x):
        x = x.unsqueeze(0)  # non-batch
        h0 = self.init_hidden_state(1)
        cache1=[]
        cache2=[]
        cache3=[]
        for _ in range(self.dilation[0]):
            cache1.append(h0)
        for _ in range(self.dilation[1]):
            cache2.append(h0)
        for _ in range(self.dilation[2]):
            cache3.append(h0)

        outputs = []
        outputs1 = []
        outputs2 = []
        outputs3 = []
        for input_t in x.split(1, dim=1):
            h1, h2, h3 = cache1.pop(0), cache2.pop(0), cache3.pop(0)
            
            hidden1 = self.layer1(input_t.squeeze(1), h1)
            hidden2 = self.layer2(hidden1, h2)
            hidden3 = self.layer3(hidden2, h3)
            cache1.append(hidden1)
            cache2.append(hidden2)
            cache3.append(hidden3)

            output1 = torch.tanh(self.h2o1(hidden1))
            output2 = torch.tanh(self.h2o2(hidden2))
            output3 = torch.tanh(self.h2o3(hidden3))
            output = output1 + output2 + output3

            outputs += [output]
            outputs1 += [output1]
            outputs2 += [output2]
            outputs3 += [output3]
        outputs = torch.stack(outputs, 1).squeeze(2)
        outputs1 = torch.stack(outputs1, 1).squeeze(2)
        outputs2 = torch.stack(outputs2, 1).squeeze(2)
        outputs3 = torch.stack(outputs3, 1).squeeze(2)
        return outputs[0], outputs1[0], outputs2[0], outputs3[0]

