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


def week_avg_mse(week_list, s):
    n = len(week_list)
    L = 0
    if n == 0:
        return 0
    for si in week_list:
        L += F.mse_loss(s, si)
    return L / n

# 2 RNN layer, a gate netween them and 2 output
class GatedRNN(nn.Module): 
    def __init__(self, input_dim, output_dim, hidden_size, hard_gate=False):
        super(GatedRNN, self).__init__()
        self.hidden_size = hidden_size
        self.layer1 = nn.RNNCell(input_dim, hidden_size)
        self.layer2 = nn.RNNCell(hidden_size, hidden_size)
        self.h2o1 = nn.Linear(hidden_size, output_dim)
        self.h2o2 = nn.Linear(hidden_size, output_dim)
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        self.binary = BinaryFunction.apply
        self.hard_gate = hard_gate

    def get_gate_state(self, hidden1, hidden2):
        s = torch.cat((hidden1, hidden2), 1)
        u = torch.sigmoid(self.gate(s)).squeeze(0)
        if self.hard_gate:
            u = self.binary(u)
        return u

    def init_hidden_state(self, batch_size):
        return to_gpu(torch.zeros(batch_size, self.hidden_size))

    def init_gate_state(self, batch_size):
        return to_gpu(torch.ones(batch_size, self.hidden_size))

    def forward(self, x):
        T = 7
        batch_size = x.shape[0]
        hidden1 = self.init_hidden_state(batch_size)
        hidden2 = self.init_hidden_state(batch_size)
        u = self.init_gate_state(batch_size)
        outputs = []
        for input_t in x.split(1, dim=1):
            hidden1 = self.layer1(input_t.squeeze(1), hidden1)
            hidden2 = u * self.layer2(hidden1, hidden2) + (1 - u) * hidden2
            output = torch.tanh(self.h2o1(hidden1)) + torch.tanh(self.h2o2(hidden2))
            u = self.get_gate_state(hidden1, hidden2)
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

    def forecast(self, x):
        x = x.unsqueeze(0)  # non-batch
        hidden1 = self.init_hidden_state(1)
        hidden2 = self.init_hidden_state(1)
        u = self.init_gate_state(1)
        outputs = []
        outputs1 = []
        outputs2 = []
        for input_t in x.split(1, dim=1):
            hidden1 = self.layer1(input_t.squeeze(1), hidden1)
            hidden2 = u * self.layer2(hidden1, hidden2) + (1 - u) * hidden2
            output1 = torch.tanh(self.h2o1(hidden1))
            output2 = torch.tanh(self.h2o2(hidden2))
            output = output1 + output2

            u = self.get_gate_state(hidden1, hidden2)
            outputs += [output]
            outputs1 += [output1]
            outputs2 += [output2]
        outputs = torch.stack(outputs, 1).squeeze(2)
        outputs1 = torch.stack(outputs1, 1).squeeze(2)
        outputs2 = torch.stack(outputs2, 1).squeeze(2)
        return outputs[0], outputs1[0], outputs2[0]

# 2 GRU layer, no gate and 2 output
class StackGRU(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, hard_gate=False):
        super(StackGRU, self).__init__()
        self.hidden_size = hidden_size
        self.layer1 = nn.GRUCell(input_dim, hidden_size)
        self.layer2 = nn.GRUCell(hidden_size, hidden_size)
        self.h2o1 = nn.Linear(hidden_size, output_dim)
        self.h2o2 = nn.Linear(hidden_size, output_dim)

    def init_hidden_state(self, batch_size):
        return to_gpu(torch.zeros(batch_size, self.hidden_size))

    def forward(self, x):
        T = 7
        batch_size = x.shape[0]
        hidden1 = self.init_hidden_state(batch_size)
        hidden2 = self.init_hidden_state(batch_size)
        outputs = []
        s_t_cached = [torch.zeros_like(hidden2)] * T
        aux_loss = 0
        for input_t in x.split(1, dim=1):
            s_t_T = hidden2
            hidden1 = self.layer1(input_t.squeeze(1), hidden1)
            hidden2 = self.layer2(hidden1, hidden2)
            output = torch.tanh(self.h2o1(hidden1)) + torch.tanh(self.h2o2(hidden2))
            outputs += [output]

            s_t = hidden2
            s_t_T = s_t_cached.pop(0)
            aux_loss += F.mse_loss(s_t, s_t_T)
            s_t_cached.append(s_t)

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs, aux_loss

    def forecast(self, x):
        x = x.unsqueeze(0)  # non-batch
        hidden1 = self.init_hidden_state(1)
        hidden2 = self.init_hidden_state(1)
        outputs = []
        outputs1 = []
        outputs2 = []
        for input_t in x.split(1, dim=1):
            hidden1 = self.layer1(input_t.squeeze(1), hidden1)
            hidden2 = self.layer2(hidden1, hidden2)
            output1, output2 = torch.tanh(self.h2o1(hidden1)), torch.tanh(self.h2o2(hidden2))
            output = output1 + output2

            outputs += [output]
            outputs1 += [output1]
            outputs2 += [output2]
        outputs = torch.stack(outputs, 1).squeeze(2)
        outputs1 = torch.stack(outputs1, 1).squeeze(2)
        outputs2 = torch.stack(outputs2, 1).squeeze(2)
        return outputs[0], outputs1[0], outputs2[0]

# 2 GRU layer, no gate and 3 output branch
class MultiGRU(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, hard_gate=False):
        super(MultiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.layer1 = nn.GRUCell(input_dim, hidden_size)
        self.layer2 = nn.GRUCell(hidden_size, hidden_size)
        self.layer3 = nn.GRUCell(hidden_size, hidden_size)
        self.layer4 = nn.GRUCell(hidden_size, hidden_size)
        self.h2o1 = nn.Linear(hidden_size, output_dim)
        self.h2o2 = nn.Linear(hidden_size, output_dim)
        self.h2o3 = nn.Linear(hidden_size, output_dim)

    def init_hidden_state(self, batch_size):
        return to_gpu(torch.zeros(batch_size, self.hidden_size))

    def forward(self, x):
        T = 7
        batch_size = x.shape[0]
        hidden1 = self.init_hidden_state(batch_size)
        hidden2 = self.init_hidden_state(batch_size)
        hidden3 = self.init_hidden_state(batch_size)
        hidden4 = self.init_hidden_state(batch_size)
        outputs = []
        s_t_cached = []
        s_d_cached = []
        aux_loss1, aux_loss2, aux_loss3 = 0, 0, 0

        for input_t in x.split(1, dim=1):
            hidden1 = self.layer1(input_t.squeeze(1), hidden1)
            hidden2 = self.layer2(hidden1, hidden2)
            hidden3 = self.layer3(hidden1, hidden3)
            hidden4 = self.layer4(hidden1, hidden4)
            output1, output2, output3 = torch.tanh(self.h2o1(hidden2)), torch.tanh(
                self.h2o2(hidden3)), torch.tanh(self.h2o3(hidden4))
            output = output1 + output2 + output3
            outputs += [output]

            if len(s_t_cached) == T:
                s_t = s_t_cached.pop(0)
                aux_loss1 += F.mse_loss(hidden2, s_t)
            if len(s_d_cached) == 1:
                s_d = s_d_cached.pop(0)
                aux_loss2 += F.mse_loss(hidden3, s_d)

            var = torch.std(torch.tanh(self.h2o1(hidden4)), 1)
            aux_loss3 += F.mse_loss(var, to_gpu(torch.zeros_like(var)))

            s_t_cached.append(hidden2)
            s_d_cached.append(hidden3)

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs, aux_loss1, aux_loss2, aux_loss3

    def forecast(self, x):
        x = x.unsqueeze(0)  # non-batch
        hidden1 = self.init_hidden_state(1)
        hidden2 = self.init_hidden_state(1)
        hidden3 = self.init_hidden_state(1)
        hidden4 = self.init_hidden_state(1)
        outputs = []
        outputs1 = []
        outputs2 = []
        outputs3 = []
        for input_t in x.split(1, dim=1):
            hidden1 = self.layer1(input_t.squeeze(1), hidden1)
            hidden2 = self.layer2(hidden1, hidden2)
            hidden3 = self.layer3(hidden1, hidden3)
            hidden4 = self.layer4(hidden1, hidden4)
            output1, output2, output3 = torch.tanh(self.h2o1(hidden2)), torch.tanh(
                self.h2o2(hidden3)), torch.tanh(self.h2o3(hidden4))
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


class ND(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, hard_gate=False):
        super(ND, self).__init__()
        self.hidden_size = hidden_size
        self.layer1 = nn.GRUCell(input_dim, hidden_size)
        self.layer2 = nn.GRUCell(hidden_size, hidden_size)
        self.layer3 = nn.GRUCell(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, hidden_size)
        self.h2o1 = nn.Linear(hidden_size, output_dim, bias=False)
        self.h2o2 = nn.Linear(hidden_size, output_dim, bias=False)
        self.h2o3 = nn.Linear(hidden_size, output_dim)

    def init_hidden_state(self, batch_size):
        return to_gpu(torch.zeros(batch_size, self.hidden_size))

    def forward(self, x):
        T = 7
        batch_size = x.shape[0]
        hidden1 = self.init_hidden_state(batch_size)
        hidden2 = self.init_hidden_state(batch_size)
        hidden3 = self.init_hidden_state(batch_size)
        outputs = []
        s_t_cached = []
        s_d_cached = []
        aux_loss1, aux_loss2, aux_loss3 = 0, 0, 0

        for input_t in x.split(1, dim=1):
            hidden1 = self.layer1(input_t.squeeze(1), hidden1)
            hidden2 = self.layer2(hidden1, hidden2)
            hidden3 = self.layer3(hidden1, hidden3)
            f = torch.tanh(self.layer4(hidden1))
            output1, output2, output3 = torch.tanh(self.h2o1(hidden2)), torch.tanh(
                self.h2o2(hidden3)), torch.tanh(self.h2o3(f))
            output = output1 + output2 + output3
            outputs += [output]

            if len(s_t_cached) == T:
                s_t = s_t_cached.pop(0)
                aux_loss1 += F.mse_loss(output1, s_t)
            if len(s_d_cached) == 1:
                s_d = s_d_cached.pop(0)
                aux_loss2 += F.mse_loss(output2, s_d)
            # var = torch.std(torch.tanh(self.h2o1(hidden4)),1)
            # aux_loss3 += F.mse_loss(var, to_gpu(torch.zeros_like(var)))

            s_t_cached.append(output1)
            s_d_cached.append(output2)

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs, aux_loss1, aux_loss2  # , aux_loss3

    def forecast(self, x):
        x = x.unsqueeze(0)  # non-batch
        hidden1 = self.init_hidden_state(1)
        hidden2 = self.init_hidden_state(1)
        hidden3 = self.init_hidden_state(1)
        outputs = []
        outputs1 = []
        outputs2 = []
        outputs3 = []
        for input_t in x.split(1, dim=1):
            hidden1 = self.layer1(input_t.squeeze(1), hidden1)
            hidden2 = self.layer2(hidden1, hidden2)
            hidden3 = self.layer3(hidden1, hidden3)
            f = torch.tanh(self.layer4(hidden1))
            output1, output2, output3 = torch.tanh(self.h2o1(hidden2)), torch.tanh(
                self.h2o2(hidden3)), torch.tanh(self.h2o3(f))
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

# stupid structure, unreasonable good
class ResRNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, hard_gate=False):
        super(ResRNN, self).__init__()
        self.hidden_size = hidden_size
        self.layer1 = nn.RNNCell(input_dim, hidden_size)
        self.layer2 = nn.RNNCell(hidden_size, hidden_size)
        self.layer3 = nn.RNNCell(hidden_size * 2, hidden_size)
        self.h2o1 = nn.Linear(hidden_size, output_dim)
        self.h2o2 = nn.Linear(hidden_size, output_dim)
        self.gate1 = nn.Linear(hidden_size * 2, hidden_size)
        self.gate2 = nn.Linear(hidden_size * 3, hidden_size)
        self.binary = BinaryFunction.apply
        self.hard_gate = hard_gate

    def init_hidden_state(self, batch_size):
        return to_gpu(torch.zeros(batch_size, self.hidden_size))

    def init_gate_state(self, batch_size):
        return to_gpu(torch.ones(batch_size, self.hidden_size))

    def forward(self, x):
        batch_size = x.shape[0]
        hidden1 = self.init_hidden_state(batch_size)
        hidden2 = self.init_hidden_state(batch_size)
        hidden3 = self.init_hidden_state(batch_size)
        z1 = self.init_gate_state(batch_size)
        z2 = self.init_gate_state(batch_size)
        outputs = []
        for input_t in x.split(1, dim=1):
            hidden1 = self.layer1(input_t.squeeze(1), hidden1)
            hidden2 = z1 * self.layer2(hidden1, hidden2) + (1 - z1) * hidden2
            hidden3 = z2 * self.layer3(torch.cat((hidden1, hidden2), 1), hidden3) + (1 - z2) * hidden3

            z1 = torch.sigmoid(self.gate1(torch.cat((hidden1, hidden2), 1))).squeeze(0)
            z2 = torch.sigmoid(self.gate2(torch.cat((hidden1, hidden2, hidden3), 1))).squeeze(0)
            if self.hard_gate:
                z1 = self.binary(z1)
                # z2 = self.binary(z2)
            z1 = z1 * z2

            output = torch.tanh(self.h2o1(hidden2)) + torch.tanh(self.h2o2(hidden3))
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

    def forecast(self, x):
        x = x.unsqueeze(0)  # non-batch
        hidden1 = self.init_hidden_state(1)
        hidden2 = self.init_hidden_state(1)
        hidden3 = self.init_hidden_state(1)
        z1 = self.init_gate_state(1)
        z2 = self.init_gate_state(1)
        outputs = []
        outputs1 = []
        outputs2 = []
        for input_t in x.split(1, dim=1):
            hidden1 = self.layer1(input_t.squeeze(1), hidden1)
            hidden2 = z1 * self.layer2(hidden1, hidden2) + (1 - z1) * hidden2
            hidden3 = z2 * self.layer3(torch.cat((hidden1, hidden2), 1), hidden3) + (1 - z2) * hidden3

            z1 = torch.sigmoid(self.gate1(torch.cat((hidden1, hidden2), 1))).squeeze(0)
            z2 = torch.sigmoid(self.gate2(torch.cat((hidden1, hidden2, hidden3), 1))).squeeze(0)
            if self.hard_gate:
                z1 = self.binary(z1)
                # z2 = self.binary(z2)
            z1 = z1 * z2

            output1 = torch.tanh(self.h2o1(hidden2))
            output2 = torch.tanh(self.h2o2(hidden3))
            output = output1 + output2

            outputs += [output]
            outputs1 += [output1]
            outputs2 += [output2]
        outputs = torch.stack(outputs, 1).squeeze(2)
        outputs1 = torch.stack(outputs1, 1).squeeze(2)
        outputs2 = torch.stack(outputs2, 1).squeeze(2)
        return outputs[0], outputs1[0], outputs2[0]
