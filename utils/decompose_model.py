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

def moving_average(array, N) :
    ret = np.cumsum(array, axis=1, dtype=float)
    ret[:, N:] = ret[:, N:] - ret[:, :-N]
    return ret[:, N-1:] / N


# 3 RNN layers decomposing model, gate between layers and 3 outputs
class TDec_RNN(nn.Module): 
    def __init__(self, input_dim, output_dim, hidden_size, cell='RNN', hard_gate=False):
        super(TDec_RNN, self).__init__()
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
        self.gate1 = nn.Linear(hidden_size * 2, hidden_size)
        self.gate2 = nn.Linear(hidden_size * 2, hidden_size)
        self.binary = BinaryFunction.apply
        self.hard_gate = hard_gate

    def get_gate_state(self, hidden1, hidden2, gate):
        s = torch.cat((hidden1, hidden2), 1)
        u = torch.sigmoid(gate(s)).squeeze(0)
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
        hidden3 = self.init_hidden_state(batch_size)
        u1 = self.init_gate_state(batch_size)
        u2 = self.init_gate_state(batch_size)
        outputs = []
        s_cached = []
        res_loss = 0
        seasonal_loss = 0
        smooth_loss = 0

        for input_t in x.split(1, dim=1):
            hidden1 = self.layer1(input_t.squeeze(1), hidden1)
            hidden2 = u1 * self.layer2(hidden1, hidden2) + (1 - u1) * hidden2
            hidden3 = u2 * self.layer3(hidden2, hidden3) + (1 - u2) * hidden3
            output1 = torch.tanh(self.h2o1(hidden1))
            output2 = torch.tanh(self.h2o2(hidden2))
            output3 = torch.tanh(self.h2o3(hidden3))
            output = output1 + output2 + output3
            u1 = self.get_gate_state(hidden1, hidden2, self.gate1)
            u2 = self.get_gate_state(hidden2, hidden3, self.gate2)
            outputs += [output]

            res_loss += F.mse_loss(output1, to_gpu(torch.zeros_like(output1)))

            if len(s_cached) == T:
                s_w = s_cached.pop(0)
                seasonal_loss += F.mse_loss(output2, s_w)

            var = torch.std(output3, dim=1)
            smooth_loss += F.mse_loss(var, to_gpu(torch.zeros_like(var)))

            s_cached.append(output2)

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs, res_loss, seasonal_loss, smooth_loss

    def forecast(self, x, state=None):
        x = x.unsqueeze(0)  # non-batch
        if state==None:
            hidden1 = self.init_hidden_state(1)
            hidden2 = self.init_hidden_state(1)
            hidden3 = self.init_hidden_state(1)
            u1 = self.init_gate_state(1)
            u2 = self.init_gate_state(1)
        else:
            hidden1, hidden2, hidden3, u1, u2 = state
        outputs = []
        outputs1 = []
        outputs2 = []
        outputs3 = []
        for input_t in x.split(1, dim=1):
            hidden1 = self.layer1(input_t.squeeze(1), hidden1)
            hidden2 = u1 * self.layer2(hidden1, hidden2) + (1 - u1) * hidden2
            hidden3 = u2 * self.layer3(hidden2, hidden3) + (1 - u2) * hidden3
            output1 = torch.tanh(self.h2o1(hidden1))
            output2 = torch.tanh(self.h2o2(hidden2))
            output3 = torch.tanh(self.h2o3(hidden3))
            output = output1 + output2 + output3

            u1 = self.get_gate_state(hidden1, hidden2, self.gate1)
            u2 = self.get_gate_state(hidden2, hidden3, self.gate2)
            outputs += [output]
            outputs1 += [output1]
            outputs2 += [output2]
            outputs3 += [output3]
        outputs = torch.stack(outputs, 1).squeeze(2)
        outputs1 = torch.stack(outputs1, 1).squeeze(2)
        outputs2 = torch.stack(outputs2, 1).squeeze(2)
        outputs3 = torch.stack(outputs3, 1).squeeze(2)
        state = (hidden1, hidden2, hidden3, u1, u2)
        return outputs[0], outputs1[0], outputs2[0], outputs3[0], state

class TDec_RNN_v1(nn.Module): 
    def __init__(self, input_dim, output_dim, hidden_size, cell='RNN', hard_gate=False):
        super(TDec_RNN_v1, self).__init__()
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
        self.gate1 = nn.Linear(hidden_size * 2, hidden_size)
        self.gate2 = nn.Linear(hidden_size * 2, hidden_size)
        self.binary = BinaryFunction.apply
        self.hard_gate = hard_gate

    def get_gate_state(self, hidden1, hidden2, gate):
        s = torch.cat((hidden1, hidden2), 1)
        u = torch.sigmoid(gate(s)).squeeze(0)
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
        hidden3 = self.init_hidden_state(batch_size)
        u1 = self.init_gate_state(batch_size)
        u2 = self.init_gate_state(batch_size)
        outputs = []
        s_cached1 = []
        s_cached2 = []
        res_loss = 0
        seasonal_loss = 0
        smooth_loss = 0

        for input_t in x.split(1, dim=1):
            hidden1 = self.layer1(input_t.squeeze(1), hidden1)
            hidden2 = u1 * self.layer2(hidden1, hidden2) + (1 - u1) * hidden2
            hidden3 = u2 * self.layer3(hidden2, hidden3) + (1 - u2) * hidden3
            output1 = torch.tanh(self.h2o1(hidden1))
            output2 = torch.tanh(self.h2o2(hidden2))
            output3 = torch.tanh(self.h2o3(hidden3))
            output = output1 + output2 + output3
            u1 = self.get_gate_state(hidden1, hidden2, self.gate1)
            u2 = self.get_gate_state(hidden2, hidden3, self.gate2)
            outputs += [output]

            res_loss += F.mse_loss(output1, to_gpu(torch.zeros_like(output1)))

            if len(s_cached1) == T:
                s_w = s_cached1.pop(0)
                seasonal_loss += F.mse_loss(output2, s_w)

            if len(s_cached2) == 2:
                s_2 = s_cached2.pop(0)
                smooth_loss += 30 * F.mse_loss(s_2.mean(dim=1), output3.mean(dim=1))
            var = torch.std(output3, dim=1)
            smooth_loss += F.mse_loss(var, to_gpu(torch.zeros_like(var)))

            s_cached1.append(output2)

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs, res_loss, seasonal_loss, smooth_loss

    def forecast(self, x, state=None):
        x = x.unsqueeze(0)  # non-batch
        if state==None:
            hidden1 = self.init_hidden_state(1)
            hidden2 = self.init_hidden_state(1)
            hidden3 = self.init_hidden_state(1)
            u1 = self.init_gate_state(1)
            u2 = self.init_gate_state(1)
        else:
            hidden1, hidden2, hidden3, u1, u2 = state
        outputs = []
        outputs1 = []
        outputs2 = []
        outputs3 = []
        for input_t in x.split(1, dim=1):
            hidden1 = self.layer1(input_t.squeeze(1), hidden1)
            hidden2 = u1 * self.layer2(hidden1, hidden2) + (1 - u1) * hidden2
            hidden3 = u2 * self.layer3(hidden2, hidden3) + (1 - u2) * hidden3
            output1 = torch.tanh(self.h2o1(hidden1))
            output2 = torch.tanh(self.h2o2(hidden2))
            output3 = torch.tanh(self.h2o3(hidden3))
            output = output1 + output2 + output3

            u1 = self.get_gate_state(hidden1, hidden2, self.gate1)
            u2 = self.get_gate_state(hidden2, hidden3, self.gate2)
            outputs += [output]
            outputs1 += [output1]
            outputs2 += [output2]
            outputs3 += [output3]
        outputs = torch.stack(outputs, 1).squeeze(2)
        outputs1 = torch.stack(outputs1, 1).squeeze(2)
        outputs2 = torch.stack(outputs2, 1).squeeze(2)
        outputs3 = torch.stack(outputs3, 1).squeeze(2)
        state = (hidden1, hidden2, hidden3, u1, u2)
        return outputs[0], outputs1[0], outputs2[0], outputs3[0], state


# 3 RNN layers decomposing model, gate between layers and 3 outputs, with moving average
class TDec_RNN_ma(nn.Module): 
    def __init__(self, input_dim, output_dim, hidden_size, cell='RNN', hard_gate=False):
        super(TDec_RNN, self).__init__()
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
        self.gate1 = nn.Linear(hidden_size * 2, hidden_size)
        self.gate2 = nn.Linear(hidden_size * 2, hidden_size)
        self.binary = BinaryFunction.apply
        self.hard_gate = hard_gate


    def get_gate_state(self, hidden1, hidden2, gate):
        s = torch.cat((hidden1, hidden2), 1)
        u = torch.sigmoid(gate(s)).squeeze(0)
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
        hidden3 = self.init_hidden_state(batch_size)
        u1 = self.init_gate_state(batch_size)
        u2 = self.init_gate_state(batch_size)
        outputs = []
        season_cached = []
        smooth_cached = []
        res_loss = 0
        seasonal_loss = 0
        smooth_loss = 0

        for input_t in x.split(1, dim=1):
            hidden1 = self.layer1(input_t.squeeze(1), hidden1)
            hidden2 = u1 * self.layer2(hidden1, hidden2) + (1 - u1) * hidden2
            hidden3 = u2 * self.layer3(hidden2, hidden3) + (1 - u2) * hidden3
            output1 = torch.tanh(self.h2o1(hidden1))
            output2 = torch.tanh(self.h2o2(hidden2))
            output3 = torch.tanh(self.h2o3(hidden3))
            output = output1 + output2 + output3
            u1 = self.get_gate_state(hidden1, hidden2, self.gate1)
            u2 = self.get_gate_state(hidden2, hidden3, self.gate2)
            outputs += [output]

            res_loss += F.mse_loss(output1, to_gpu(torch.zeros_like(output1)))

            if len(season_cached) == T:
                s_w = season_cached.pop(0)
                seasonal_loss += F.mse_loss(output2, s_w)

            
            if smooth_cached == []:
                ma = moving_average(output3.cpu().data.numpy(), N=5)
                smooth = output3.data
                smooth[:, 2:-2] = torch.Tensor(ma)
            else:
                tmp=torch.cat((smooth_cached.pop(), output3), 1)[:,12:]
                ma = moving_average(tmp.cpu().data.numpy(), N=5)[:, 10:]
                smooth = output3.data
                smooth[:, :-2] = torch.Tensor(ma)
            smooth_loss += F.mse_loss(output3, to_gpu(smooth))
            
            var = torch.std(output3, dim=1)
            smooth_loss += F.mse_loss(var, to_gpu(torch.zeros_like(var)))

            season_cached.append(output2)
            smooth_cached.append(output3)


        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs, res_loss, seasonal_loss, smooth_loss

    def forecast(self, x, state=None):
        x = x.unsqueeze(0)  # non-batch
        if state==None:
            hidden1 = self.init_hidden_state(1)
            hidden2 = self.init_hidden_state(1)
            hidden3 = self.init_hidden_state(1)
            u1 = self.init_gate_state(1)
            u2 = self.init_gate_state(1)
        else:
            hidden1, hidden2, hidden3, u1, u2 = state
        outputs = []
        outputs1 = []
        outputs2 = []
        outputs3 = []
        for input_t in x.split(1, dim=1):
            hidden1 = self.layer1(input_t.squeeze(1), hidden1)
            hidden2 = u1 * self.layer2(hidden1, hidden2) + (1 - u1) * hidden2
            hidden3 = u2 * self.layer3(hidden2, hidden3) + (1 - u2) * hidden3
            output1 = torch.tanh(self.h2o1(hidden1))
            output2 = torch.tanh(self.h2o2(hidden2))
            output3 = torch.tanh(self.h2o3(hidden3))
            output = output1 + output2 + output3

            u1 = self.get_gate_state(hidden1, hidden2, self.gate1)
            u2 = self.get_gate_state(hidden2, hidden3, self.gate2)

            outputs += [output]
            outputs1 += [output1]
            outputs2 += [output2]
            outputs3 += [output3]
        outputs = torch.stack(outputs, 1).squeeze(2)
        outputs1 = torch.stack(outputs1, 1).squeeze(2)
        outputs2 = torch.stack(outputs2, 1).squeeze(2)
        outputs3 = torch.stack(outputs3, 1).squeeze(2)
        state = (hidden1, hidden2, hidden3, u1, u2)
        return outputs[0], outputs1[0], outputs2[0], outputs3[0], state
