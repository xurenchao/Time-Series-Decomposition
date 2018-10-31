import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from tool import to_gpu
import numpy as numpy
import types


class RNN(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(RNN, self).__init__()
        self.n_hidden = n_hidden
        self.cell = nn.RNNCell(n_input, n_hidden)
        self.h2o = nn.Linear(n_hidden, n_output)

    def init_hidden_state(self, batch_size):
        return to_gpu(torch.zeros(batch_size, self.n_hidden))

    # PeriodDataset, P=1
    def forward(self, x):
        outputs = []
        hidden = self.init_hidden_state(x.shape[0])
        for input_t in x.split(1, dim=1):
            hidden = self.cell(input_t.squeeze(1), hidden)
            output = torch.tanh(self.h2o(hidden))
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

    def forecast(self, x, step):
        T = 14
        outputs = []
        hidden = self.init_hidden_state(1)
        for input_t in x.split(1, dim=0):
            hidden = self.cell(input_t, hidden)
            output = torch.tanh(self.h2o(hidden))
            outputs += [output]
        keep = torch.clone(x[-T:, :])
        catout = keep[0]
        catout[0:24] = output
        for i in range(step):  # if we should predict the future
            hidden = self.cell(catout.unsqueeze(0), hidden)
            output = torch.tanh(self.h2o(hidden))
            outputs += [output]
            catout = keep[(i + 1) % T]
            catout[0:24] = output

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs[0]

    # # DailyDataset
    # def forward(self, x):
    #     b, d, h = x.shape
    #     hidden = self.run(x.reshape(b, 1, d * h))
    #     output = torch.tanh(self.h2o(hidden))

    #     return output[0]

    # def forecast(self, x, step):
    #     # x = x.unsqueeze(0)
    #     d, h = x.shape
    #     outputs = []
    #     for i in range(d - self.W + 1):
    #         hidden = self.run(x[i:i + self.W, :].reshape(1, 1, self.W * h))
    #         output = torch.tanh(self.h2o(hidden))
    #         outputs += [output]
    #     keep = x[-self.W:, :]
    #     catout = keep[0]
    #     catout[0:24] = output
    #     x = torch.cat((x, [catout]), dim=0)
    #     for i in for i in range(step):  # if we should predict the future
    #         hidden = self.run(x[-self.W:, :].reshape(1, 1, self.W * h))
    #         output = torch.tanh(self.h2o(hidden))
    #         outputs += [output]
    #         catout = keep[(i+1) % self.W]
    #         catout[0:24] = output
    #         x = torch.cat((x, [catout]), dim=0)

    #     outputs = torch.stack(outputs, 1).squeeze(2)[0]

    #     return outputs


class GatedRNN(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(GatedRNN, self).__init__()
        self.n_hidden = n_hidden
        self.layer1 = nn.RNNCell(n_input, n_hidden)
        self.layer2 = nn.RNNCell(n_hidden, n_hidden)
        self.h2o1 = nn.Linear(n_hidden, n_output)
        self.h2o2 = nn.Linear(n_hidden, n_output)
        self.gate = nn.Linear(n_hidden * 2, n_hidden)

    def get_gate_state(self, hidden1, hidden2):
        s = torch.cat((hidden1, hidden2), 1)
        u = torch.sigmoid(self.gate(s)).squeeze(0)
        return u

    def init_hidden_state(self, batch_size):
        return to_gpu(torch.zeros(batch_size, self.n_hidden))

    def init_gate_state(self, batch_size):
        return to_gpu(torch.ones(batch_size, self.n_hidden))

    def forward(self, x):
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

    def forecast(self, x, step):
        T = 14
        hidden1 = self.init_hidden_state(1)
        hidden2 = self.init_hidden_state(1)
        u = self.init_gate_state(1)
        outputs = []
        for input_t in x.split(1, dim=0):
            hidden1 = self.layer1(input_t.squeeze(1), hidden1)
            hidden2 = u * self.layer2(hidden1, hidden2) + (1 - u) * hidden2
            output = torch.tanh(self.h2o1(hidden1)) + torch.tanh(self.h2o2(hidden2))
            u = self.get_gate_state(hidden1, hidden2)
            outputs += [output]

        keep = torch.clone(x[-T:, :])
        catout = keep[0]
        catout[0:24] = output
        for i in range(step):  # if we should predict the future
            hidden1 = self.layer1(catout.unsqueeze(0), hidden1)
            hidden2 = u * self.layer2(hidden1, hidden2) + (1 - u) * hidden2
            u = self.get_gate_state(hidden1, hidden2)
            output = torch.tanh(self.h2o1(hidden1)) + torch.tanh(self.h2o2(hidden2))
            outputs += [output]
            catout = keep[(i + 1) % T]
            catout[0:24] = output
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs[0]

class GatedRNN_a(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(GatedRNN_a, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.layer1 = nn.RNNCell(n_input, n_hidden)
        self.layer2 = nn.RNNCell(n_hidden, n_hidden)
        self.h2o1 = nn.Linear(n_hidden, n_output)
        self.h2o2 = nn.Linear(n_hidden, n_output)
        self.gate = nn.Linear(n_hidden * 2, n_hidden)
        self.W1 = nn.Linear(n_hidden, n_input)
        self.W2 = nn.Linear(n_input, n_input)

    def get_gate_state(self, hidden1, hidden2):
        s = torch.cat((hidden1, hidden2), 1)
        u = torch.sigmoid(self.gate(s)).squeeze(0)
        return u

    def init_hidden_state(self, batch_size):
        return to_gpu(torch.zeros(batch_size, self.n_hidden))

    def init_gate_state(self, batch_size):
        return to_gpu(torch.ones(batch_size, self.n_hidden))
    
    def init_attention(self, batch_size, n=0):
        if n == 0:
            return to_gpu(torch.zeros(batch_size, self.n_input))
        else:
            return to_gpu(torch.ones(batch_size, self.n_input))

    def forward(self, x):
        batch_size = x.shape[0]
        hidden1 = self.init_hidden_state(batch_size)
        hidden2 = self.init_hidden_state(batch_size)
        u = self.init_gate_state(batch_size)
        a1 = self.init_attention(batch_size, 1)
        a2 = self.init_attention(batch_size, 0)
        outputs = []
        for input_t in x.split(1, dim=1):
            h = input_t.squeeze(1)
            h1, h2 = h[:,:24], h[:,24:]
            c = a1 * h1 + a2 * h2
            hidden1 = self.layer1(c, hidden1)
            hidden2 = u * self.layer2(hidden1, hidden2) + (1 - u) * hidden2
            output = torch.tanh(self.h2o1(hidden1)) + torch.tanh(self.h2o2(hidden2))
            u = self.get_gate_state(hidden1, hidden2)
            outputs += [output]

            e1 = torch.exp(torch.tanh(self.W1(hidden2)+self.W2(h1)))
            e2 = torch.exp(torch.tanh(self.W1(hidden2)+self.W2(h2)))
            a1 = e1/(e1+e2)
            a2 = e2/(e1+e2)

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

    def forecast(self, x, step):
        T = 7
        hidden1 = self.init_hidden_state(1)
        hidden2 = self.init_hidden_state(1)
        u = self.init_gate_state(1)
        a1 = self.init_attention(1, 1)
        a2 = self.init_attention(1, 0)
        outputs = []
        for input_t in x.split(1, dim=0):
            h = input_t.squeeze(1)
            h1, h2 = h[:,:24], h[:,24:]
            c = a1 * h1 + a2 * h2
            hidden1 = self.layer1(c, hidden1)
            hidden2 = u * self.layer2(hidden1, hidden2) + (1 - u) * hidden2
            output = torch.tanh(self.h2o1(hidden1)) + torch.tanh(self.h2o2(hidden2))
            u = self.get_gate_state(hidden1, hidden2)
            outputs += [output]

            e1 = torch.exp(torch.tanh(self.W1(hidden2)+self.W2(h1)))
            e2 = torch.exp(torch.tanh(self.W1(hidden2)+self.W2(h2)))
            a1 = e1/(e1+e2)
            a2 = e2/(e1+e2)

        keep = torch.clone(x[-T:, 24:])
        h1 = output
        h2 = keep[0]
        for i in range(step):  # if we should predict the future
            c = a1 * h1 + a2 * h2
            hidden1 = self.layer1(c, hidden1)
            hidden2 = u * self.layer2(hidden1, hidden2) + (1 - u) * hidden2
            u = self.get_gate_state(hidden1, hidden2)
            output = torch.tanh(self.h2o1(hidden1)) + torch.tanh(self.h2o2(hidden2))
            outputs += [output]
            
            e1 = torch.exp(torch.tanh(self.W1(hidden2)+self.W2(h1)))
            e2 = torch.exp(torch.tanh(self.W1(hidden2)+self.W2(h2)))
            a1 = e1/(e1+e2)
            a2 = e2/(e1+e2)

            h2 = keep[(i + 1) % T]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs[0]


class GRU(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(GRU, self).__init__()
        self.n_hidden = n_hidden
        self.cell = nn.GRUCell(n_input, n_hidden)
        self.h2o = nn.Linear(n_hidden, n_output)

    def init_hidden_state(self, batch_size):
        return to_gpu(torch.zeros(batch_size, self.n_hidden))

    # PeriodDataset, P=1
    def forward(self, x):
        outputs = []
        hidden = self.init_hidden_state(x.shape[0])
        for input_t in x.split(1, dim=1):
            hidden = self.cell(input_t.squeeze(1), hidden)
            output = torch.tanh(self.h2o(hidden))
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

    def forecast(self, x, step):
        T = 7
        outputs = []
        hidden = self.init_hidden_state(1)
        for input_t in x.split(1, dim=0):
            hidden = self.cell(input_t, hidden)
            output = torch.tanh(self.h2o(hidden))
            outputs += [output]
        keep = torch.clone(x[-T:, :])
        catout = keep[0]
        catout[0:24] = output
        for i in range(step):  # if we should predict the future
            hidden = self.cell(catout.unsqueeze(0), hidden)
            output = torch.tanh(self.h2o(hidden))
            outputs += [output]
            catout = keep[(i + 1) % T]
            catout[0:24] = output

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs[0]


class StackGRU(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(StackGRU, self).__init__()
        self.n_hidden = n_hidden
        self.layer1 = nn.GRUCell(n_input, n_hidden)
        self.layer2 = nn.GRUCell(n_hidden, n_hidden)
        self.h2o1 = nn.Linear(n_hidden, n_output)
        self.h2o2 = nn.Linear(n_hidden, n_output)

    def init_hidden_state(self, batch_size):
        return to_gpu(torch.zeros(batch_size, self.n_hidden))

    # PeriodDataset, P=1
    def forward(self, x):
        outputs = []
        hidden1 = self.init_hidden_state(x.shape[0])
        hidden2 = self.init_hidden_state(x.shape[0])
        for input_t in x.split(1, dim=1):
            hidden1 = self.layer1(input_t.squeeze(1), hidden1)
            hidden2 = self.layer2(hidden1, hidden2)
            output = torch.tanh(self.h2o1(hidden1)) + torch.tanh(self.h2o2(hidden2))
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

    def forecast(self, x, step):
        T = 7
        outputs = []
        hidden1 = self.init_hidden_state(1)
        hidden2 = self.init_hidden_state(1)
        for input_t in x.split(1, dim=0):
            hidden1 = self.layer1(input_t.squeeze(1), hidden1)
            hidden2 = self.layer2(hidden1, hidden2)
            output = torch.tanh(self.h2o1(hidden1)) + torch.tanh(self.h2o2(hidden2))
            outputs += [output]
        keep = torch.clone(x[-T:, :])
        catout = keep[0]
        catout[0:24] = output
        for i in range(step):  # if we should predict the future
            hidden1 = self.layer1(catout.unsqueeze(0), hidden1)
            hidden2 = self.layer2(hidden1, hidden2)
            output = torch.tanh(self.h2o1(hidden1)) + torch.tanh(self.h2o2(hidden2))
            outputs += [output]
            catout = keep[(i + 1) % T]
            catout[0:24] = output

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs[0]


# Attention Models
class GRU_a(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(GRU_a, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.cell = nn.GRUCell(n_input, n_hidden)
        self.h2o = nn.Linear(n_hidden, n_output)
        self.W1 = nn.Linear(n_hidden, n_input)
        self.W2 = nn.Linear(n_input, n_input)

    def init_hidden_state(self, batch_size):
        return to_gpu(torch.zeros(batch_size, self.n_hidden))

    def init_attention(self, batch_size, n=0):
        if n == 0:
            return to_gpu(torch.zeros(batch_size, self.n_input))
        else:
            return to_gpu(torch.ones(batch_size, self.n_input))

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
