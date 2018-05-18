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


class GatedRNN(nn.Module):
    def __init__(self, seq_dim, hidden_size, hard_gate=False):
        super(GatedRNN, self).__init__()
        self.seq_dim = seq_dim
        self.hidden_size = hidden_size
        self.l1_W = nn.Linear(seq_dim, hidden_size)
        self.l1_U = nn.Linear(hidden_size, hidden_size)
        self.l2_W = nn.Linear(hidden_size, hidden_size)
        self.l2_U = nn.Linear(hidden_size, hidden_size)
        self.out1 = nn.Linear(hidden_size, seq_dim)
        self.out2 = nn.Linear(hidden_size, seq_dim)
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        self.binary = BinaryFunction.apply
        self.hard_gate = hard_gate

    def init_hidden_state(self):
        return to_gpu(torch.zeros(1, self.hidden_size))

    def init_gate_state(self):
        return to_gpu(torch.ones(1, 1))

    def get_gate_state(self, hidden1, hidden2):
        s = torch.cat((hidden1, hidden2), 1)
        u = F.sigmoid(self.gate(s)).squeeze(0)
        if self.hard_gate:
            u = self.binary(u)
        return u

    def forward(self, x):
        hidden1 = hidden2 = self.init_hidden_state()
        u = self.init_gate_state()
        outputs = []
        for input_t in x.split(1, dim=1):
            hidden1 = F.tanh(self.l1_W(input_t.squeeze(1)) + self.l1_U(hidden1))
            hidden2 = u * F.tanh(self.l2_W(hidden1) + self.l2_U(hidden2)) + (1 - u) * hidden2
            output = self.out1(hidden1) + self.out2(hidden2)
            u = self.get_gate_state(hidden1, hidden2)
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

    def forecast(self, x):
        x = x.unsqueeze(0)  # non-batch
        hidden1 = hidden2 = self.init_hidden_state()
        u = self.init_gate_state()
        outputs = []
        outputs1 = []
        outputs2 = []
        for input_t in x.split(1, dim=1):
            hidden1 = F.tanh(self.l1_W(input_t.squeeze(1)) + self.l1_U(hidden1))
            hidden2 = u * F.tanh(self.l2_W(hidden1) + self.l2_U(hidden2)) + (1 - u) * hidden2
            output1 = self.out1(hidden1)
            output2 = self.out2(hidden2)
            output = output1 + output2

            u = self.get_gate_state(hidden1, hidden2)
            outputs += [output]
            outputs1 += [output1]
            outputs2 += [output2]
        outputs = torch.stack(outputs, 1).squeeze(2)
        outputs1 = torch.stack(outputs1, 1).squeeze(2)
        outputs2 = torch.stack(outputs2, 1).squeeze(2)
        return outputs[0], outputs1[0], outputs2[0]

    def self_forecast(self, x, step):
        x = x.unsqueeze(0)  # non-batch
        hidden1 = hidden2 = self.init_hidden_state()
        u = self.init_gate_state()
        outputs = []
        for input_t in x.split(1, dim=1):
            hidden1 = F.tanh(self.l1_W(input_t.squeeze(1)) + self.l1_U(hidden1))
            hidden2 = u * F.tanh(self.l2_W(hidden1) + self.l2_U(hidden2)) + (1 - u) * hidden2
            output = self.out1(hidden1) + self.out2(hidden2)
            u = self.get_gate_state(hidden1, hidden2)
            outputs += [output]
        for i in range(step - 1):  # if we should predict the future
            hidden1 = F.tanh(self.l1_W(output) + self.l1_U(hidden1))
            hidden2 = u * F.tanh(self.l2_W(hidden1) + self.l2_U(hidden2)) + (1 - u) * hidden2
            output = self.out1(hidden1) + self.out2(hidden2)
            u = self.get_gate_state(hidden1, hidden2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs[0]

class StackGatedRNN(nn.Module):
    def __init__(self, seq_dim, hidden_size, hard_gate=False):
        super(StackGatedRNN, self).__init__()
        self.seq_dim = seq_dim
        self.hidden_size = hidden_size
        self.l1_W = nn.Linear(seq_dim, hidden_size)
        self.l1_U = nn.Linear(hidden_size, hidden_size)
        self.l2_W = nn.Linear(hidden_size, hidden_size)
        self.l2_U = nn.Linear(hidden_size, hidden_size)
        self.l3_W = nn.Linear(hidden_size, hidden_size)
        self.l3_U = nn.Linear(hidden_size, hidden_size)
        self.out1 = nn.Linear(hidden_size, seq_dim)
        self.out2 = nn.Linear(hidden_size, seq_dim)
        self.out3 = nn.Linear(hidden_size, seq_dim)
        self.gate1 = nn.Linear(hidden_size * 2, hidden_size)
        self.gate2 = nn.Linear(hidden_size * 2, hidden_size)
        self.binary = BinaryFunction.apply
        self.hard_gate = hard_gate

    def init_hidden_state(self):
        return to_gpu(torch.zeros(1, self.hidden_size))

    def init_gate_state(self):
        return to_gpu(torch.ones(1, 1))

    def forward(self, x):
        hidden1 = hidden2 = hidden3 = self.init_hidden_state()
        z1 = z2 = self.init_gate_state()
        outputs = []
        for input_t in x.split(1, dim=1):
            hidden1 = F.tanh(self.l1_W(input_t.squeeze(1)) + self.l1_U(hidden1))
            hidden2 = z1 * F.tanh(self.l2_W(hidden1) + self.l2_U(hidden2)) + (1 - z1) * hidden2
            hidden3 = z2 * F.tanh(self.l3_W(hidden2) + self.l3_U(hidden3)) + (1 - z2) * hidden3
            
            z1 = F.sigmoid(self.gate1(torch.cat((hidden1, hidden2), 1))).squeeze(0)
            z2 = F.sigmoid(self.gate2(torch.cat((hidden2, hidden3), 1))).squeeze(0)
            if self.hard_gate:
                z1 = self.binary(z1)
                z2 = self.binary(z2)

            output = self.out1(hidden1) + self.out2(hidden2) + self.out3(hidden3)
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

    def forecast(self, x):
        x = x.unsqueeze(0)  # non-batch
        hidden1 = hidden2 = hidden3 = self.init_hidden_state()
        z1 = z2 = self.init_gate_state()
        outputs = []
        outputs1 = []
        outputs2 = []
        outputs3 = []
        for input_t in x.split(1, dim=1):
            hidden1 = F.tanh(self.l1_W(input_t.squeeze(1)) + self.l1_U(hidden1))
            hidden2 = z1 * F.tanh(self.l2_W(hidden1) + self.l2_U(hidden2)) + (1 - z1) * hidden2
            hidden3 = z2 * F.tanh(self.l3_W(hidden2) + self.l3_U(hidden3)) + (1 - z2) * hidden3
            
            z1 = F.sigmoid(self.gate1(torch.cat((hidden1, hidden2), 1))).squeeze(0)
            z2 = F.sigmoid(self.gate2(torch.cat((hidden2, hidden3), 1))).squeeze(0)
            if self.hard_gate:
                z1 = self.binary(z1)
                z2 = self.binary(z2)

            output1, output2, output3 = self.out1(hidden1), self.out2(hidden2), self.out3(hidden3)
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

    def self_forecast(self, x, step):
        x = x.unsqueeze(0)  # non-batch
        hidden1 = hidden2 = hidden3 = self.init_hidden_state()
        z1 = z2 = self.init_gate_state()
        outputs = []
        for input_t in x.split(1, dim=1):
            hidden1 = F.tanh(self.l1_W(input_t.squeeze(1)) + self.l1_U(hidden1))
            hidden2 = z1 * F.tanh(self.l2_W(hidden1) + self.l2_U(hidden2)) + (1 - z1) * hidden2
            hidden3 = z2 * F.tanh(self.l3_W(hidden2) + self.l3_U(hidden3)) + (1 - z2) * hidden3
            
            z1 = F.sigmoid(self.gate1(torch.cat((hidden1, hidden2), 1))).squeeze(0)
            z2 = F.sigmoid(self.gate2(torch.cat((hidden2, hidden3), 1))).squeeze(0)
            if self.hard_gate:
                z1 = self.binary(z1)
                z2 = self.binary(z2)

            output = self.out1(hidden1) + self.out2(hidden2) + self.out3(hidden3)
            outputs += [output]
        for i in range(step - 1):  # if we should predict the future
            hidden1 = F.tanh(self.l1_W(output) + self.l1_U(hidden1))
            hidden2 = z1 * F.tanh(self.l2_W(hidden1) + self.l2_U(hidden2)) + (1 - z1) * hidden2
            hidden3 = z2 * F.tanh(self.l3_W(hidden2) + self.l3_U(hidden3)) + (1 - z2) * hidden3
            
            z1 = F.sigmoid(self.gate1(torch.cat((hidden1, hidden2), 1))).squeeze(0)
            z2 = F.sigmoid(self.gate2(torch.cat((hidden2, hidden3), 1))).squeeze(0)
            if self.hard_gate:
                z1 = self.binary(z1)
                z2 = self.binary(z2)

            output = self.out1(hidden1) + self.out2(hidden2) + self.out3(hidden3)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs[0]

class ResRNN(nn.Module):
    def __init__(self, seq_dim, hidden_size, hard_gate=False):
        super(ResRNN, self).__init__()
        self.seq_dim = seq_dim
        self.hidden_size = hidden_size

        self.l1_W = nn.Linear(seq_dim, hidden_size)
        self.l1_U = nn.Linear(hidden_size, hidden_size)
        self.l2_W = nn.Linear(hidden_size, hidden_size)
        self.l2_U = nn.Linear(hidden_size, hidden_size)
        self.l3_W = nn.Linear(hidden_size, hidden_size)
        self.l3_U = nn.Linear(hidden_size, hidden_size)
        self.l3_V = nn.Linear(hidden_size, hidden_size)

        self.gate1 = nn.Linear(hidden_size * 2, hidden_size)
        self.gate2 = nn.Linear(hidden_size * 3, hidden_size)

        self.out1 = nn.Linear(hidden_size, seq_dim)
        self.out2 = nn.Linear(hidden_size, seq_dim)

        self.hard_gate = hard_gate
        self.binary = BinaryFunction.apply

    def init_hidden_state(self):
        return to_gpu(torch.zeros(1, self.hidden_size))

    def init_gate_state(self):
        return to_gpu(torch.ones(1, 1))

    def forward(self, x):
        hidden1 = hidden2 = hidden3 = self.init_hidden_state()
        z1 = z2 = self.init_gate_state()
        outputs = []
        for input_t in x.split(1, dim=1):
            hidden1 = F.tanh(self.l1_W(input_t.squeeze(1)) + self.l1_U(hidden1))
            hidden2 = z1 * F.tanh(self.l2_W(hidden1) + self.l2_U(hidden2)) + (1 - z1) * hidden2
            hidden3 = z2 * F.tanh(self.l3_W(hidden2) + self.l3_U(hidden3) + self.l3_V(hidden1)) + (1 - z2) * hidden3

            z1 = F.sigmoid(self.gate1(torch.cat((hidden1, hidden2), 1))).squeeze(0)
            z2 = F.sigmoid(self.gate2(torch.cat((hidden1, hidden2, hidden3), 1))).squeeze(0)
            if self.hard_gate:
                z1 = self.binary(z1)
                z2 = self.binary(z2)
            z1 = z1 * z2

            output = self.out1(hidden2) + self.out2(hidden3)
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

    def forecast(self, x):
        x = x.unsqueeze(0)  # non-batch
        hidden1 = hidden2 = hidden3 = self.init_hidden_state()
        z1 = z2 = self.init_gate_state()
        outputs = []
        outputs1 = []
        outputs2 = []
        for input_t in x.split(1, dim=1):
            hidden1 = F.tanh(self.l1_W(input_t.squeeze(1)) + self.l1_U(hidden1))
            hidden2 = z1 * F.tanh(self.l2_W(hidden1) + self.l2_U(hidden2)) + (1 - z1) * hidden2
            hidden3 = z2 * F.tanh(self.l3_W(hidden2) + self.l3_U(hidden3) + self.l3_V(hidden1)) + (1 - z2) * hidden3

            z1 = F.sigmoid(self.gate1(torch.cat((hidden1, hidden2), 1))).squeeze(0)
            z2 = F.sigmoid(self.gate2(torch.cat((hidden1, hidden2, hidden3), 1))).squeeze(0)
            if self.hard_gate:
                z1 = self.binary(z1)
                z2 = self.binary(z2)
            z1 = z1 * z2

            output1 = self.out1(hidden2)
            output2 = self.out2(hidden3)
            output = output1 + output2

            outputs += [output]
            outputs1 += [output1]
            outputs2 += [output2]
        outputs = torch.stack(outputs, 1).squeeze(2)
        outputs1 = torch.stack(outputs1, 1).squeeze(2)
        outputs2 = torch.stack(outputs2, 1).squeeze(2)
        return outputs[0], outputs1[0], outputs2[0]

    # def self_forecast(self, x, step):
    #     x = x.unsqueeze(0)  # non-batch
    #     hidden1 = hidden2 = hidden3 = self.init_hidden_state()
    #     z1 = z2 = self.init_gate_state()
    #     outputs = []
    #     for input_t in x.split(1, dim=1):
    #         hidden1 = F.tanh(self.l1_W(input_t.squeeze(1)) + self.l1_U(hidden1))
    #         hidden2 = z1 * F.tanh(self.l2_W(hidden1) + self.l2_U(hidden2)) + (1 - z1) * hidden2
    #         hidden3 = z2 * F.tanh(self.l3_W(hidden2) + self.l3_U(hidden3) + self.l3_V(hidden1)) + (1 - z2) * hidden3

    #         z1 = F.sigmoid(self.gate1(torch.cat((hidden1, hidden2), 1))).squeeze(0)
    #         z2 = F.sigmoid(self.gate2(torch.cat((hidden1, hidden2, hidden3), 1))).squeeze(0)
    #         if self.hard_gate:
    #             z1 = self.binary(z1)
    #             z2 = self.binary(z2)
    #         z2 = z1 * z2

    #         output = self.out1(hidden2) + self.out2(hidden3)
    #         outputs += [output]
    #     for i in range(step - 1):  # if we should predict the future
    #         hidden1 = F.tanh(self.l1_W(output) + self.l1_U(hidden1))
    #         hidden2 = z1 * F.tanh(self.l2_W(hidden1) + self.l2_U(hidden2)) + (1 - z1) * hidden2
    #         hidden3 = z2 * F.tanh(self.l3_W(hidden2) + self.l3_U(hidden3) + self.l3_V(hidden1)) + (1 - z2) * hidden3

    #         z1 = F.sigmoid(self.gate1(torch.cat((hidden1, hidden2), 1))).squeeze(0)
    #         z2 = F.sigmoid(self.gate2(torch.cat((hidden1, hidden2, hidden3), 1))).squeeze(0)
    #         if self.hard_gate:
    #             z1 = self.binary(z1)
    #             z2 = self.binary(z2)
    #         z2 = z1 * z2

    #         output = self.out1(hidden2) + self.out2(hidden3)
    #         outputs += [output]
    #     outputs = torch.stack(outputs, 1).squeeze(2)
    #     return outputs[0]


class StackResRNN(nn.Module):
    def __init__(self, seq_dim, hidden_size, hard_gate=False):
        super(StackResRNN, self).__init__()
        self.seq_dim = seq_dim
        self.hidden_size = hidden_size

        self.l1_W = nn.Linear(seq_dim, hidden_size)
        self.l1_U = nn.Linear(hidden_size, hidden_size)
        self.l2_W = nn.Linear(hidden_size, hidden_size)
        self.l2_U = nn.Linear(hidden_size, hidden_size)
        self.l3_W = nn.Linear(hidden_size, hidden_size)
        self.l3_U = nn.Linear(hidden_size, hidden_size)
        self.l4_W = nn.Linear(hidden_size, hidden_size)
        self.l4_U = nn.Linear(hidden_size, hidden_size)
        self.l4_V = nn.Linear(hidden_size, hidden_size)

        self.gate1 = nn.Linear(hidden_size * 2, hidden_size)
        self.gate2 = nn.Linear(hidden_size * 2, hidden_size)
        self.gate3 = nn.Linear(hidden_size * 3, hidden_size)

        self.l3_h2o = nn.Linear(hidden_size, seq_dim)
        self.l4_h2o = nn.Linear(hidden_size, seq_dim)

        self.hard_gate = hard_gate
        self.binary = BinaryFunction.apply

    def init_hidden_state(self):
        return to_gpu(torch.zeros(1, self.hidden_size))

    def init_gate_state(self):
        return to_gpu(torch.ones(1, 1))

    def forward(self, x):
        hidden1 = hidden2 = hidden3 = hidden4 = self.init_hidden_state()
        z1 = z2 = z3 = self.init_gate_state()
        outputs = []
        for input_t in x.split(1, dim=1):
            hidden1 = F.tanh(self.l1_W(input_t.squeeze(1)) + self.l1_U(hidden1))
            hidden2 = z1 * F.tanh(self.l2_W(hidden1) + self.l2_U(hidden2)) + (1 - z1) * hidden2
            hidden3 = z2 * F.tanh(self.l3_W(hidden2) + self.l3_U(hidden3)) + (1 - z2) * hidden3
            hidden4 = z3 * F.tanh(self.l4_W(hidden3) + self.l4_U(hidden4) + self.l4_V(hidden1)) + (1 - z3) * hidden4

            z1 = F.sigmoid(self.gate1(torch.cat((hidden1, hidden2), 1))).squeeze(0)
            z2 = F.sigmoid(self.gate2(torch.cat((hidden2, hidden3), 1))).squeeze(0)
            z3 = F.sigmoid(self.gate3(torch.cat((hidden3, hidden4, hidden1), 1))).squeeze(0)
            if self.hard_gate:
                z1 = self.binary(z1)
                z2 = self.binary(z2)
                z3 = self.binary(z3)
            z2 = z2 * z3
            output = self.l3_h2o(hidden3) + self.l4_h2o(hidden4)
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

    def forecast(self, x):
        x = x.unsqueeze(0)  # non-batch
        hidden1 = hidden2 = hidden3 = hidden4 = self.init_hidden_state()
        z1 = z2 = z3 = self.init_gate_state()
        outputs = []
        outputs1 = []
        outputs2 = []
        for input_t in x.split(1, dim=1):
            hidden1 = F.tanh(self.l1_W(input_t.squeeze(1)) + self.l1_U(hidden1))
            hidden2 = z1 * F.tanh(self.l2_W(hidden1) + self.l2_U(hidden2)) + (1 - z1) * hidden2
            hidden3 = z2 * F.tanh(self.l3_W(hidden2) + self.l3_U(hidden3)) + (1 - z2) * hidden3
            hidden4 = z3 * F.tanh(self.l4_W(hidden3) + self.l4_U(hidden4) + self.l4_V(hidden1)) + (1 - z3) * hidden4

            z1 = F.sigmoid(self.gate1(torch.cat((hidden1, hidden2), 1))).squeeze(0)
            z2 = F.sigmoid(self.gate2(torch.cat((hidden2, hidden3), 1))).squeeze(0)
            z3 = F.sigmoid(self.gate3(torch.cat((hidden3, hidden4, hidden1), 1))).squeeze(0)

            if self.hard_gate:
                z1 = self.binary(z1)
                z2 = self.binary(z2)
                z3 = self.binary(z3)
            z2 = z2 * z3
            output1 = self.l3_h2o(hidden3)
            output2 = self.l4_h2o(hidden4)
            output = output1 + output2

            outputs += [output]
            outputs1 += [output1]
            outputs2 += [output2]

        outputs = torch.stack(outputs, 1).squeeze(2)
        outputs1 = torch.stack(outputs1, 1).squeeze(2)
        outputs2 = torch.stack(outputs2, 1).squeeze(2)

        return outputs[0], outputs1[0], outputs2[0]