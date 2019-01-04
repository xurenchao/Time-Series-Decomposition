import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from torch.nn import Parameter
from tool import to_gpu
import math


def hard_sigm(a, x):
    temp = torch.div(torch.add(torch.mul(x, a), 1), 2.0)
    output = torch.clamp(temp, min=0, max=1)
    return output


# class bound(Function):
#     def forward(ctx, x):
#         # forward : x -> output
#         ctx.save_for_backward(x)
#         output = x > 0.5
#         return output.float()

#     def backward(ctx, output_grad):
#         # backward: output_grad -> x_grad
#         x = ctx.saved_tensors
#         x_grad = None

#         if ctx.needs_input_grad[0]:
#             x_grad = output_grad.clone()

#         return x_grad

class bound(Function):
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


class HM_LSTMCell(nn.Module):
    def __init__(self, bottom_size, hidden_size, top_size, a, last_layer):
        super(HM_LSTMCell, self).__init__()
        self.bottom_size = bottom_size
        self.hidden_size = hidden_size
        self.top_size = top_size
        self.a = a
        self.last_layer = last_layer
        self.binary = bound.apply
        '''
        U_11 means the state transition parameters from layer l (current layer) to layer l
        U_21 means the state transition parameters from layer l+1 (top layer) to layer l
        W_01 means the state transition parameters from layer l-1 (bottom layer) to layer l
        '''
        self.U_11 = Parameter(torch.FloatTensor(self.hidden_size, 4 * self.hidden_size + 1))
        if not self.last_layer:
            self.U_21 = Parameter(torch.FloatTensor(self.top_size, 4 * self.hidden_size + 1))
        self.W_01 = Parameter(torch.FloatTensor(self.bottom_size, 4 * self.hidden_size + 1))
        self.bias = Parameter(torch.FloatTensor(4 * self.hidden_size + 1))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for par in self.parameters():
            par.data.uniform_(-stdv, stdv)

    def forward(self, c, h_bottom, h, h_top, z, z_bottom):
        # h_bottom.size = bottom_size * batch_size
        # s_recur = torch.mm(self.W_01, h_bottom) # 是不是写错了
        s_recur = torch.mm(h, self.U_11)
        if not self.last_layer:
            s_topdown_ = torch.mm(h_top, self.U_21)
            s_topdown = z.expand_as(s_topdown_) * s_topdown_
        else:
            s_topdown = to_gpu(torch.zeros(s_recur.size()))
        # s_bottomup_ = torch.mm(self.U_11, h) # 是不是写错了
        s_bottomup_ = torch.mm(h_bottom, self.W_01)
        s_bottomup = z_bottom.expand_as(s_bottomup_) * s_bottomup_

        # print(self.bias.unsqueeze(1).shape, s_recur.shape)
        # print(s_recur.is_cuda, s_topdown.is_cuda, s_bottomup.is_cuda, self.bias.unsqueeze(0).expand_as(s_recur).is_cuda)
        f_s = s_recur + s_topdown + s_bottomup + self.bias.unsqueeze(0).expand_as(s_recur)
        # f_s.size = (4 * hidden_size + 1) * batch_size
        f = F.sigmoid(f_s[:, 0:self.hidden_size])  # hidden_size * batch_size
        i = F.sigmoid(f_s[:, self.hidden_size:self.hidden_size * 2])
        o = F.sigmoid(f_s[:, self.hidden_size * 2:self.hidden_size * 3])
        g = F.tanh(f_s[:, self.hidden_size * 3:self.hidden_size * 4])
        z_hat = hard_sigm(self.a, f_s[:, self.hidden_size * 4:self.hidden_size * 4 + 1])

        one = to_gpu(torch.ones(f.size()))
        z = z.expand_as(f)
        z_bottom = z_bottom.expand_as(f)

        c_new = z * (i * g) + (one - z) * (one - z_bottom) * c + (one - z) * z_bottom * (f * c + i * g)
        h_new = (one - z) * (one - z_bottom) * h + (z + (one - z) * z_bottom) * o * F.tanh(c_new)

        z_new = self.binary(z_hat)

        return h_new, c_new, z_new


class HM_LSTM(nn.Module):
    def __init__(self, seq_dim, hidden_size):
        super(HM_LSTM, self).__init__()
        self.seq_dim = seq_dim
        self.hidden_size = hidden_size

        self.cell_1 = HM_LSTMCell(seq_dim, hidden_size, hidden_size, 1.0, False)
        self.cell_2 = HM_LSTMCell(hidden_size, hidden_size, None, 1.0, True)
        self.l1_h2o = nn.Linear(hidden_size, seq_dim)
        self.l2_h2o = nn.Linear(hidden_size, seq_dim)

    def init_hidden(self):
        h_t1 = to_gpu(torch.zeros(1, self.hidden_size))
        c_t1 = to_gpu(torch.zeros(1, self.hidden_size))
        z_t1 = to_gpu(torch.zeros(1, 1))
        h_t2 = to_gpu(torch.zeros(1, self.hidden_size))
        c_t2 = to_gpu(torch.zeros(1, self.hidden_size))
        z_t2 = to_gpu(torch.zeros(1, 1))

        hidden = (h_t1, c_t1, z_t1, h_t2, c_t2, z_t2)
        return hidden

    def forward(self, x):
        (h_t1, c_t1, z_t1, h_t2, c_t2, z_t2) = self.init_hidden()
        z_one = to_gpu(torch.ones(1, 1))

        outputs = []
        for input_t in x.split(1, dim=1):
            h_t1, c_t1, z_t1 = self.cell_1(c_t1, input_t.squeeze(1), h_t1, h_t2, z_t1, z_one)
            h_t2, c_t2, z_t2 = self.cell_2(c_t2, h_t1, h_t2, None, z_t2, z_t1)
            output = self.l1_h2o(h_t1) + self.l2_h2o(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

    def forecast(self, x):
        x = x.unsqueeze(0)  # non-batch

        (h_t1, c_t1, z_t1, h_t2, c_t2, z_t2) = self.init_hidden()
        z_one = to_gpu(torch.ones(1, 1))

        outputs = []
        outputs1 = []
        outputs2 = []
        for input_t in x.split(1, dim=1):
            h_t1, c_t1, z_t1 = self.cell_1(c_t1, input_t.squeeze(1), h_t1, h_t2, z_t1, z_one)
            h_t2, c_t2, z_t2 = self.cell_2(c_t2, h_t1, h_t2, None, z_t2, z_t1)
            output1, output2 = self.l1_h2o(h_t1), self.l2_h2o(h_t2)
            output = output1 + output2
            outputs += [output]
            outputs1 += [output1]
            outputs2 += [output2]
        outputs = torch.stack(outputs, 1).squeeze(2)
        outputs1 = torch.stack(outputs1, 1).squeeze(2)
        outputs2 = torch.stack(outputs2, 1).squeeze(2)
        return outputs[0], outputs1[0], outputs2[0]

    def self_forecast(self, x, step):
        x = x.unsqueeze(0)  # non-batch

        (h_t1, c_t1, z_t1, h_t2, c_t2, z_t2) = self.init_hidden()
        z_one = to_gpu(torch.ones(1, 1))

        outputs = []
        for input_t in x.split(1, dim=1):
            h_t1, c_t1, z_t1 = self.cell_1(c_t1, input_t.squeeze(1), h_t1, h_t2, z_t1, z_one)
            h_t2, c_t2, z_t2 = self.cell_2(c_t2, h_t1, h_t2, None, z_t2, z_t1)
            output = self.l1_h2o(h_t1) + self.l2_h2o(h_t2)
        outputs += [output]
        for i in range(step - 1):  # if we should predict the future
            h_t1, c_t1, z_t1 = self.cell_1(c_t1, output, h_t1, h_t2, z_t1, z_one)
            h_t2, c_t2, z_t2 = self.cell_2(c_t2, h_t1, h_t2, None, z_t2, z_t1)
            output = self.l1_h2o(h_t1) + self.l2_h2o(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)

        return outputs[0]


class HM_Net(nn.Module):
    def __init__(self, a, size_list, dict_size, embed_size):
        super(HM_Net, self).__init__()
        self.dict_size = dict_size
        self.size_list = size_list
        self.drop = nn.Dropout(p=0.5)
        self.embed_in = nn.Embedding(dict_size, embed_size)
        self.HM_LSTM = HM_LSTM(a, embed_size, size_list)
        self.weight = nn.Linear(size_list[0]+size_list[1], 2)
        self.embed_out1 = nn.Linear(size_list[0], dict_size)
        self.embed_out2 = nn.Linear(size_list[1], dict_size)
        self.relu = nn.ReLU()
        # self.logsoftmax = nn.LogSoftmax()
        # self.loss = masked_NLLLoss()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs, target, hidden):
        # inputs : batch_size * time_steps
        # mask : batch_size * time_steps

        emb = self.embed_in(Variable(inputs, volatile=not self.training))  # batch_size * time_steps * embed_size
        emb = self.drop(emb)
        h_1, h_2, z_1, z_2, hidden = self.HM_LSTM(emb, hidden)  # batch_size * time_steps * hidden_size

        # mask = Variable(mask, requires_grad=False)
        # batch_loss = Variable(torch.zeros(batch_size).cuda())

        h_1 = self.drop(h_1)  # batch_size * time_steps * hidden_size
        h_2 = self.drop(h_2)
        h = torch.cat((h_1, h_2), 2)

        g = F.sigmoid(self.weight(h.view(h.size(0)*h.size(1), h.size(2))))
        g_1 = g[:, 0:1]  # batch_size * time_steps, 1
        g_2 = g[:, 1:2]

        h_e1 = g_1.expand(g_1.size(0), self.dict_size)*self.embed_out1(h_1.view(h_1.size(0)*h_1.size(1), h_2.size(2)))
        h_e2 = g_2.expand(g_2.size(0), self.dict_size)*self.embed_out2(h_2.view(h_2.size(0)*h_2.size(1), h_2.size(2)))

        h_e = self.relu(h_e1 + h_e2)  # batch_size*time_steps, hidden_size
        batch_loss = self.loss(h_e, Variable(target))

        return batch_loss, hidden, z_1, z_2

    def init_hidden(self, batch_size):
        h_t1 = torch.zeros(self.size_list[0], batch_size)
        c_t1 = torch.zeros(self.size_list[0], batch_size)
        z_t1 = torch.zeros(1, batch_size)
        h_t2 = torch.zeros(self.size_list[1], batch_size)
        c_t2 = torch.zeros(self.size_list[1], batch_size)
        z_t2 = torch.zeros(1, batch_size)

        hidden = (h_t1, c_t1, z_t1, h_t2, c_t2, z_t2)
        return hidden