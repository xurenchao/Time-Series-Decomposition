import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from tool import to_gpu
import types


class DRNN(nn.Module):

    def __init__(self, n_input, n_hidden, n_layers, n_output, dropout=0, cell_type='GRU', batch_first=True):

        super(DRNN, self).__init__()
        self.W = n_input // 24
        # self.dilations = [2 ** i for i in range(n_layers)]
        # self.dilations = [1, 2, 7, 14, 24][0:n_layers]
        self.dilations = [1, 2, 4, 2][0:n_layers]
        self.n_output = n_output
        self.cell_type = cell_type
        self.batch_first = batch_first
        self.h2o1 = nn.Linear(n_hidden, n_output)
        self.h2o2 = nn.Linear(n_hidden, n_output)
        self.h2o3 = nn.Linear(n_hidden, n_output)

        layers = []
        if self.cell_type == "GRU":
            cell = nn.GRU
        elif self.cell_type == "RNN":
            cell = nn.RNN
        elif self.cell_type == "LSTM":
            cell = nn.LSTM
        else:
            raise NotImplementedError

        for i in range(n_layers):
            if i == 0:
                c = cell(n_input, n_hidden, dropout=dropout)
            else:
                c = cell(n_hidden, n_hidden, dropout=dropout)
            layers.append(c)
        self.cells = nn.Sequential(*layers)

    def forward(self, x):
        b, d, h = x.shape
        hiddens = self.drnn_run(x.reshape(b, 1, d * h))
        output = F.tanh(self.h2o1(hiddens[0])) + \
            F.tanh(self.h2o2(hiddens[2])) + F.tanh(self.h2o2(hiddens[-1]))

        return output[0]

    def forecast(self, x):
        # x = x.unsqueeze(0)
        d, _ = x.shape
        outputs = []
        outputs1 = []
        outputs2 = []
        outputs3 = []
        for i in range(d - self.W + 1):
            hiddens = self.drnn_run(x[i:i + self.W, :].reshape(1, 1, self.W * 24))
            output1 = F.tanh(self.h2o1(hiddens[0]))
            output2 = F.tanh(self.h2o2(hiddens[2]))
            output3 = F.tanh(self.h2o3(hiddens[-1]))
            outputs += [output1 + output2]
            outputs1 += [output1]
            outputs2 += [output2]
            outputs3 += [output3]

        outputs = torch.stack(outputs, 1).squeeze(2)[0]
        outputs1 = torch.stack(outputs1, 1).squeeze(2)[0]
        outputs2 = torch.stack(outputs2, 1).squeeze(2)[0]
        outputs3 = torch.stack(outputs3, 1).squeeze(2)[0]
        return outputs, outputs1, outputs2, outputs3

    def drnn_run(self, inputs, hidden=None):
        if self.batch_first:
            inputs = inputs.transpose(0, 1)
        outputs = []
        for i, (cell, dilation) in enumerate(zip(self.cells, self.dilations)):
            if hidden is None:
                inputs, _ = self.drnn_layer(cell, inputs, dilation)
            else:
                inputs, hidden[i] = self.drnn_layer(cell, inputs, dilation, hidden[i])

            outputs += [inputs]
        return outputs

    def drnn_layer(self, cell, inputs, rate, hidden=None):

        n_steps = len(inputs)
        batch_size = inputs[0].size(0)
        hidden_size = cell.hidden_size

        inputs, dilated_steps = self._pad_inputs(inputs, n_steps, rate)
        dilated_inputs = self._prepare_inputs(inputs, rate)

        if hidden is None:
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size)
        else:
            hidden = self._prepare_inputs(hidden, rate)
            dilated_outputs, hidden = self._apply_cell(
                dilated_inputs, cell, batch_size, rate, hidden_size, hidden=hidden)

        splitted_outputs = self._split_outputs(dilated_outputs, rate)
        outputs = self._unpad_outputs(splitted_outputs, n_steps)

        return outputs, hidden

    def _apply_cell(self, dilated_inputs, cell, batch_size, rate, hidden_size, hidden=None):
        if hidden is None:
            if self.cell_type == 'LSTM':
                c, m = self.init_hidden(batch_size * rate, hidden_size)
                hidden = (c.unsqueeze(0), m.unsqueeze(0))
            else:
                hidden = self.init_hidden(batch_size * rate, hidden_size).unsqueeze(0)

        dilated_outputs, hidden = cell(dilated_inputs, hidden)

        return dilated_outputs, hidden

    def _unpad_outputs(self, splitted_outputs, n_steps):
        return splitted_outputs[:n_steps]

    def _split_outputs(self, dilated_outputs, rate):
        batchsize = dilated_outputs.size(1) // rate

        blocks = [dilated_outputs[:, i * batchsize: (i + 1) * batchsize, :] for i in range(rate)]

        interleaved = torch.stack((blocks)).transpose(1, 0).contiguous()
        interleaved = interleaved.view(dilated_outputs.size(0) * rate,
                                       batchsize,
                                       dilated_outputs.size(2))
        return interleaved

    def _pad_inputs(self, inputs, n_steps, rate):
        iseven = (n_steps % rate) == 0

        if not iseven:
            dilated_steps = n_steps // rate + 1

            zeros_ = torch.zeros(dilated_steps * rate - inputs.size(0),
                                 inputs.size(1),
                                 inputs.size(2))
            zeros_ = to_gpu(zeros_)

            inputs = torch.cat((inputs, zeros_))
        else:
            dilated_steps = n_steps // rate

        return inputs, dilated_steps

    def _prepare_inputs(self, inputs, rate):
        dilated_inputs = torch.cat([inputs[j::rate, :, :] for j in range(rate)], 1)
        return dilated_inputs

    def init_hidden(self, batch_size, hidden_dim):
        hidden = to_gpu(torch.zeros(batch_size, hidden_dim))

        if self.cell_type == "LSTM":
            memory = to_gpu(torch.zeros(batch_size, hidden_dim))
            return (hidden, memory)
        else:
            return hidden



class ResDRNN(nn.Module):
    def __init__(self, n_input, n_hidden, dilation, n_output, dropout=0, cell_type='GRU', batch_first=True):
        super(ResDRNN, self).__init__()
        self.W = n_input // 24
        self.dilations = dilation[0:3]
        self.d = dilation[-1]
        self.n_output = n_output
        self.cell_type = cell_type
        self.batch_first = batch_first
        self.h2o1 = nn.Linear(n_hidden, n_output)
        self.h2o2 = nn.Linear(n_hidden, n_output)
        self.h2o3 = nn.Linear(n_hidden, n_output)

        layers = []
        if self.cell_type == "GRU":
            cell = nn.GRU
        elif self.cell_type == "RNN":
            cell = nn.RNN
        elif self.cell_type == "LSTM":
            cell = nn.LSTM
        else:
            raise NotImplementedError

        for i in range(3):
            if i == 0:
                c = cell(n_input, n_hidden, dropout=dropout)
            else:
                c = cell(n_hidden, n_hidden, dropout=dropout)
            layers.append(c)
        self.cells = nn.Sequential(*layers)
        self.topcell = cell(n_hidden * 2, n_hidden, dropout=dropout)

    def init_bias(self, size):
        return to_gpu(torch.zeros(size))

    def forward(self, x):
        b, d, h = x.shape
        b_o1 = b_o2 = b_o3 = self.init_bias(self.n_output)
        hiddens = self.drnn_run(x.reshape(b, 1, d * h))
        hcat = torch.cat((hiddens[0], hiddens[2]), 2)
        hidden_, _ = self.drnn_layer(self.topcell, hcat, self.d)
        hiddens += [hidden_]

        output = F.tanh(self.h2o1(hiddens[1]) + b_o1) + \
            F.tanh(self.h2o2(hiddens[2]) + b_o2) + F.tanh(self.h2o3(hiddens[-1]) + b_o3)

        return output[0]

    def forecast(self, x):
        # x = x.unsqueeze(0)
        d, _ = x.shape
        b_o1 = b_o2 = b_o3 = self.init_bias(self.n_output)
        outputs = []
        outputs1 = []
        outputs2 = []
        outputs3 = []
        for i in range(d - self.W + 1):
            hiddens = self.drnn_run(x[i:i + self.W, :].reshape(1, 1, self.W * 24))
            hcat = torch.cat((hiddens[0], hiddens[2]), 2)
            hidden_, _ = self.drnn_layer(self.topcell, hcat, self.d)
            hiddens += [hidden_]
            output1 = F.tanh(self.h2o1(hiddens[1]) + b_o1)
            output2 = F.tanh(self.h2o2(hiddens[2]) + b_o2)
            output3 = F.tanh(self.h2o3(hiddens[-1]) + b_o3)
            outputs += [output1 + output2 + output3]
            outputs1 += [output1]
            outputs2 += [output2]
            outputs3 += [output3]

        outputs = torch.stack(outputs, 1).squeeze(2)[0]
        outputs1 = torch.stack(outputs1, 1).squeeze(2)[0]
        outputs2 = torch.stack(outputs2, 1).squeeze(2)[0]
        outputs3 = torch.stack(outputs3, 1).squeeze(2)[0]
        return outputs, outputs1, outputs2, outputs3

    def drnn_run(self, inputs, hidden=None):
        if self.batch_first:
            inputs = inputs.transpose(0, 1)
        outputs = []
        for i, (cell, dilation) in enumerate(zip(self.cells, self.dilations)):
            if hidden is None:
                inputs, _ = self.drnn_layer(cell, inputs, dilation)
            else:
                inputs, hidden[i] = self.drnn_layer(cell, inputs, dilation, hidden[i])

            outputs += [inputs]
        return outputs

    def drnn_layer(self, cell, inputs, rate, hidden=None):

        n_steps = len(inputs)
        batch_size = inputs[0].size(0)
        hidden_size = cell.hidden_size

        inputs, dilated_steps = self._pad_inputs(inputs, n_steps, rate)
        dilated_inputs = self._prepare_inputs(inputs, rate)

        if hidden is None:
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size)
        else:
            hidden = self._prepare_inputs(hidden, rate)
            dilated_outputs, hidden = self._apply_cell(
                dilated_inputs, cell, batch_size, rate, hidden_size, hidden=hidden)

        splitted_outputs = self._split_outputs(dilated_outputs, rate)
        outputs = self._unpad_outputs(splitted_outputs, n_steps)

        return outputs, hidden

    def _apply_cell(self, dilated_inputs, cell, batch_size, rate, hidden_size, hidden=None):
        if hidden is None:
            if self.cell_type == 'LSTM':
                c, m = self.init_hidden(batch_size * rate, hidden_size)
                hidden = (c.unsqueeze(0), m.unsqueeze(0))
            else:
                hidden = self.init_hidden(batch_size * rate, hidden_size).unsqueeze(0)

        dilated_outputs, hidden = cell(dilated_inputs, hidden)

        return dilated_outputs, hidden

    def _unpad_outputs(self, splitted_outputs, n_steps):
        return splitted_outputs[:n_steps]

    def _split_outputs(self, dilated_outputs, rate):
        batchsize = dilated_outputs.size(1) // rate

        blocks = [dilated_outputs[:, i * batchsize: (i + 1) * batchsize, :] for i in range(rate)]

        interleaved = torch.stack((blocks)).transpose(1, 0).contiguous()
        interleaved = interleaved.view(dilated_outputs.size(0) * rate,
                                       batchsize,
                                       dilated_outputs.size(2))
        return interleaved

    def _pad_inputs(self, inputs, n_steps, rate):
        iseven = (n_steps % rate) == 0

        if not iseven:
            dilated_steps = n_steps // rate + 1

            zeros_ = torch.zeros(dilated_steps * rate - inputs.size(0),
                                 inputs.size(1),
                                 inputs.size(2))
            zeros_ = to_gpu(zeros_)

            inputs = torch.cat((inputs, zeros_))
        else:
            dilated_steps = n_steps // rate

        return inputs, dilated_steps

    def _prepare_inputs(self, inputs, rate):
        dilated_inputs = torch.cat([inputs[j::rate, :, :] for j in range(rate)], 1)
        return dilated_inputs

    def init_hidden(self, batch_size, hidden_dim):
        hidden = to_gpu(torch.zeros(batch_size, hidden_dim))

        if self.cell_type == "LSTM":
            memory = to_gpu(torch.zeros(batch_size, hidden_dim))
            return (hidden, memory)
        else:
            return hidden
