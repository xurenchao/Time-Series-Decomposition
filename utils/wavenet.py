import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(CausalConv1d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, groups, bias)

    def forward(self, inputs):
        outputs = super(CausalConv1d, self).forward(inputs)
        return outputs[:, :, :-1]


class DilatedConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super(DilatedConv1d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                            padding, dilation, groups, bias)

    def forward(self, inputs):
        outputs = super(DilatedConv1d, self).forward(inputs)
        return outputs


class ResidualBlock(nn.Module):
    def __init__(self, res_channels, skip_channels, dilation):
        super(ResidualBlock, self).__init__()
        self.filter_conv = DilatedConv1d(in_channels=res_channels, out_channels=res_channels, dilation=dilation)
        self.gate_conv = DilatedConv1d(in_channels=res_channels, out_channels=res_channels, dilation=dilation)
        self.skip_conv = nn.Conv1d(in_channels=res_channels, out_channels=skip_channels, kernel_size=1)
        self.residual_conv = nn.Conv1d(in_channels=res_channels, out_channels=res_channels, kernel_size=1)

    def forward(self, inputs):
        sigmoid_out = F.sigmoid(self.gate_conv(inputs))
        tahn_out = F.tanh(self.filter_conv(inputs))
        output = sigmoid_out * tahn_out
        #
        skip_out = self.skip_conv(output)
        res_out = self.residual_conv(output)
        res_out = res_out + inputs[:, :, -res_out.size(2):]
        # res
        return res_out, skip_out


class WaveNet(nn.Module):
    def __init__(self, in_depth=1, res_channels=16, skip_channels=32, dilation_depth=9, n_repeat=1):
        super(WaveNet, self).__init__()
        self.dilations = [2**i for i in range(dilation_depth)] * n_repeat
        self.main = nn.ModuleList([ResidualBlock(res_channels, skip_channels, dilation) for dilation in self.dilations])
        self.pre_conv = nn.Conv1d(in_channels=in_depth, out_channels=res_channels, kernel_size=1)
        self.post = nn.Sequential(nn.ReLU(),
                                  nn.Conv1d(skip_channels, res_channels, 1),
                                  nn.ReLU(),
                                  nn.Conv1d(res_channels, in_depth, 1))

    def forward(self, inputs):
        output = self.preprocess(inputs)
        skip_connections = []
        outputs = []

        for layer in self.main:
            output, skip = layer(output)
            skip_connections.append(skip)

        print(output.size(2))
        output = sum([s[:, :, -output.size(2):] for s in skip_connections])
        print(output.shape)
        output = self.post(output)

        return output

    # def forward(self, x):
    #     hidden1 = hidden2 = self.init_hidden_state()
    #     u = self.init_gate_state()
    #     outputs = []
    #     for input_t in x.split(1, dim=1):
    #         hidden1 = F.tanh(self.l1_W(input_t.squeeze(1)) + self.l1_U(hidden1))
    #         hidden2 = u * F.tanh(self.l2_W(hidden1) + self.l2_U(hidden2)) + (1 - u) * hidden2
    #         output = self.out1(hidden1) + self.out2(hidden2)
    #         u = self.get_gate_state(hidden1, hidden2)
    #         outputs += [output]

    #     outputs = torch.stack(outputs, 1).squeeze(2)
    #     return outputs

    def preprocess(self, inputs):
        b, d, h = inputs.shape
        x = inputs.reshape(b, 1, d * h)
        out = self.pre_conv(x)
        return out
