import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from tool import to_gpu


class BasicRNN(nn.Module):  # without state smooth
    def __init__(self, seq_dim, hidden_size):
        super(BasicRNN, self).__init__()
        self.seq_dim = seq_dim
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(seq_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, seq_dim)

    def init_hidden_state(self):
        return to_gpu(torch.zeros(1, self.hidden_size))

    def forward(self, x):
        hidden = self.init_hidden_state()
        outputs = []
        for input_t in x.split(1, dim=1):
            hidden = F.tanh(self.fc1(input_t.squeeze(1)) + self.fc2(hidden))
            output = self.fc3(hidden)
            outputs += [output]
            
        outputs = torch.stack(outputs, 1).squeeze(2)  # 不理解
        return outputs

    def forecast(self, x):
        x = x.unsqueeze(0)  # non-batch
        hidden = self.init_hidden_state()
        outputs = []
        for input_t in x.split(1, dim=1):
            hidden = F.tanh(self.fc1(input_t.squeeze(1)) + self.fc2(hidden))
            output = self.fc3(hidden)
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)  # 不理解
        return outputs[0]
    
    def self_forecast(self, x, step):
        x = x.unsqueeze(0)  # non-batch
        hidden = self.init_hidden_state()
        outputs = []
        for input_t in x.split(1, dim=1):
            hidden = F.tanh(self.fc1(input_t.squeeze(1)) + self.fc2(hidden))
            output = self.fc3(hidden)
        outputs += [output]
        for i in range(step - 1):  # if we should predict the future
            hidden = F.tanh(self.fc1(input_t.squeeze(1)) + self.fc2(hidden))
            output = self.fc3(hidden)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs[0]


class RNNwss(nn.Module):  # with state smooth
    def __init__(self, hidden_size=64, seq_dim=24, num_layers=2, nonlinearity='tanh',
                 dropout=0, bidirectional='False', state_momentum=0.9, cached_step=5000):
        super(RNNwss, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size=seq_dim, hidden_size=hidden_size, num_layers=num_layers,
                          nonlinearity=nonlinearity, dropout=dropout, bidirectional=bidirectional)
        self.fc = nn.Linear(in_features=hidden_size, out_features=seq_dim)
        self.register_buffer('cached_state',
                        torch.zeros((cached_step, hidden_size)))
        self.state_momentum = state_momentum

    def address_state(self, idx):
        h_t = self.cached_state.index_select(dim=0, index=idx.data)
        return to_gpu(h_t)

    def update_state(self, idx, h_t):
        momentum = self.state_momentum
        self.cached_state[idx.data, :] = momentum * \
            self.cached_state[idx.data, :] + (1 - momentum) * h_t.data

    def forward(self, x, idx):
        h_t = self.address_state(idx)
        outputs = []
        for input_t in x.split(1, dim=1):
            h_t = self.rnn(input_t.squeeze(dim=1), h_t)
            output = self.fc(h_t)
            outputs += [output]
            idx = idx.add_(1)
            self.update_state(idx, h_t)
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs
