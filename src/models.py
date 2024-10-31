"""Wrappers for `RNNCell` and `LSTMCell` modules from `torch`."""
import torch
from torch import nn

class RNN(nn.Module):
    def __init__(self, n_inp, n_hid, n_out):
        super(RNN, self).__init__()
        self.n_hid = n_hid
        self.cell = nn.RNNCell(n_inp, n_hid, bias=True, nonlinearity='relu')
        self.readout = nn.Linear(n_hid, n_out)
        self.state_initial = torch.ones(1, n_hid)
        self.state = self.state_initial

    def forward(self, x):
        self.state = self.cell(x, self.state)
        output = self.readout(self.state)
        
        return torch.squeeze(output)
    
    def reset(self,):
        self.state = self.state_initial
        
class LSTM(nn.Module):
    def __init__(self, n_inp, n_hid, n_out):
        super(LSTM, self).__init__()
        self.n_hid = n_hid
        self.cell = nn.LSTMCell(n_inp, n_hid, bias=True)
        self.readout = nn.Linear(n_hid, n_out)
        self.state_initial = torch.ones(1, n_hid)
        self.state = self.state_initial
        self.memory_initial = torch.ones(1, n_hid)
        self.memory = self.memory_initial

    def forward(self, x):
        self.state, self.memory = self.cell(x, (self.state, self.memory))
        output = self.readout(self.state)
        
        return torch.squeeze(output)
    
    def reset(self,):
        self.state = self.state_initial
        self.memory = self.memory_initial