'''
Author: jianzhnie
Date: 2021-12-23 15:26:52
LastEditTime: 2022-01-13 12:24:47
LastEditors: jianzhnie
Description:

'''

import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init

from .function import GRUCell as grucell
from .rnn import RNNCellBase


class NaiveCustomGRU(nn.Module):
    """
    A custom implementation of a GRU (Gated Recurrent Unit) layer.

    - 𝐑𝑡=𝜎(𝐗𝑡𝐖𝑥𝑟+𝐇𝑡−1𝐖ℎ𝑟+𝐛𝑟),
    - 𝐙𝑡=𝜎(𝐗𝑡𝐖𝑥𝑧+𝐇𝑡−1𝐖ℎ𝑧+𝐛𝑧),
    - 𝐇̃𝑡=tanh(𝐗𝑡𝐖𝑥ℎ+(𝐑𝑡⊙𝐇𝑡−1)𝐖ℎℎ+𝐛ℎ)
    - 𝐇𝑡=𝐙𝑡⊙𝐇𝑡−1+(1−𝐙𝑡)⊙𝐇̃ 𝑡.

    
    Args:
        input_size (int): The number of expected features in the input.
        hidden_size (int): The number of features in the hidden state.

    Attributes:
        input_size (int): The number of expected features in the input.
        hidden_size (int): The number of features in the hidden state.
        W_xz (nn.Parameter): Weight matrix for the update gate.
        W_hz (nn.Parameter): Weight matrix for the hidden-to-hidden update gate.
        b_z (nn.Parameter): Bias for the update gate.
        W_xr (nn.Parameter): Weight matrix for the reset gate.
        W_hr (nn.Parameter): Weight matrix for the hidden-to-hidden reset gate.
        b_r (nn.Parameter): Bias for the reset gate.
        W_xh (nn.Parameter): Weight matrix for the candidate hidden state.
        W_hh (nn.Parameter): Weight matrix for the hidden-to-hidden candidate hidden state.
        b_h (nn.Parameter): Bias for the candidate hidden state.
        W_hq (nn.Parameter): Weight matrix for the hidden-to-output connections.
        b_q (nn.Parameter): Bias for the output.

    Reference:
        https://d2l.ai
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Update gate parameters (z_gate)
        self.W_xz = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hz = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_z = nn.Parameter(torch.Tensor(hidden_size))

        # Reset gate parameters (r_gate)
        self.W_xr = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hr = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_r = nn.Parameter(torch.Tensor(hidden_size))

        # Candidate hidden state parameters (h_tilda)
        self.W_xh = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_h = nn.Parameter(torch.Tensor(hidden_size))

        # Output parameters
        self.W_hq = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_q = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize weights and biases with uniform random values.

        Weight initialization follows the Xavier initialization scheme.

        Returns:
            None
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, hx):
        """
        Forward pass of the GRU layer.

        Args:
            input (torch.Tensor): The input tensor of shape (batch_size, input_size).
            hx (torch.Tensor): The initial hidden state tensor of shape (batch_size, hidden_size).

        Returns:
            out (torch.Tensor): The output tensor of shape (batch_size, hidden_size).
            hy (torch.Tensor): The final hidden state tensor of shape (batch_size, hidden_size).
        """
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size)
        
        z_gate = torch.sigmoid(torch.mm(input, self.W_xz) + torch.mm(hx, self.W_hz) + self.b_z)
        r_gate = torch.sigmoid(torch.mm(input, self.W_xr) + torch.mm(hx, self.W_hr) + self.b_r)
        
        # Candidate hidden state
        h_tilda = torch.tanh(torch.mm(input, self.W_xh) + torch.mm(r_gate * hx, self.W_hh) + self.b_h)
        
        # Hidden state
        hy = z_gate * hx + (1 - z_gate) * h_tilda
        out = torch.mm(hy, self.W_hq) + self.b_q
        
        return out, hy


class GRUCell(RNNCellBase):
    r"""A gated recurrent unit (GRU) cell

    .. math::

        \begin{array}{ll}
        r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
        z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
        n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
        h' = (1 - z) * n + z * h
        \end{array}

    where :math:`\sigma` is the sigmoid function.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        bias: If `False`, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: `True`

    Inputs: input, hidden
        - **input** of shape `(batch, input_size)`: tensor containing input features
        - **hidden** of shape `(batch, hidden_size)`: tensor containing the initial hidden
          state for each element in the batch.
          Defaults to zero if not provided.

    Outputs: h'
        - **h'** of shape `(batch, hidden_size)`: tensor containing the next hidden state
          for each element in the batch

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(3*hidden_size x input_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(3*hidden_size x hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(3*hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(3*hidden_size)`

    Examples::

        >>> rnn = nn.GRUCell(10, 20)
        >>> input = torch.randn(6, 3, 10)
        >>> hx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
                hx = rnn(input[i], hx)
                output.append(hx)
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_size,
                                                   input_size))
        self.weight_hh = nn.Parameter(
            torch.Tensor(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None):
        self.check_forward_input(input)
        if hx is None:
            hx = input.new_zeros(input.size(0),
                                 self.hidden_size,
                                 requires_grad=False)
        self.check_forward_hidden(input, hx)
        return grucell(
            input,
            hx,
            self.weight_ih,
            self.weight_hh,
            self.bias_ih,
            self.bias_hh,
        )


class GRUBase(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.input2hidden = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.hidden2hidden = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, input, hx=None):
        """
        Inputs:
              input: of shape (batch_size, input_size)
              hx: of shape (batch_size, hidden_size)
        Output:
              hy: of shape (batch_size, hidden_size)
        """

        if hx is None:
            hx = Variable(input.new_zeros(input.size(0), self.hidden_size))

        x_t = self.input2hidden(input)
        h_t = self.hidden2hidden(hx)

        x_reset, x_upd, x_new = x_t.chunk(3, 1)
        h_reset, h_upd, h_new = h_t.chunk(3, 1)

        reset_gate = torch.sigmoid(x_reset + h_reset)
        update_gate = torch.sigmoid(x_upd + h_upd)
        new_gate = torch.tanh(x_new + (reset_gate * h_new))

        hy = update_gate * hx + (1 - update_gate) * new_gate

        return hy


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias, output_size):
        super(GRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        self.rnn_cell_list = nn.ModuleList()

        self.rnn_cell_list.append(
            GRUBase(self.input_size, self.hidden_size, self.bias))
        for layer in range(1, self.num_layers):
            self.rnn_cell_list.append(
                GRUBase(self.hidden_size, self.hidden_size, self.bias))
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hx=None):

        # Input of shape (batch_size, seqence length, input_size)
        #
        # Output of shape (batch_size, output_size)

        if hx is None:
            if torch.cuda.is_available():
                h0 = Variable(
                    torch.zeros(self.num_layers, input.size(0),
                                self.hidden_size).cuda())
            else:
                h0 = Variable(
                    torch.zeros(self.num_layers, input.size(0),
                                self.hidden_size))

        else:
            h0 = hx

        outs = []

        hidden = list()
        for layer in range(self.num_layers):
            hidden.append(h0[layer, :, :])

        for t in range(input.size(1)):

            for layer in range(self.num_layers):

                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](input[:, t, :],
                                                         hidden[layer])
                else:
                    hidden_l = self.rnn_cell_list[layer](hidden[layer - 1],
                                                         hidden[layer])
                hidden[layer] = hidden_l

                hidden[layer] = hidden_l

            outs.append(hidden_l)

        # Take only last time step. Modify for seq to seq
        out = outs[-1].squeeze()

        out = self.fc(out)

        return out
