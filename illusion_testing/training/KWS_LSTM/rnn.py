# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
# !/usr/bin/env python3
###################################################################################################
#
#  Project:  EdgeML (https://github.com/microsoft/EdgeML/blob/pytorch/pytorch/pytorch_edgeml/graph/rnn.py)
#  File:     rnn.py
#  Requires: Python 3.x
#
###################################################################################################

import torch
import torch.nn as nn


def gen_nonlinearity(A, nonlinearity):
    '''
    Returns required activation for a tensor based on the inputs

    nonlinearity is either a callable or a value in
        ['tanh', 'sigmoid', 'relu', 'quantTanh', 'quantSigm', 'quantSigm4']
    '''
    if nonlinearity == "tanh":
        return torch.tanh(A)
    elif nonlinearity == "sigmoid":
        return torch.sigmoid(A)
    elif nonlinearity == "relu":
        return torch.relu(A, 0.0)
    elif nonlinearity == "quantTanh":
        return torch.max(torch.min(A, torch.ones_like(A)), -1.0 * torch.ones_like(A))
    elif nonlinearity == "quantSigmoid":
        A = (A + 1.0) / 2.0
        return torch.max(torch.min(A, torch.ones_like(A)), torch.zeros_like(A))
    elif nonlinearity == "quantSigmoid4":
        A = (A + 2.0) / 4.0
        return torch.max(torch.min(A, torch.ones_like(A)), torch.zeros_like(A))
    elif nonlinearity == "quantSigmoid8":
        A = (A + 4.0) / 8.0
        return torch.max(torch.min(A, torch.ones_like(A)), torch.zeros_like(A))
    else:
        # nonlinearity is a user specified function
        if not callable(nonlinearity):
            raise ValueError("nonlinearity must either be a callable or a value " +
                             + "['tanh', 'sigmoid', 'relu', 'quantTanh', " +
                             "'quantSigm'")
        return nonlinearity(A)


class BaseRNN(nn.Module):
    '''
    Generic equivalent of static_rnn in tf
    Used to unroll all the cell written in this file
    We assume data to be batch_first by default ie.,
    [batchSize, timeSteps, inputDims] else
    [timeSteps, batchSize, inputDims]
    '''

    def __init__(self, RNNCell):
        super(BaseRNN, self).__init__()
        self.RNNCell = RNNCell

    def getWeights(self):
        return self.RNNCell.getWeights()

    def forward(self, input, hiddenState=None,
                cellState=None, batch_first=True):
        #print(input.shape)
        if batch_first is True:
            self.device = input.device
            hiddenStates = torch.zeros(
                [input.shape[0], input.shape[1],
                 self.RNNCell.output_size]).to(self.device)
            if hiddenState is None:
                hiddenState = torch.zeros([input.shape[0],
                                           self.RNNCell.output_size]).to(self.device)
            #print(self.RNNCell.cellType)
            if "LSTM" in self.RNNCell.cellType:
                cellStates = torch.zeros(
                    [input.shape[0], input.shape[1],
                     self.RNNCell.output_size]).to(self.device)
                if cellState is None:
                    cellState = torch.zeros(
                        [input.shape[0], self.RNNCell.output_size]).to(self.device)
                for i in range(0, input.shape[1]):
                    hiddenState, cellState = self.RNNCell(
                        input[:, i, :], (hiddenState, cellState))
                    hiddenStates[:, i, :] = hiddenState
                    cellStates[:, i, :] = cellState
                
                return hiddenStates, cellStates
            else:
                for i in range(0, input.shape[1]):
                    hiddenState = self.RNNCell(input[:, i, :], hiddenState)
                    hiddenStates[:, i, :] = hiddenState
                return hiddenStates
        else:
            self.device = input.device
            hiddenStates = torch.zeros(
                [input.shape[0], input.shape[1],
                 self.RNNCell.output_size]).to(self.device)
            if hiddenState is None:
                hiddenState = torch.zeros([input.shape[1],
                                           self.RNNCell.output_size]).to(self.device)
            if "LSTM" in self.RNNCell.cellType:
                cellStates = torch.zeros(
                    [input.shape[0], input.shape[1],
                     self.RNNCell.output_size]).to(self.device)
                if cellState is None:
                    cellState = torch.zeros(
                        [input.shape[1], self.RNNCell.output_size]).to(self.device)
                for i in range(0, input.shape[0]):
                    hiddenState, cellState = self.RNNCell(
                        input[i, :, :], (hiddenState, cellState))
                    hiddenStates[i, :, :] = hiddenState
                    cellStates[i, :, :] = cellState
                return hiddenStates, cellStates
            else:
                hiddenState = hiddenState[0]
                for i in range(0, input.shape[0]):
                    hiddenState = self.RNNCell(input[i, :, :], hiddenState)
                    hiddenStates[i, :, :] = hiddenState
                return hiddenStates

class LSTMQCell(nn.Module):
    '''
    LSTM Cell with Full Rank 
    Has multiple activation functions for the gates
    hidden_size = # hidden units

    gate_nonlinearity = nonlinearity for the gate can be chosen from
    [tanh, sigmoid, relu, quantTanh, quantSigm]
    update_nonlinearity = nonlinearity for final rnn update
    can be chosen from [tanh, sigmoid, relu, quantTanh, quantSigm]

    wRank = rank of all W matrices
    (creates 5 matrices if not None else creates 4 matrices)
    uRank = rank of all U matrices
    (creates 5 matrices if not None else creates 4 matrices)

    LSTM architecture and compression techniques are found in
    LSTM paper

    Basic architecture is like:

    f_t = gate_nl(W1x_t + U1h_{t-1} + B_f)
    i_t = gate_nl(W2x_t + U2h_{t-1} + B_i)
    C_t^ = update_nl(W3x_t + U3h_{t-1} + B_c)
    o_t = gate_nl(W4x_t + U4h_{t-1} + B_o)
    C_t = f_t*C_{t-1} + i_t*C_t^
    h_t = o_t*update_nl(C_t)

    Wi and Ui can further parameterised into low rank version by
    Wi = matmul(W, W_i) and Ui = matmul(U, U_i)
    '''

    def __init__(self, input_size, hidden_size, quant, quant2, gate_nonlinearity=torch.sigmoid,
                 update_nonlinearity=torch.tanh,name="LSTMQ"):
        super(LSTMQCell, self).__init__()

        self._input_size = input_size
        self._hidden_size = hidden_size
        self._gate_nonlinearity = torch.sigmoid
        self._update_nonlinearity = torch.tanh
        self._num_weight_matrices = [4, 4]
        self._name = name
        self.quant = quant()
        self.quant2 = quant2()
        self.weight_i = nn.Parameter(0.1 * torch.randn(4*hidden_size, input_size))
        self.weight_h = nn.Parameter(0.1 * torch.randn(4*hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.ones(4*hidden_size))

    @property
    def state_size(self):
        return 2 * self._hidden_size

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def gate_nonlinearity(self):
        return self._gate_nonlinearity

    @property
    def update_nonlinearity(self):
        return self._update_nonlinearity

    @property
    def wRank(self):
        return self._wRank

    @property
    def uRank(self):
        return self._uRank

    @property
    def num_weight_matrices(self):
        return self._num_weight_matrices

    @property
    def name(self):
        return self._name

    @property
    def cellType(self):
        return "LSTMQ"

    def forward(self, input, hiddenStates):
        (h, c) = hiddenStates
        print(input.shape)
        print(self.quant)
        inputq = self.quant(input)
        hq = self.quant(h)
        cq = self.quant(c)
        print(inputq[0])
        print(hq[0])
        print(self.bias)
        gates = self.quant2(torch.mm(inputq, self.weight_i.t()) + torch.mm(hq, self.weight_h.t()) + self.bias)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        #print(gates.shape)
        #print(gates[0].detach().cpu())
        ingate = self.quant(gen_nonlinearity(ingate,'quantSigmoid8'))
        forgetgate = self.quant(gen_nonlinearity(forgetgate,'quantSigmoid8'))
        cellgate = self.quant(gen_nonlinearity(cellgate,'quantTanh'))
        outgate = self.quant(gen_nonlinearity(outgate,'quantSigmoid8'))
        #ingate = self._gate_nonlinearity(ingate)
        #forgetgate = self._gate_nonlinearity(forgetgate)
        #cellgate = self._update_nonlinearity(cellgate)
        #outgate = self._gate_nonlinearity(outgate)
        print("Gates")
        print(ingate[0])
        print(forgetgate[0])
        print(cellgate[0])
        print(outgate[0])
        print("Outputs") 
        new_c = self.quant2((forgetgate * cq) + (ingate * cellgate))
        #new_h = outgate * self._update_nonlinearity(new_c)
        print(new_c[0])
        new_h = outgate * self.quant(gen_nonlinearity(new_c,'quantTanh'))
        print(new_h[0])
        return new_h, new_c

    def getVars(self):
        Vars = []
        Vars.extend([self.W1, self.W2, self.W3, self.W4])
        Vars.extend([self.U1, self.U2, self.U3, self.U4])
        Vars.extend([self.bias_f, self.bias_i, self.bias_c, self.bias_o])
        return Vars

class LSTMCell(nn.Module):
    '''
    LSTM Cell with Full Rank 
    Has multiple activation functions for the gates
    hidden_size = # hidden units

    gate_nonlinearity = nonlinearity for the gate can be chosen from
    [tanh, sigmoid, relu, quantTanh, quantSigm]
    update_nonlinearity = nonlinearity for final rnn update
    can be chosen from [tanh, sigmoid, relu, quantTanh, quantSigm]

    wRank = rank of all W matrices
    (creates 5 matrices if not None else creates 4 matrices)
    uRank = rank of all U matrices
    (creates 5 matrices if not None else creates 4 matrices)

    LSTM architecture and compression techniques are found in
    LSTM paper

    Basic architecture is like:

    f_t = gate_nl(W1x_t + U1h_{t-1} + B_f)
    i_t = gate_nl(W2x_t + U2h_{t-1} + B_i)
    C_t^ = update_nl(W3x_t + U3h_{t-1} + B_c)
    o_t = gate_nl(W4x_t + U4h_{t-1} + B_o)
    C_t = f_t*C_{t-1} + i_t*C_t^
    h_t = o_t*update_nl(C_t)

    Wi and Ui can further parameterised into low rank version by
    Wi = matmul(W, W_i) and Ui = matmul(U, U_i)
    '''

    def __init__(self, input_size, hidden_size, gate_nonlinearity=torch.sigmoid,
                 update_nonlinearity=torch.tanh,name="LSTM"):
        super(LSTMCell, self).__init__()

        self._input_size = input_size
        self._hidden_size = hidden_size
        self._gate_nonlinearity = torch.sigmoid
        self._update_nonlinearity = torch.tanh
        self._name = name
        self.weight_i = nn.Parameter(0.1 * torch.randn(4*hidden_size, input_size))
        self.weight_h = nn.Parameter(0.1 * torch.randn(4*hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.ones(4*hidden_size))

    @property
    def state_size(self):
        return 2 * self._hidden_size

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def gate_nonlinearity(self):
        return self._gate_nonlinearity

    @property
    def update_nonlinearity(self):
        return self._update_nonlinearity

    @property
    def name(self):
        return self._name

    @property
    def cellType(self):
        return "LSTM"

    def forward(self, input, hiddenStates):
        (h, c) = hiddenStates
        gates = (torch.mm(input, self.weight_i.t()) + torch.mm(h, self.weight_h.t()) + self.bias)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = self._gate_nonlinearity(ingate)
        forgetgate = self._gate_nonlinearity(forgetgate)
        cellgate = self._update_nonlinearity(cellgate)
        outgate = self._gate_nonlinearity(outgate)
        new_c = (forgetgate * c) + (ingate * cellgate)
        new_h = outgate * self._update_nonlinearity((new_c))
        return new_h, new_c

    def getVars(self):
        Vars = []
        Vars.extend([self.weight_i, self.weight_h, self.bias])
        return Vars

class LSTMLRCell(nn.Module):
    '''
    LR - Low Rank
    LSTM LR Cell with Both Full Rank and Low Rank Formulations
    Has multiple activation functions for the gates
    hidden_size = # hidden units

    gate_nonlinearity = nonlinearity for the gate can be chosen from
    [tanh, sigmoid, relu, quantTanh, quantSigm]
    update_nonlinearity = nonlinearity for final rnn update
    can be chosen from [tanh, sigmoid, relu, quantTanh, quantSigm]

    wRank = rank of all W matrices
    (creates 5 matrices if not None else creates 4 matrices)
    uRank = rank of all U matrices
    (creates 5 matrices if not None else creates 4 matrices)

    LSTM architecture and compression techniques are found in
    LSTM paper

    Basic architecture is like:

    f_t = gate_nl(W1x_t + U1h_{t-1} + B_f)
    i_t = gate_nl(W2x_t + U2h_{t-1} + B_i)
    C_t^ = update_nl(W3x_t + U3h_{t-1} + B_c)
    o_t = gate_nl(W4x_t + U4h_{t-1} + B_o)
    C_t = f_t*C_{t-1} + i_t*C_t^
    h_t = o_t*update_nl(C_t)

    Wi and Ui can further parameterised into low rank version by
    Wi = matmul(W, W_i) and Ui = matmul(U, U_i)
    '''

    def __init__(self, input_size, hidden_size, gate_nonlinearity="sigmoid",
                 update_nonlinearity="tanh", wRank=None, uRank=None,
                 name="LSTMLR"):
        super(LSTMLRCell, self).__init__()

        self._input_size = input_size
        self._hidden_size = hidden_size
        self._gate_nonlinearity = gate_nonlinearity
        self._update_nonlinearity = update_nonlinearity
        self._num_weight_matrices = [4, 4]
        self._wRank = wRank
        self._uRank = uRank
        if wRank is not None:
            self._num_weight_matrices[0] += 1
        if uRank is not None:
            self._num_weight_matrices[1] += 1
        self._name = name

        if wRank is None:
            self.W1 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W2 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W3 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
            self.W4 = nn.Parameter(
                0.1 * torch.randn([input_size, hidden_size]))
        else:
            self.W = nn.Parameter(0.1 * torch.randn([input_size, wRank]))
            self.W1 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W2 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W3 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
            self.W4 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))

        if uRank is None:
            self.U1 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U2 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U3 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
            self.U4 = nn.Parameter(
                0.1 * torch.randn([hidden_size, hidden_size]))
        else:
            self.U = nn.Parameter(0.1 * torch.randn([hidden_size, uRank]))
            self.U1 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            self.U2 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            self.U3 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
            self.U4 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))

        self.bias_f = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_i = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_c = nn.Parameter(torch.ones([1, hidden_size]))
        self.bias_o = nn.Parameter(torch.ones([1, hidden_size]))

    @property
    def state_size(self):
        return 2 * self._hidden_size

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def gate_nonlinearity(self):
        return self._gate_nonlinearity

    @property
    def update_nonlinearity(self):
        return self._update_nonlinearity

    @property
    def wRank(self):
        return self._wRank

    @property
    def uRank(self):
        return self._uRank

    @property
    def num_weight_matrices(self):
        return self._num_weight_matrices

    @property
    def name(self):
        return self._name

    @property
    def cellType(self):
        return "LSTMLR"

    def forward(self, input, hiddenStates):
        (h, c) = hiddenStates

        if self._wRank is None:
            wComp1 = torch.matmul(input, self.W1)
            wComp2 = torch.matmul(input, self.W2)
            wComp3 = torch.matmul(input, self.W3)
            wComp4 = torch.matmul(input, self.W4)
        else:
            wComp1 = torch.matmul(
                torch.matmul(input, self.W), self.W1)
            wComp2 = torch.matmul(
                torch.matmul(input, self.W), self.W2)
            wComp3 = torch.matmul(
                torch.matmul(input, self.W), self.W3)
            wComp4 = torch.matmul(
                torch.matmul(input, self.W), self.W4)

        if self._uRank is None:
            uComp1 = torch.matmul(h, self.U1)
            uComp2 = torch.matmul(h, self.U2)
            uComp3 = torch.matmul(h, self.U3)
            uComp4 = torch.matmul(h, self.U4)
        else:
            uComp1 = torch.matmul(
                torch.matmul(h, self.U), self.U1)
            uComp2 = torch.matmul(
                torch.matmul(h, self.U), self.U2)
            uComp3 = torch.matmul(
                torch.matmul(h, self.U), self.U3)
            uComp4 = torch.matmul(
                torch.matmul(h, self.U), self.U4)
        pre_comp1 = wComp1 + uComp1
        pre_comp2 = wComp2 + uComp2
        pre_comp3 = wComp3 + uComp3
        pre_comp4 = wComp4 + uComp4

        i = gen_nonlinearity(pre_comp1 + self.bias_i,
                             self._gate_nonlinearity)
        f = gen_nonlinearity(pre_comp2 + self.bias_f,
                             self._gate_nonlinearity)
        o = gen_nonlinearity(pre_comp4 + self.bias_o,
                             self._gate_nonlinearity)

        c_ = gen_nonlinearity(pre_comp3 + self.bias_c,
                              self._update_nonlinearity)

        new_c = f * c + i * c_
        new_h = o * gen_nonlinearity(new_c, self._update_nonlinearity)
        return new_h, new_c

    def getVars(self):
        Vars = []
        if self._num_weight_matrices[0] == 4:
            Vars.extend([self.W1, self.W2, self.W3, self.W4])
        else:
            Vars.extend([self.W, self.W1, self.W2, self.W3, self.W4])

        if self._num_weight_matrices[1] == 4:
            Vars.extend([self.U1, self.U2, self.U3, self.U4])
        else:
            Vars.extend([self.U, self.U1, self.U2, self.U3, self.U4])

        Vars.extend([self.bias_f, self.bias_i, self.bias_c, self.bias_o])

        return Vars



class LSTM(nn.Module):
    """equivalent to nn.lstm using lstmlrcell"""

    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size

        self.cell = LSTMCell(input_size, hidden_size)
        self.unrollRNN = BaseRNN(self.cell)

    def forward(self, input, hiddenState=None,
                cellState=None, batch_first=True):
        return self.unrollRNN(input, hiddenState, cellState, batch_first)

class LSTMQ(nn.Module):
    def __init__(self, input_size, hidden_size, quant, quant2,
            gate_nonlinearity=torch.sigmoid,update_nonlinearity=torch.tanh):
        super(LSTMQ, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self.quant = quant
        self.quant2 = quant2
        self._gate_nonlinearity = gate_nonlinearity
        self._update_nonlinearity = update_nonlinearity
        self.cell = LSTMQCell(input_size, hidden_size, quant, quant2,
                gate_nonlinearity=gate_nonlinearity,
                update_nonlinearity=update_nonlinearity,
                )
        self.unrollRNN = BaseRNN(self.cell)

    def forward(self, input, hiddenState=None,
                cellState=None, batch_first=True):
        return self.unrollRNN(input, hiddenState, cellState, batch_first)

