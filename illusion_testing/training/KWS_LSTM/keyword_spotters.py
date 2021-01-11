#
# Copyright (C) 2020 by The Board of Trustees of Stanford University
# This program is free software: you can redistribute it and/or modify it under
# the terms of the Modified BSD-3 License as published by the Open Source
# Initiative.
# If you use this program in your research, we request that you reference the
# Illusion paper, and that you send us a citation of your work.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the BSD-3 License for more details.
# You should have received a copy of the Modified BSD-3 License along with this
# program. If not, see <https://opensource.org/licenses/BSD-3-Clause>.
#


import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, Function
import torch.onnx
from torch.utils.data import Dataset, DataLoader

from qtorch import FixedPoint, FloatingPoint
from qtorch.quant import Quantizer, quantizer
from qtorch.optim import OptimLP

from training_config import TrainingConfig
from rnn import *
from datasets import *
from quants import *

class KeywordSpotter(nn.Module):
    """ This baseclass provides the PyTorch Module pattern for defining and training keyword spotters """

    def __init__(self, input_dim, num_keywords, batch_first=False):
        """
        Initialize the KeywordSpotter with the following parameters:
        input_dim - the size of the input audio frame in # samples
        num_keywords - the number of predictions to come out of the model.
        """
        super(KeywordSpotter, self).__init__()

        self.input_dim = input_dim
        self.num_keywords = num_keywords
        self.batch_first = batch_first
        self.training = False
        self.tracking = False

        self.init_hidden()

    def name(self):
        return "KeywordSpotter"

    def init_hidden(self):
        """ Clear any  hidden state """
        pass

    def forward(self, input):
        """ Perform the forward processing of the given input and return the prediction """
        raise Exception("need to implement the forward method")

    def export(self, name, device):
        """ Export the model to the ONNX file format """
        self.init_hidden()
        self.tracking = True
        dummy_input = Variable(torch.randn(1, 1, self.input_dim))
        if device:
            dummy_input = dummy_input.to(device)
        torch.onnx.export(self, dummy_input, name, verbose=True)
        self.tracking = False

    def batch_accuracy(self, scores, labels):
        """ Compute the training accuracy of the results of a single mini-batch """
        batch_size = scores.shape[0]
        passed = 0
        results = []
        for i in range(batch_size):
            expected = labels[i]
            actual = scores[i].argmax()
            results += [int(actual)]
            if expected == actual:
                passed += 1
        return (float(passed) * 100.0 / float(batch_size), passed, results)

    def fit(self, training_data, validation_data, options, model, device=None, detail=False, run=None):
        """
        Perform the training.  This is not called "train" because the base class already defines
        that method with a different meaning.  The base class "train" method puts the Module into
        "training mode".
        """
        print("Training {} using {} rows of featurized training input...".format(self.name(), training_data.num_rows))

        if training_data.mean is not None:
            self.mean = torch.from_numpy(np.array([[training_data.mean]])).to(device)
            self.std = torch.from_numpy(np.array([[training_data.std]])).to(device)
        else:
            self.mean = None
            self.std = None

        start = time.time()
        #loss_function = nn.CrossEntropyLoss()
        loss_function = nn.NLLLoss()
        initial_rate = options.learning_rate
        lr_scheduler = options.lr_scheduler
        oo = options.optimizer_options
        self.training = True

        if options.optimizer == "Adadelta":
            optimizer = optim.Adadelta(self.parameters(), lr=initial_rate, weight_decay=oo.weight_decay,
                                       rho=oo.rho, eps=oo.eps)
        elif options.optimizer == "Adagrad":
            optimizer = optim.Adagrad(self.parameters(), lr=initial_rate, weight_decay=oo.weight_decay,
                                      lr_decay=oo.lr_decay)
        elif options.optimizer == "Adam":
            optimizer = optim.Adam(self.parameters(), lr=initial_rate, weight_decay=oo.weight_decay,
                                   betas=oo.betas, eps=oo.eps)
        elif options.optimizer == "Adamax":
            optimizer = optim.Adamax(self.parameters(), lr=initial_rate, weight_decay=oo.weight_decay,
                                     betas=oo.betas, eps=oo.eps)
        elif options.optimizer == "ASGD":
            optimizer = optim.ASGD(self.parameters(), lr=initial_rate, weight_decay=oo.weight_decay,
                                   lambd=oo.lambd, alpha=oo.alpha, t0=oo.t0)
        elif options.optimizer == "RMSprop":
            optimizer = optim.RMSprop(self.parameters(), lr=initial_rate, weight_decay=oo.weight_decay,
                                      eps=oo.eps, alpha=oo.alpha, momentum=oo.momentum, centered=oo.centered)
        elif options.optimizer == "Rprop":
            optimizer = optim.Rprop(self.parameters(), lr=initial_rate, etas=oo.etas,
                                    step_sizes=oo.step_sizes)
        elif options.optimizer == "SGD":
            optimizer = optim.SGD(self.parameters(), lr=initial_rate, weight_decay=oo.weight_decay,
                                  momentum=oo.momentum, dampening=oo.dampening)

        num_epochs = options.max_epochs
        batch_size = options.batch_size
        learning_rate = options.learning_rate
        lr_min = options.lr_min
        lr_peaks = options.lr_peaks
        ticks = training_data.num_rows / batch_size  # iterations per epoch

        # Calculation of total iterations in non-rolling vs rolling training
        # ticks = num_rows/batch_size (total number of iterations per epoch)
        # Non-Rolling Training:
        # Total Iteration = num_epochs * ticks
        # Rolling Training:
        # irl = Initial_rolling_length (We are using 2)
        # If num_epochs <=  max_rolling_length:
        # Total Iterations = sum(range(irl, irl + num_epochs))
        # If num_epochs > max_rolling_length:
        # Total Iterations = sum(range(irl, irl + max_rolling_length)) + (num_epochs - max_rolling_length)*ticks
        if options.rolling:
            rolling_length = 2
            max_rolling_length = int(ticks)
            if max_rolling_length > options.max_rolling_length + rolling_length:
                max_rolling_length = options.max_rolling_length + rolling_length
            bag_count = 100
            hidden_bag_size = batch_size * bag_count
            if num_epochs + rolling_length < max_rolling_length:
                max_rolling_length = num_epochs + rolling_length
            total_iterations = sum(range(rolling_length, max_rolling_length))
            if num_epochs + rolling_length > max_rolling_length:
                epochs_remaining = num_epochs + rolling_length - max_rolling_length
                total_iterations += epochs_remaining * training_data.num_rows / batch_size
            ticks = total_iterations / num_epochs
        else:
            total_iterations = ticks * num_epochs
        gamma = options.lr_gamma

        if not lr_min:
            lr_min = learning_rate
        scheduler = None
        if lr_scheduler == "TriangleLR":
            steps = lr_peaks * 2 + 1
            stepsize = num_epochs / steps
            scheduler = TriangularLR(optimizer, stepsize * ticks, lr_min, learning_rate, gamma)
        elif lr_scheduler == "CosineAnnealingLR":
            # divide by odd number to finish on the minimum learning rate
            cycles = lr_peaks * 2 + 1
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iterations / cycles,
                                                             eta_min=lr_min)
        elif lr_scheduler == "ExponentialLR":
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        elif lr_scheduler == "StepLR":
            print(options.lr_step_size)
            print(gamma)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=options.lr_step_size, gamma=gamma)
        elif lr_scheduler == "ExponentialResettingLR":
            reset = (num_epochs * ticks) / 3  # reset at the 1/3 mark.
            scheduler = ExponentialResettingLR(optimizer, gamma, reset)

        optimizer = OptimLP(optimizer,
                    weight_quant=weight_quant,
                    grad_scaling=1/1000)
        log = []
        for epoch in range(num_epochs):
            self.train()
            if options.rolling:
                rolling_length += 1
                if rolling_length <= max_rolling_length:
                    hidden1_bag = torch.from_numpy(np.zeros([1, hidden_bag_size, model.hidden_units],
                                                            dtype=np.float32)).to(device)
                    if model.architecture == 'LSTM':
                        cell1_bag = torch.from_numpy(np.zeros([1, hidden_bag_size, model.hidden_units],
                                                              dtype=np.float32)).to(device)
                    if model.num_layers >= 2:
                        hidden2_bag = torch.from_numpy(np.zeros([1, hidden_bag_size, model.hidden_units],
                                                                dtype=np.float32)).to(device)
                        if model.architecture == 'LSTM':
                            cell2_bag = torch.from_numpy(np.zeros([1, hidden_bag_size, model.hidden_units],
                                                                  dtype=np.float32)).to(device)
                    if model.num_layers == 3:
                        hidden3_bag = torch.from_numpy(np.zeros([1, hidden_bag_size, training_data.num_keywords],
                                                                dtype=np.float32)).to(device)
                        if model.architecture == 'LSTM':
                            cell3_bag = torch.from_numpy(np.zeros([1, hidden_bag_size, training_data.num_keywords],
                                                                  dtype=np.float32)).to(device)
            for i_batch, (audio, labels) in enumerate(training_data.get_data_loader(batch_size)):
                if not self.batch_first:
                    audio = audio.transpose(1, 0)  # GRU wants seq,batch,feature

                if device:
                    audio = audio.to(device)
                    labels = labels.to(device)

                # Also, we need to clear out the hidden state,
                # detaching it from its history on the last instance.
                if options.rolling:
                    if rolling_length <= max_rolling_length:
                        if (i_batch + 1) % rolling_length == 0:
                            self.init_hidden()
                            break
                    shuffled_indices = list(range(hidden_bag_size))
                    np.random.shuffle(shuffled_indices)
                    temp_indices = shuffled_indices[:batch_size]
                    if model.architecture == 'LSTM':
                        if self.hidden1 is not None:
                            hidden1_bag[:, temp_indices, :], cell1_bag[:, temp_indices, :] = self.hidden1
                            self.hidden1 = (hidden1_bag[:, 0:batch_size, :], cell1_bag[:, 0:batch_size, :])
                            if model.num_layers >= 2:
                                hidden2_bag[:, temp_indices, :], cell2_bag[:, temp_indices, :] = self.hidden2
                                self.hidden2 = (hidden2_bag[:, 0:batch_size, :], cell2_bag[:, 0:batch_size, :])
                            if model.num_layers == 3:
                                hidden3_bag[:, temp_indices, :], cell3_bag[:, temp_indices, :] = self.hidden3
                                self.hidden3 = (hidden3_bag[:, 0:batch_size, :], cell3_bag[:, 0:batch_size, :])
                    else:
                        if self.hidden1 is not None:
                            hidden1_bag[:, temp_indices, :] = self.hidden1
                            self.hidden1 = hidden1_bag[:, 0:batch_size, :]
                            if model.num_layers >= 2:
                                hidden2_bag[:, temp_indices, :] = self.hidden2
                                self.hidden2 = hidden2_bag[:, 0:batch_size, :]
                            if model.num_layers == 3:
                                hidden3_bag[:, temp_indices, :] = self.hidden3
                                self.hidden3 = hidden3_bag[:, 0:batch_size, :]
                else:
                    self.init_hidden()

                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the variables it will update (which are the learnable
                # weights of the model). This is because by default, gradients are
                # accumulated in buffers( i.e, not overwritten) whenever .backward()
                # is called. Checkout docs of torch.autograd.backward for more details.
                optimizer.zero_grad()

                # optionally normalize the audio
                if self.mean is not None:
                    audio = (audio - self.mean) / self.std

                # Run our forward pass.
                keyword_scores = self(audio)

                # Compute the loss, gradients
                loss = loss_function(keyword_scores, labels)*1000
                
                # Backward pass: compute gradient of the loss with respect to all the learnable
                # parameters of the model. Internally, the parameters of each Module are stored
                # in Tensors with requires_grad=True, so this call will compute gradients for
                # all learnable parameters in the model.
                loss.backward()

                # Calling the step function on an Optimizer makes an update to its parameters
                # applying the gradients we computed during back propagation
                optimizer.step()
                

                learning_rate = optimizer.param_groups[0]['lr']
                if detail:
                    learning_rate = optimizer.param_groups[0]['lr']
                    log += [{'iteration': iteration, 'loss': loss.item(), 'learning_rate': learning_rate}]
            # Find the best prediction in each sequence and return it's accuracy
            # move to next learning rate
            if scheduler:
                scheduler.step()
            passed, total, rate = self.evaluate(validation_data, batch_size, device)
            learning_rate = optimizer.param_groups[0]['lr']
            current_loss = float(loss.item())
            print("Epoch {}, Loss {:.3f}, Validation Accuracy {:.3f}, Learning Rate {}".format(
                  epoch, current_loss, rate * 100, learning_rate))
            log += [{'epoch': epoch, 'loss': current_loss, 'accuracy': rate, 'learning_rate': learning_rate}]
            if run is not None:
                run.log('progress', epoch / num_epochs)
                run.log('epoch', epoch)
                run.log('accuracy', rate)
                run.log('loss', current_loss)
                run.log('learning_rate', learning_rate)

        end = time.time()
        self.training = False
        print("Trained in {:.2f} seconds".format(end - start))
        return log

    def evaluate(self, test_data, batch_size, device=None, outfile=None):
        """
        Evaluate the given test data and print the pass rate
        """
        self.eval()
        passed = 0
        total = 0

        if test_data.mean is not None:
            mean = torch.from_numpy(np.array([[test_data.mean]])).to(device)
            std = torch.from_numpy(np.array([[test_data.std]])).to(device)
        else:
            mean = None
            std = None

        self.zero_grad()
        results = []
        with torch.no_grad():
            for i_batch, (audio, labels) in enumerate(test_data.get_data_loader(batch_size,shuffle=False)):
                batch_size = audio.shape[0]
                if not self.batch_first:
                    audio = audio.transpose(1, 0)  # GRU wants seq,batch,feature
                if device:
                    audio = audio.to(device)
                    labels = labels.to(device)
                if mean is not None:
                    #print("De-mean audio")
                    audio = (audio - mean) / std
                total += batch_size
                self.init_hidden()
                keyword_scores = self(audio)
                last_accuracy, ok, actual = self.batch_accuracy(keyword_scores, labels)
                results += actual
                passed += ok

        if outfile:
            print("Saving evaluation results in '{}'".format(outfile))
            with open(outfile, "w") as f:
                json.dump(results, f)

        return (passed, total, passed / total)

class LSTMKeywordSpotter(KeywordSpotter):
    """This class is a PyTorch Module that implements a 1, 2 or 3 layer LSTM based audio classifier"""

    def __init__(self, input_dim, num_keywords, model):
        """
        Initialize the KeywordSpotter with the following parameters:
        input_dim - the size of the input audio frame in # samples.
        hidden_units - the size of the hidden state of the LSTM nodes
        num_keywords - the number of predictions to come out of the model.
        num_layers - the number of LSTM layers to use (1, 2 or 3)
        """
        self.hidden_units = model.hidden_units
        self.num_layers = model.num_layers
        self.input_dim = input_dim
        super(LSTMKeywordSpotter, self).__init__(input_dim, num_keywords)

        # The LSTM takes audio sequences as input, and outputs hidden states
        # with dimensionality hidden_units.
        self.lstm1 = LSTM(input_dim, self.hidden_units)
        self.lstm2 = None
        if self.num_layers > 1:
            self.lstm2 = LSTM(self.hidden_units, self.hidden_units)
        self.lstm3 = None
        last_output_size = self.hidden_units
        if self.num_layers > 2:
            # layer 3 can reduce output to num_keywords, this makes for a smaller
            # layer and a much smaller Linear layer below so we get some of the
            # size back.
            self.lstm3 = LSTM(self.hidden_units, num_keywords)
            last_output_size = num_keywords

        # The linear layer is a fully connected layer that maps from hidden state space
        # to number of expected keywords
        self.hidden2keyword = nn.Linear(last_output_size, num_keywords)
        self.init_hidden()

    def name(self):
        return "{} layer LSTM {}".format(self.num_layers, self.hidden_units)

    def init_hidden(self):
        """ Clear the hidden state for the LSTM nodes """
        self.hidden1 = None
        self.hidden2 = None
        self.hidden3 = None
        self.cell1 = None
        self.cell2 = None
        self.cell3 = None

    def forward(self, input):
        """ Perform the forward processing of the given input and return the prediction """
        # input is shape: [seq,batch,feature]
        if self.tracking:
            if self.mean is not None:
                input = (input - self.mean) / self.std
        lstm_out1, lstm_cell1 = self.lstm1(input, hiddenState=self.hidden1, cellState = self.cell1, batch_first=False)
        # we have to detach the hidden states because we may keep them longer than 1 iteration.
        self.hidden1 = lstm_out1.detach()[-1,:,:]
        self.cell1 = lstm_cell1.detach()[-1,:,:]
        if self.lstm2 is not None:
            lstm_out2, lstm_cell2 = self.lstm2(lstm_out1, hiddenState=self.hidden2, cellState = self.cell2, batch_first=False)
            lstm_output = lstm_out2
            # we have to detach the hidden states because we may keep them longer than 1 iteration.
            self.hidden2 = lstm_out2.detach()[-1,:,:]
            self.cell2 = lstm_cell2.detach()[-1,:,:]
        if self.lstm3 is not None:
            lstm_out3, lstm_cell3 = self.lstm3(lstm_out2, hiddenState=self.hidden3, cellState = self.cell3, batch_first=False)
            lstm_output = lstm_out3
            # we have to detach the hidden states because we may keep them longer than 1 iteration.
            self.hidden3 = lstm_out3.detach()[-1,:,:]
            self.cell3 = lstm_cell3.detach()[-1,:,:]
        keyword_space = self.hidden2keyword(lstm_out1[-1, :, :])
        result = F.log_softmax(keyword_space, dim=1)
        return result

class LSTMnnKeywordSpotter(KeywordSpotter):
    """This class is a PyTorch Module that implements a 1, 2 or 3 layer LSTM based audio classifier"""

    def __init__(self, input_dim, num_keywords, model):
        """
        Initialize the KeywordSpotter with the following parameters:
        input_dim - the size of the input audio frame in # samples.
        hidden_units - the size of the hidden state of the LSTM nodes
        num_keywords - the number of predictions to come out of the model.
        num_layers - the number of LSTM layers to use (1, 2 or 3)
        """
        self.hidden_units = model.hidden_units
        self.num_layers = model.num_layers
        self.input_dim = input_dim
        super(LSTMnnKeywordSpotter, self).__init__(input_dim, num_keywords)

        # The LSTM takes audio sequences as input, and outputs hidden states
        # with dimensionality hidden_units.
        self.lstm1 = nn.LSTM(input_dim, self.hidden_units)
        self.lstm2 = None
        if self.num_layers > 1:
            self.lstm2 = nn.LSTM(self.hidden_units, self.hidden_units)
        self.lstm3 = None
        last_output_size = self.hidden_units
        if self.num_layers > 2:
            # layer 3 can reduce output to num_keywords, this makes for a smaller
            # layer and a much smaller Linear layer below so we get some of the
            # size back.
            self.lstm3 = nn.LSTM(self.hidden_units, num_keywords)
            last_output_size = num_keywords

        # The linear layer is a fully connected layer that maps from hidden state space
        # to number of expected keywords
        self.hidden2keyword = nn.Linear(last_output_size, num_keywords)
        self.init_hidden()

    def name(self):
        return "{} layer LSTM {}".format(self.num_layers, self.hidden_units)

    def init_hidden(self):
        """ Clear the hidden state for the LSTM nodes """
        self.hidden1 = None
        self.hidden2 = None
        self.hidden3 = None
    
    def forward(self, input):
        """ Perform the forward processing of the given input and return the prediction """
        # input is shape: [seq,batch,feature]
        if self.tracking:
            if self.mean is not None:
                input = (input - self.mean) / self.std
        self.lstm1.flatten_parameters()
        lstm_out, self.hidden1 = self.lstm1(inputq, self.hidden1)
        hidden, cell = self.hidden1
        # we have to detach the hidden states because we may keep them longer than 1 iteration.
        self.hidden1 = (hidden.detach(), cell.detach())
        if self.lstm2 is not None:
            self.lstm2.flatten_parameters()
            lstm_out, self.hidden2 = self.lstm2(lstm_out, self.hidden2)
            hidden, cell = self.hidden2
            self.hidden2 = (hidden.detach(), cell.detach())
        if self.lstm3 is not None:
            self.lstm3.flatten_parameters()
            lstm_out, self.hidden3 = self.lstm3(lstm_out, self.hidden3)
            hidden, cell = self.hidden3
            self.hidden3 = (hidden.detach(), cell.detach())
        keyword_space = self.hidden2keyword(lstm_out[-1, :, :])
        result = F.log_softmax(keyword_space, dim=1)
        return result

class LSTMQKeywordSpotter(KeywordSpotter):
    """This class is a PyTorch Module that implements a 1, 2 or 3 layer LSTM based audio classifier"""

    def __init__(self, input_dim, num_keywords, act_error_quant, act2_error_quant, model):
        """
        Initialize the KeywordSpotter with the following parameters:
        input_dim - the size of the input audio frame in # samples.
        hidden_units - the size of the hidden state of the LSTM nodes
        num_keywords - the number of predictions to come out of the model.
        num_layers - the number of LSTM layers to use (1, 2 or 3)
        """
        self.hidden_units = model.hidden_units
        self.num_layers = model.num_layers
        self.input_dim = input_dim
        super(LSTMQKeywordSpotter, self).__init__(input_dim, num_keywords)

        # The LSTM takes audio sequences as input, and outputs hidden states
        # with dimensionality hidden_units.
        self.lstm1 = LSTMQ(input_dim, self.hidden_units, act_error_quant, act2_error_quant)
        self.lstm2 = None
        if self.num_layers > 1:
            self.lstm2 = LSTMQ(self.hidden_units, self.hidden_units, act_error_quant, act2_error_quant)
        self.lstm3 = None
        last_output_size = self.hidden_units
        if self.num_layers > 2:
            # layer 3 can reduce output to num_keywords, this makes for a smaller
            # layer and a much smaller Linear layer below so we get some of the
            # size back.
            self.lstm3 = LSTMQ(self.hidden_units, num_keywords, act_error_quant, act2_error_quant)
            last_output_size = num_keywords

        # The linear layer is a fully connected layer that maps from hidden state space
        # to number of expected keywords
        self.hidden2keyword = nn.Linear(last_output_size, num_keywords)
        self.init_hidden()

    def name(self):
        return "{} layer LSTM {}".format(self.num_layers, self.hidden_units)

    def init_hidden(self):
        """ Clear the hidden state for the LSTM nodes """
        self.hidden1 = None
        self.hidden2 = None
        self.hidden3 = None
        self.cell1 = None
        self.cell2 = None
        self.cell3 = None

    def forward(self, input):
        """ Perform the forward processing of the given input and return the prediction """
        # input is shape: [seq,batch,feature]
        if self.tracking:
            if self.mean is not None:
                input = (input - self.mean) / self.std
        lstm_out1, lstm_cell1 = self.lstm1(input, hiddenState=self.hidden1, cellState = self.cell1, batch_first=False)
        # we have to detach the hidden states because we may keep them longer than 1 iteration.
        self.hidden1 = lstm_out1.detach()[-1,:,:]
        self.cell1 = lstm_cell1.detach()[-1,:,:]
        if self.lstm2 is not None:
            lstm_out2, lstm_cell2 = self.lstm2(lstm_out1, hiddenState=self.hidden2, cellState = self.cell2, batch_first=False)
            lstm_output = lstm_out2
            # we have to detach the hidden states because we may keep them longer than 1 iteration.
            self.hidden2 = lstm_out2.detach()[-1,:,:]
            self.cell2 = lstm_cell2.detach()[-1,:,:]
        if self.lstm3 is not None:
            lstm_out3, lstm_cell3 = self.lstm3(lstm_out2, hiddenState=self.hidden3, cellState = self.cell3, batch_first=False)
            lstm_output = lstm_out3
            # we have to detach the hidden states because we may keep them longer than 1 iteration.
            self.hidden3 = lstm_out3.detach()[-1,:,:]
            self.cell3 = lstm_cell3.detach()[-1,:,:]
        keyword_space = self.hidden2keyword(lstm_out1[-1, :, :])
        #return keyword_space
        result = F.log_softmax(keyword_space, dim=1)
        return result
