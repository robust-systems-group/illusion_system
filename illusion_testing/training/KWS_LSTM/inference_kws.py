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


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from qtorch.quant import fixed_point_quantize
from training_config import TrainingConfig
import os
import pdb

wl = 8
fl = 4

normal_test = False
# normal_test = True

def sigmoid_approx(a):
    bound = 3
    mask = ((a < bound) * (a > -bound)).type(torch.float32)
    out = (a > bound).type(torch.float32) + (a*mask*(1/(2*bound))) + 0.5*mask
    
    return out

def sigmoid_approx_C(a):
    bound = 3 * 2**8
    x = a * 2**8
    mask = ((x <= bound) * (x >= -bound)).type(torch.float32)
    out = (x > bound).type(torch.float32)*2**8 + (43*x*mask*(1/(2**8))).floor() + 0.5*mask*2**8
    # pdb.set_trace()
    
    return out/2**8


def tanh_approx(a):
    bound = 1
    mask = ((a < bound) * (a > -bound)).type(torch.float32)
    out = -1*(a <= -bound).type(torch.float32) + (a >= bound).type(torch.float32) +  (a*mask)

    return out



def lstm_step_forward(t,x, prev_h, prev_c, Wx, Wh, b, pause):
    """
    Forward pass for a single timestep of an LSTM.
    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.
    Note that a sigmoid() function has already been provided for you in this file.
    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)
    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N,H = prev_c.shape
    # x_q = fixed_point_quantize(x,32,4,rounding="nearest")
    # prev_h_q = fixed_point_quantize(prev_h,32,4,rounding="nearest")
    # prev_state = torch.mm(Wh, torch.transpose(prev_h_q,0,1))
    prev_state = torch.mm(Wh, torch.transpose(prev_h,0,1))

    # op = torch.mm(Wx, torch.transpose(x_q,0,1)) + prev_state
    op = torch.mm(Wx, torch.transpose(x,0,1)) + prev_state
    test0 = torch.mm(Wx, torch.transpose(x,0,1))
    test1 = torch.transpose(torch.mm(Wh, torch.transpose(prev_h,0,1)),0,1)
    
    op = torch.transpose(op,0,1) + b

    ai = op[:,:H]
    af = op[:,H:2*H]
    ag = op[:,2*H:3*H]
    ao = op[:,3*H:]
    
    # i = torch.sigmoid(ai)`
    # f = torch.sigmoid(af)
    # o = torch.sigmoid(ao)
    # g = torch.tanh(ag)

    i = sigmoid_approx_C(ai)
    f = sigmoid_approx_C(af)
    o = sigmoid_approx_C(ao)
    g = tanh_approx(ag)
    # test = sigmoid_approx_C(ai)
    # pdb.set_trace()

    next_c = f*prev_c + i*g
    # next_c_q = fixed_point_quantize(next_c,32,8,rounding="nearest")
    # tanh_next_c = tanh_approx(next_c_q)
    tanh_next_c = tanh_approx(next_c)
    # next_h = o*torch.tanh(next_c)
    next_h = o*tanh_next_c#tanh_approx(next_c)
    # next_h_q = fixed_point_quantize(next_h,32,8,rounding="nearest")


    cache = (H, op, i, f, o, g, x, Wx, Wh, b, prev_c, prev_h, next_c, next_h)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    # pdb.set_trace()
    # return next_h_q, next_c_q, cache
    return next_h, next_c, cache


def lstm_forward(x, h0, c0, Wx, Wh, b, pause):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.
    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.
    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)
    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N,T,D = x.shape
    N,H = h0.shape
               
    h = torch.zeros((N,T,H))
    c = c0
    cache = [None for i in range(T)]

    for i in range(T):
        if i==0:
            h[:,i,:] , c, cache[i] = lstm_step_forward(i,torch.squeeze(x[:,i,:]), h0, c, Wx, Wh, b, pause)
        else:
            h[:,i,:] , c, cache[i] = lstm_step_forward(i,torch.squeeze(x[:,i,:]), h[:,i-1,:], c, Wx, Wh, b, pause)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, (torch.squeeze(h[:,T-1,:]), c)


class AudioDataset(Dataset):
    """
    Featurized Audio in PyTorch Dataset so we can get a DataLoader that is needed for
    mini-batch training.
    """

    def __init__(self, filename, config, keywords):
        """ Initialize the AudioDataset from the given *.npz file """
        self.dataset = np.load(filename)

        # get parameters saved by make_dataset.py
        parameters = self.dataset["parameters"]
        self.sample_rate = int(parameters[0])
        self.audio_size = int(parameters[1])
        self.input_size = int(parameters[2])
        self.window_size = int(parameters[3])
        self.shift = int(parameters[4])
        self.features = self.dataset["features"].astype(np.float32)
        self.num_rows = len(self.features)
        self.features = self.features.reshape((self.num_rows, self.window_size, self.input_size))

        if config.normalize:
            mean = self.features.mean(axis=0)
            std = self.features.std(axis=0)
            self.mean = mean.mean(axis=0).astype(np.float32)
            std = std.mean(axis=0)
            # self.std is a divisor, so make sure it contains no zeros
            self.std = np.array(np.where(std == 0, 1, std)).astype(np.float32)
        else:
            self.mean = None
            self.std = None
        self.label_names = self.dataset["labels"]
        self.keywords = keywords
        self.num_keywords = len(self.keywords)
        self.labels = self.to_long_vector()

        msg = "Loaded dataset {} and found sample rate {}, audio_size {}, input_size {}, window_size {} and shift {}"
        print(msg.format(os.path.basename(filename), self.sample_rate, self.audio_size, self.input_size,
                         self.window_size, self.shift))

    def get_data_loader(self, batch_size):
        """ Get a DataLoader that can enumerate shuffled batches of data in this dataset """
        return DataLoader(self, batch_size=batch_size, shuffle=True, drop_last=True)

    def to_long_vector(self):
        """ convert the expected labels to a list of integer indexes into the array of keywords """
        indexer = [(0 if x == "<null>" else self.keywords.index(x)) for x in self.label_names]
        return np.array(indexer, dtype=np.longlong)

    def __len__(self):
        """ Return the number of rows in this Dataset """
        return self.num_rows

    def __getitem__(self, idx):
        """ Return a single labelled sample here as a tuple """
        audio = self.features[idx]  # batch index is second dimension
        label = self.labels[idx]
        sample = (audio, label)
        return sample

class LSTMKeywordSpotter(nn.Module):
    """
    The RNN model that will be used to perform Keyword Spotter classification.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super().__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        # dropout layer
        self.dropout = nn.Dropout(0.3)
        
        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()
        

    def forward(self, x, hidden, Wx1, Wh1, b1, Wx2, Wh2, b2):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out, htuple = lstm_forward(embeds, hidden[0][0], hidden[1][0], Wx1, Wh1, b1.view(1,-1),False)
        hidden[0][0] = htuple[0]
        hidden[1][0] = htuple[1]
        # pdb.set_trace()
        lstm_out, htuple = lstm_forward(lstm_out, hidden[0][1], hidden[1][1], Wx2, Wh2, b2.view(1,-1),True)
        hidden[0][1] = htuple[0]
        hidden[1][1] = htuple[1]

        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        pdb.set_trace()
        # sigmoid function
        # sig_out = self.sig(out)
        sig_out = sigmoid_approx_C(out)
        
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels
        
        # return last sigmoid output and hidden state
        return sig_out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden

config = TrainingConfig()
state_dict = torch.load('results/LSTM80KeywordSpotter.pt')
state_dict['fc.weight'] = state_dict['hidden2keyword.weight']
state_dict['fc.bias'] = state_dict['hidden2keyword.bias']

categories_file = "categories.txt"
with open(categories_file, "r") as f:
        keywords = [x.strip() for x in f.readlines()]

# state_dict = torch.load('../checkpoints/LSTM_sentiment_analysis_ckpt_16Hidden.pt')
# for key in state_dict.keys(): # CHANGE BACK
#     state_dict[key] = fixed_point_quantize(state_dict[key],wl,fl,rounding="nearest")
train_on_gpu = False
criterion = nn.BCELoss()

vocab_size = 121365 # +1 for the 0 padding
output_size = 11
embedding_dim = 400
hidden_dim = 80
n_layers = 1

net = LSTMKeywordSpotter(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

if normal_test:
    #test_x = np.load('data/sentiment_data/test_data.npy')
    #test_y = np.load('data/sentiment_data/test_labels.npy')
    test_x = np.load('../data/sentiment_data/embedded_test_data.npy')
    test_y = np.load('../data/sentiment_data/embedded_test_labels.npy')
    # create Tensor datasets
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
    batch_size = 50
else:
    test_file = 'testing_list.npz'
    test_data = AudioDataset(test_file, config.dataset, keywords)
    print("Evaluating LSTM keyword spotter using {} rows of featurized test audio...".format(
            test_data.num_rows))
    msg = "Loading trained model with input size {}, hidden units {} and num keywords {}"
    print(msg.format(test_data.input_size, hidden_dim, test_data.num_keywords))
    #pdb.set_trace()

# Get test data loss and accuracy

test_losses = [] # track loss
num_correct = 0

net.load_state_dict(state_dict, strict = False)

Wx1 = state_dict['lstm.weight_ih_l0']
Wh1 = state_dict['lstm.weight_hh_l0']
b1 = state_dict['lstm.bias_ih_l0'] + state_dict['lstm.bias_hh_l0']

# init hidden state
h = net.init_hidden(batch_size)
counter = 0

net.eval()
print('Starting inference...')
# iterate over test data
with torch.no_grad():
    if normal_test:
        for inputs, labels in test_loader:
            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])
            # pdb.set_trace()
            # get predicted outputs
            #inputs = inputs.type(torch.LongTensor)
            output, h = net(inputs, h, Wx1, Wh1, b1, Wx2, Wh2, b2)
        
            # calculate loss
            test_loss = criterion(output.squeeze(), labels.float())
            test_losses.append(test_loss.item())
        
            # convert output probabilities to predicted class (0 or 1)
            pred = torch.round(output.squeeze())  # rounds to the nearest integer
        
            # compare predictions to true label
            correct_tensor = pred.eq(labels.float().view_as(pred))
            correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
            num_correct += np.sum(correct)

            if counter % 10 == 0:
                print("Count =",counter)
            counter += 1
    else:
        # inputs_q = inputs # CHANGE BACK
        # inputs_q = fixed_point_quantize(inputs,wl,fl,rounding='nearest')
        # pdb.set_trace()
        # output, h = net(inputs_q, h, Wx1, Wh1, b1, Wx2, Wh2, b2)
        output, h = net(inputs, h, Wx1, Wh1, b1, Wx2, Wh2, b2)
        #pdb.set_trace()
        # test_loss = criterion(output.squeeze(), labels.float())
        # test_losses.append(test_loss.item())
        pred = torch.round(output.squeeze())  # rounds to the nearest integer
        correct_tensor = pred.eq(labels.float().view_as(pred))
        #pdb.set_trace()
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)


if normal_test:
    dataLen = len(test_loader.dataset)
else:
    dataLen = len(inputs)
# -- stats! -- ##
# avg test loss
# print("Test loss: {:.3f}".format(np.mean(test_losses)))

# accuracy over all test data
#pdb.set_trace()
print('# correct:',num_correct,'total:',dataLen)
test_acc = num_correct/dataLen
print("Test accuracy: {:.3f}".format(test_acc))
