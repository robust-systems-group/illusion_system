#!/usr/bin/env python3
###################################################################################################
#
#  Project:  Embedded Learning Library (ELL)
#  File:     train_classifier.py
#  Authors:  Chris Lovett
#
#  Requires: Python 3.x
#
###################################################################################################

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
from keyword_spotters import *

from quants import *


def create_model(model, input_size, num_keywords):
    if model.architecture == "LSTM":
        return LSTMKeywordSpotter(input_size, num_keywords, model)
    elif model.architecture == "LSTMnn":
        return LSTMnnKeywordSpotter(input_size, num_keywords, model)
    elif model.architecture == "LSTMQ":
        return LSTMQKeywordSpotter(input_size, num_keywords, act_error_quant, act2_error_quant, model)
    else:
        raise Exception("Model architecture '{}' not supported".format(arch))


def save_json(obj, filename):
    with open(filename, "w") as f:
        json.dump(obj, f, indent=2)

def size_to_string(n):
    s = ''
    #pdb.set_trace()
    for i in range(0, n.dim()):
        s+= '[' + str(n.shape[i]) + ']'
    return s

def ndarray_to_string(x):
    x = x.numpy()
    s = np.array2string(x, separator=',',threshold=2**32)
    s = s.replace('[', '{');
    s = s.replace(']', '}');
    return s;

def get_activation(name, which, activations):
    def hook(model, input, output):
        if which=='input_in': activations[name] = input[0]#input#.detach()
        #elif which=='hidden_in': activations[name] = input[1][0]#input#.detach()
        #elif which=='cell_in': activations[name] = input[1][1]#input#.detach()
        elif which=='hidden_out': activations[name] = output[0]#input#.detach()
        elif which=='cell_out': activations[name] = output[1]#input#.detach()
        else: activations[name] = output#input#.detach()
    return hook

def test(config, outdir=".", detail=False):

    filename = config.model.filename
    categories_file = config.dataset.categories
    wav_directory = config.dataset.path
    batch_size = config.training.batch_size
    hidden_units = config.model.hidden_units
    architecture = config.model.architecture
    num_layers = config.model.num_layers
    use_gpu = config.training.use_gpu

    run = None

    valid_layers = [1, 2, 3]
    if num_layers not in valid_layers:
        raise Exception("--num_layers can only be one of these values {}".format(valid_layers))

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    if not filename:
        filename = "{}{}KeywordSpotter.pt".format(architecture, hidden_units)
        config.model.filename = filename

    # load the featurized data
    if not os.path.isdir(wav_directory):
        print("### Error: please specify valid --dataset folder location: {}".format(wav_directory))
        sys.exit(1)

    if not categories_file:
        categories_file = os.path.join(wav_directory, "categories.txt")

    with open(categories_file, "r") as f:
        keywords = [x.strip() for x in f.readlines()]

    training_file = os.path.join(wav_directory, "training_list_bal_10.npz")
    testing_file = os.path.join(wav_directory, "testing_list_bal_10.npz")
    validation_file = os.path.join(wav_directory, "validation_list_bal_10.npz")

    if not os.path.isfile(training_file):
        print("Missing file {}".format(training_file))
        print("Please run make_datasets.py")
        sys.exit(1)
    if not os.path.isfile(validation_file):
        print("Missing file {}".format(validation_file))
        print("Please run make_datasets.py")
        sys.exit(1)
    if not os.path.isfile(testing_file):
        print("Missing file {}".format(testing_file))
        print("Please run make_datasets.py")
        sys.exit(1)

    model = None

    device = torch.device("cpu")
    if use_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("### CUDA not available!!")

    print("Loading {}...".format(testing_file))
    test_data = AudioDataset(testing_file, config.dataset, keywords)

    log = None

    print("Evaluating {} keyword spotter using {} rows of featurized test audio...".format(
          architecture, test_data.num_rows))
    if model is None:
        msg = "Loading trained model with input size {}, hidden units {} and num keywords {}"
        print(msg.format(test_data.input_size, hidden_units, test_data.num_keywords))
        model = create_model(config.model, test_data.input_size, test_data.num_keywords)
        model.load_state_dict(torch.load(filename))
        if model and device.type == 'cuda':
            model.cuda()  # move the processing to GPU
    activations = {} 
    model.lstm1.unrollRNN.register_forward_hook(get_activation('lstm_input','input_in',activations))
    model.lstm1.unrollRNN.register_forward_hook(get_activation('lstm_hidden_out','hidden_out',activations))
    model.lstm1.unrollRNN.register_forward_hook(get_activation('lstm_cell_out','cell_out',activations))
    model.hidden2keyword.register_forward_hook(get_activation('lstm_out','output',activations))
    results_file = os.path.join(outdir, "results.txt")
    passed, total, rate = model.evaluate(test_data, config.testing.test_size, device, results_file)
    print("Testing accuracy = {:.3f} %".format(rate * 100))
    
    
    np.savez('int_LSTM.npz',
        inputs = activations['lstm_input'].cpu().detach(), 
        hidden_out = activations['lstm_hidden_out'].cpu().detach(), 
        cell_out = activations['lstm_cell_out'].cpu().detach(), 
        outputs = activations['lstm_out'].cpu().detach()
    
    )
    #return
    inputs = activations['lstm_input'].cpu().detach()
    data_q = (act_error_quant()(inputs)*2**4).type(torch.int32)
    for i_batch, (audio, labels) in enumerate(test_data.get_data_loader(config.testing.test_size,shuffle=False)):
        gt = labels
    f = open('data_lstm.c', 'w') 
    f.write('int16_t input_data' + size_to_string(data_q) + '  = \n')
    d_flat = data_q.transpose(1,0).flatten()
    print(d_flat.shape)
    f.write(ndarray_to_string(d_flat))
    f.write(';\n')

    f.write('const int16_t ground_truth' + size_to_string(gt) + '  = \n')
    f.write(ndarray_to_string(gt))
    f.write(';\n')
    f.close()
    return rate, log


if __name__ == '__main__':
    config = TrainingConfig()
    parser = argparse.ArgumentParser("train a GRU based neural network for keyword spotting")
    # or you can just specify an options file.
    parser.add_argument("--config", help="Use json file containing all these options (as per 'training_config.py')")
    args = parser.parse_args()

    if args.config:
        config.load(args.config)

    test(config)
