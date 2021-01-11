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


def train(config, evaluate_only=False, outdir=".", detail=False, azureml=False):

    filename = config.model.filename
    categories_file = config.dataset.categories
    wav_directory = config.dataset.path
    batch_size = config.training.batch_size
    hidden_units = config.model.hidden_units
    architecture = config.model.architecture
    num_layers = config.model.num_layers
    use_gpu = config.training.use_gpu
    refine = config.training.refine
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
    if not evaluate_only:
        print("Loading {}...".format(training_file))
        training_data = AudioDataset(training_file, config.dataset, keywords)

        print("Loading {}...".format(validation_file))
        validation_data = AudioDataset(validation_file, config.dataset, keywords)

        if training_data.mean is not None:
            fname = os.path.join(outdir, "mean.npy")
            print("Saving {}".format(fname))
            np.save(fname, training_data.mean)
            fname = os.path.join(outdir, "std.npy")
            print("Saving {}".format(fname))
            np.save(fname, training_data.std)

            # use the training_data mean and std variation
            #test_data.mean = training_data.mean
            #test_data.std = training_data.std
            #validation_data.mean = training_data.mean
            #validation_data.std = training_data.std

        print("Training model {}".format(filename))
        if refine is True:
            msg = "Loading trained model with input size {}, hidden units {} and num keywords {}"
            print(msg.format(test_data.input_size, hidden_units, test_data.num_keywords))
            model = create_model(config.model, test_data.input_size, test_data.num_keywords)
            model.load_state_dict(torch.load(filename))
        else: 
            model = create_model(config.model, training_data.input_size, training_data.num_keywords)
        if device.type == 'cuda':
            model.cuda()  # move the processing to GPU

        start = time.time()
        log = model.fit(training_data, validation_data, config.training, config.model, device, detail, run)
        end = time.time()

        passed, total, rate = model.evaluate(training_data, batch_size, device)
        print("Training accuracy = {:.3f} %".format(rate * 100))

        torch.save(model.state_dict(), os.path.join(outdir, filename))

    print("Evaluating {} keyword spotter using {} rows of featurized test audio...".format(
          architecture, test_data.num_rows))
    if model is None:
        msg = "Loading trained model with input size {}, hidden units {} and num keywords {}"
        print(msg.format(test_data.input_size, hidden_units, test_data.num_keywords))
        model = create_model(config.model, test_data.input_size, test_data.num_keywords)
        model.load_state_dict(torch.load(filename))
        if model and device.type == 'cuda':
            model.cuda()  # move the processing to GPU

    results_file = os.path.join(outdir, "results.txt")
    passed, total, rate = model.evaluate(test_data, batch_size, device, results_file)
    print("Testing accuracy = {:.3f} %".format(rate * 100))

    if not evaluate_only:
        #name = os.path.splitext(filename)[0] + ".onnx"
        #print("saving onnx file: {}".format(name))
        #model.export(os.path.join(outdir, name), device)

        config.dataset.sample_rate = test_data.sample_rate
        config.dataset.input_size = test_data.audio_size
        config.dataset.num_filters = test_data.input_size
        config.dataset.window_size = test_data.window_size
        config.dataset.shift = test_data.shift

        logdata = {
            "accuracy_val": rate,
            "training_time": end - start,
            "log": log
        }
        d = TrainingConfig.to_dict(config)
        logdata.update(d)

        logname = os.path.join(outdir, "train_results.json")
        save_json(logdata, logname)

    return rate, log


def str2bool(v):
    if v is None:
        return False
    lower = v.lower()
    return lower in ["t", "1", "true", "yes"]


if __name__ == '__main__':
    config = TrainingConfig()
    parser = argparse.ArgumentParser("train a GRU based neural network for keyword spotting")

    # all the training parameters
    parser.add_argument("--epochs", help="Number of epochs to train", type=int)
    parser.add_argument("--lr_scheduler", help="Type of learning rate scheduler (None, TriangleLR, CosineAnnealingLR,"
                                               " ExponentialLR, ExponentialResettingLR)")
    parser.add_argument("--learning_rate", help="Default learning rate, and maximum for schedulers", type=float)
    parser.add_argument("--lr_min", help="Minimum learning rate for the schedulers", type=float)
    parser.add_argument("--lr_peaks", help="Number of peaks for triangle and cosine schedules", type=float)
    parser.add_argument("--batch_size", "-bs", help="Batch size of training", type=int)
    parser.add_argument("--architecture", help="Specify model architecture (GRU, LSTM, FastGRNN)")
    parser.add_argument("--num_layers", type=int, help="Number of RNN layers (1, 2 or 3)")
    parser.add_argument("--hidden_units", "-hu", type=int, help="Number of hidden units in the GRU layers")
    parser.add_argument("--use_gpu", help="Whether to use GPU for training", action="store_true")
    parser.add_argument("--refine", help="Whether to refine model via additional training", action="store_true")
    parser.add_argument("--normalize", help="Whether to normalize audio dataset", action="store_true")
    parser.add_argument("--rolling", help="Whether to train model in rolling fashion or not", action="store_true")
    parser.add_argument("--max_rolling_length", help="Max number of epochs you want to roll the rolling training"
                        " default is 100", type=int)

    # arguments for fastgrnn
    parser.add_argument("--wRank", "-wr", help="Rank of W in FastGRNN default is None", type=int)
    parser.add_argument("--uRank", "-ur", help="Rank of U in FastGRNN default is None", type=int)
    parser.add_argument("--gate_nonlinearity", "-gnl", help="Gate Non-Linearity in FastGRNN default is sigmoid"
                        " use between [sigmoid, quantSigmoid, tanh, quantTanh]")
    parser.add_argument("--update_nonlinearity", "-unl", help="Update Non-Linearity in FastGRNN default is Tanh"
                        " use between [sigmoid, quantSigmoid, tanh, quantTanh]")

    # or you can just specify an options file.
    parser.add_argument("--config", help="Use json file containing all these options (as per 'training_config.py')")

    # and some additional stuff ...
    parser.add_argument("--azureml", help="Tells script we are running in Azure ML context")
    parser.add_argument("--eval", "-e", help="No training, just evaluate existing model", action='store_true')
    parser.add_argument("--filename", "-o", help="Name of model file to generate")
    parser.add_argument("--categories", "-c", help="Name of file containing keywords")
    parser.add_argument("--dataset", "-a", help="Path to the audio folder containing 'training.npz' file")
    parser.add_argument("--outdir", help="Folder in which to store output file and log files")
    parser.add_argument("--detail", "-d", help="Save loss info for every iteration not just every epoch",
                        action="store_true")
    args = parser.parse_args()

    if args.config:
        config.load(args.config)

    azureml = str2bool(args.azureml)

    # then any user defined options overrides these defaults
    if args.epochs:
        config.training.max_epochs = args.epochs
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.lr_min:
        config.training.lr_min = args.lr_min
    if args.lr_peaks:
        config.training.lr_peaks = args.lr_peaks
    if args.lr_scheduler:
        config.training.lr_scheduler = args.lr_scheduler
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.rolling:
        config.training.rolling = args.rolling
    if args.max_rolling_length:
        config.training.max_rolling_length = args.max_rolling_length
    if args.architecture:
        config.model.architecture = args.architecture
    if args.num_layers:
        config.model.num_layers = args.num_layers
    if args.hidden_units:
        config.model.hidden_units = args.hidden_units
    if args.filename:
        config.model.filename = args.filename
    if args.use_gpu:
        config.training.use_gpu = args.use_gpu
    if args.normalize:
        config.dataset.normalize = args.normalize
    if args.categories:
        config.dataset.categories = args.categories
    if args.dataset:
        config.dataset.path = args.dataset
    if args.wRank:
        config.model.wRank = args.wRank
    if args.uRank:
        config.model.uRank = args.wRank
    if args.gate_nonlinearity:
        config.model.gate_nonlinearity = args.gate_nonlinearity
    if args.update_nonlinearity:
        config.model.update_nonlinearity = args.update_nonlinearity
    if args.refine:
        config.training.refine = True

    if not os.path.isfile("config.json"):
        config.save("config.json")

    train(config, args.eval, args.outdir, args.detail, azureml)
