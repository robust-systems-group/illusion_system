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


from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import ImageFolder
import os
from PIL import Image
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable
from qtorch import FixedPoint, FloatingPoint
from qtorch.quant import Quantizer,quantizer
from qtorch.optim import OptimLP
import pdb
from tqdm import tqdm
import torch_models

best_result = 0
batchSize = 128

test_data = np.load('data_MNIST.npz')
print(test_data['data'].shape)
data = torch.from_numpy(test_data['data'])
target = torch.from_numpy(test_data['labels'])

ubit_8 = FixedPoint(8, 4)
ubit_16 = FixedPoint(16, 8)

weight_quant = quantizer(forward_number=ubit_8, forward_rounding="nearest")
grad_quant = quantizer(forward_number=None, forward_rounding="nearest")
momentum_quant = quantizer(forward_number=None, forward_rounding="stochastic")
acc_quant = quantizer(forward_number=None, forward_rounding="stochastic")

act_error_quant = lambda : Quantizer(forward_number=ubit_8, backward_number=None,
                        forward_rounding="nearest", backward_rounding="stochastic")

act2_error_quant = lambda : Quantizer(forward_number=ubit_16, backward_number=None,
                        forward_rounding="nearest", backward_rounding="stochastic")

device = 'cpu' # torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch_models.MNIST_Net(act_error_quant, act2_error_quant)
model.load_state_dict(torch.load('./checkpoints/mnist_model_quant.pth'))
model = model.to(device=device)
model.eval()

activation = {}
def get_activation(name, side):
    def hook(model, input, output):
        if side==1: activation[name] = output#input#.detach()
        else: activation[name] = input#input#.detach()
    return hook

model.conv1.register_forward_hook(get_activation('conv1_in',0))
model.conv1.register_forward_hook(get_activation('conv1_out',1))
model.conv2.register_forward_hook(get_activation('conv2_in',0))
model.conv2.register_forward_hook(get_activation('conv2_out',1))
model.conv3.register_forward_hook(get_activation('conv3_in',0))
model.conv3.register_forward_hook(get_activation('conv3_out',1))
model.conv4.register_forward_hook(get_activation('conv4_in',0))
model.conv4.register_forward_hook(get_activation('conv4_out',1))
model.fc.register_forward_hook(get_activation('fc_in',0))
model.fc.register_forward_hook(get_activation('fc_out',1))



output = model(data)
pred = [np.argmax(i).item()for i in output.detach()]
np.savez('int_MNIST',
        conv1_in = activation['conv1_in'][0].detach(), 
        conv1_out = activation['conv1_out'][0].detach(), 
        conv2_in = activation['conv2_in'][0].detach(), 
        conv2_out = activation['conv2_out'][0].detach(), 
        conv3_in = activation['conv3_in'][0].detach(), 
        conv3_out = activation['conv3_out'][0].detach(), 
        conv4_in = activation['conv4_in'][0].detach(), 
        conv4_out = activation['conv4_out'][0].detach(), 
        fc_in = activation['fc_in'][0].detach(), 
        fc_out = activation['fc_out'][0].detach()) 

print(np.sum(np.array(pred) == target.numpy())/1000)
