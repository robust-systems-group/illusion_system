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
import numpy as np
import os
from qtorch.quant import fixed_point_quantize
import pdb
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms

path = os.path.join('./data')
batchSize = 1000
test_loader  = torch.utils.data.DataLoader(datasets.MNIST(root=path,train=False,transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.5,), (0.5,))
                                                          ])), batch_size=batchSize, shuffle=False)

print('loading data\n')
wl_w = 8
fl_w = 4
wl_b = 8
fl_b = 4
names = [[ '_H', '_B', '_BN_H', '_BN_B'],['const int8_t layer', 'const int16_t layer', 'const int32_t layer']]


print('Load previous weights\n')
state_dict_n = torch.load('./checkpoints/mnist_d2nn_quant.pth')
state_dict_q = torch.load('./checkpoints/mnist_d2nn_q_quant.pth')
state_dicts = [state_dict_n, state_dict_q]
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

print("Saving Weights...")
f = open('model_MNIST_D2NN.c', 'w')
j = 0;
for state_dict in state_dicts:
    for key in state_dict.keys():
        if ('conv' in key):
            if 'weight' in key:
                w1 = (fixed_point_quantize(state_dict[key],wl_w,fl_w,rounding="nearest")*2**fl_w).type(torch.int32)
                w1 = w1.cpu()
                w1 = w1.flatten()
                f.write(names[1][0]+str(j+1)+names[0][0] + size_to_string(w1) + '  = \n')
                f.write(ndarray_to_string(w1))
                f.write(';\n')
            elif 'bias' in key:
                b1 = (fixed_point_quantize(state_dict[key],wl_b,fl_b,rounding="nearest")*2**fl_b).type(torch.int32)
                b1 = b1.cpu()
                f.write(names[1][0]+str(j+1)+names[0][1] + size_to_string(b1) + '  = \n')
                f.write(ndarray_to_string(b1))
                f.write(';\n')
                j += 1
        if ('fc' in key):
            if 'weight' in key:
                w1 = (fixed_point_quantize(state_dict[key],wl_w,fl_w,rounding="nearest")*2**fl_w).type(torch.int32)
                w1 = w1.cpu()
                chunk = w1.chunk(3,dim=1)
                w1a = chunk[0].flatten()
                w1b = chunk[1].flatten()
                w1c = chunk[2].flatten()
                f.write(names[1][0]+str(j+1)+names[0][0] + size_to_string(w1) + '  = \n')
                f.write(ndarray_to_string(w1a))
                f.write(';\n')
                f.write(names[1][0]+str(j+1)+names[0][0] + size_to_string(w1) + '  = \n')
                f.write(ndarray_to_string(w1b))
                f.write(';\n')
                f.write(names[1][0]+str(j+1)+names[0][0] + size_to_string(w1) + '  = \n')
                f.write(ndarray_to_string(w1c))
                f.write(';\n')
            elif 'bias' in key:
                b1 = (fixed_point_quantize(state_dict[key],wl_b,fl_b,rounding="nearest")*2**fl_b).type(torch.int32)
                b1 = b1.cpu()
                f.write(names[1][0]+str(j+1)+names[0][1] + size_to_string(b1) + '  = \n')
                f.write(ndarray_to_string(b1))
                f.write(';\n')
                j += 1
f.close()

print("Saving Input Data......")
i = 0
for data, target in test_loader:
    data_q = (fixed_point_quantize(data,wl_w,fl_w,rounding="nearest")*2**fl_w).type(torch.int32)
    #pdb.set_trace()
    gt = target
    data_set = data
    break

np.savez('data_MNIST_D2NN',data=data_set,labels=gt)
data_q = data_q.view([-1,])
f = open('data_mnist_D2NN.c', 'w') 
f.write('int8_t input_data' + size_to_string(data_q) + '  = \n')
f.write(ndarray_to_string(data_q))
f.write(';\n')
f.write('const int8_t ground_truth' + size_to_string(gt) + '  = \n')
f.write(ndarray_to_string(gt))
f.write(';\n')
f.close()

print("Done!")
