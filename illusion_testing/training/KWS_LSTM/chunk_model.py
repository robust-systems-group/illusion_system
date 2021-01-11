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

def size_to_string(n):
    s = ''
    for i in n.shape:
        s+= '[' + str(i) + ']'
    return s

def ndarray_to_string(x):
    s = np.array2string(x, separator=',',threshold=2**32)
    s = s.replace('[', '{');
    s = s.replace(']', '}');
    return s;


weights = np.load('model.npz')

weights_i = np.split(weights['lstm1.cell.weight_i'].flatten(),4*8)
weights_h = np.split(weights['lstm1.cell.weight_h'].flatten(),4*8)
bias = np.split(weights['lstm1.cell.bias'].flatten(),4*8)

fc_weights = np.hsplit(weights['hidden2keyword.weight'],8)
fc_bias = weights['hidden2keyword.bias']

f = open('model_chunked_LSTM.c', 'w')
names = [[ 'lstm_i_H','lstm_h_H','lstm_B', 'fc_H', 'fc_B'],['const int8_t ', 'const int16_t layer', 'const int32_t layer']]
numpy_files = {}


for i in range(8):
    wi = np.concatenate(weights_i[i::8])
    wh = np.concatenate(weights_h[i::8])
    b = np.concatenate(bias[i::8])
    fc_w = fc_weights[i].flatten()

    f.write(names[1][0]+names[0][0] +"_"+str(i) + size_to_string(wi) + '  = \n')
    f.write(ndarray_to_string(wi))
    f.write(';\n')
    
    f.write(names[1][0]+names[0][1] +"_"+str(i) + size_to_string(wh) + '  = \n')
    f.write(ndarray_to_string(wh))
    f.write(';\n')
    
    f.write(names[1][0]+names[0][2] +"_"+str(i) + size_to_string(b) + '  = \n')
    f.write(ndarray_to_string(b))
    f.write(';\n')
    
    f.write(names[1][0]+names[0][3] +"_"+str(i) + size_to_string(fc_w) + '  = \n')
    f.write(ndarray_to_string(fc_w))
    f.write(';\n')
f.write(names[1][0]+names[0][3] + size_to_string(fc_bias) + '  = \n')
f.write(ndarray_to_string(fc_bias))
f.write(';\n')
f.close()

