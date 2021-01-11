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
    s = np.array2string(x, separator=',', threshold=2**32)
    s = s.replace('[', '{');
    s = s.replace(']', '}');
    return s;

f = open('model.c','r')

weights = [];
lines = f.readlines()

for line in lines:
    i = -1
    if 'int16_t' in line:
        weights.append([])
        i +=1
        continue
    line_items = "".join(line.split()).rstrip(',').split(',')
    items_out = []
    for item in line_items:
        if item == "};":
            continue
        else:
            weights[i].append(int(item.strip()))
np_array = []
for wlist in weights:
    np_array.append(np.array(wlist))
#print(np_array)
f.close()
weights = np_array

weights_i = np.split(weights[0],4*8)
weights_h = np.split(weights[1],4*8)
bias = np.split(weights[2],4*8)

fc_weights = np.hsplit(weights[3].reshape([11,56]),8)
fc_bias = weights[4]

f = open('model_chunked_LSTM.c', 'w')
names = [[ 'lstm_i_H','lstm_h_H','lstm_B', 'fc_H'],['const int8_t layer', 'const int16_t layer', 'const int32_t layer']]
numpy_files = {}

print(fc_weights)
for i in range(8):
    wi = np.concatenate(weights_i[i::8])
    wh = np.concatenate(weights_h[i::8])
    b = np.concatenate(bias[i::8])
    fc_w = fc_weights[i].flatten()

    f.write(names[1][1]+names[0][0] +"_"+str(i) + size_to_string(wi) + '  = \n')
    f.write(ndarray_to_string(wi))
    f.write(';\n')
    
    f.write(names[1][1]+names[0][1] +"_"+str(i) + size_to_string(wh) + '  = \n')
    f.write(ndarray_to_string(wh))
    f.write(';\n')
    
    f.write(names[1][1]+names[0][2] +"_"+str(i) + size_to_string(b) + '  = \n')
    f.write(ndarray_to_string(b))
    f.write(';\n')
    
    f.write(names[1][1]+names[0][3] +"_"+str(i) + size_to_string(fc_w) + '  = \n')
    f.write(ndarray_to_string(fc_w))
    f.write(';\n')
f.close()

