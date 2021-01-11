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
import numpy as np
from collections import OrderedDict


def write_layer(layer_name, chip, start, length):
    array = remapped_memory[chip][start:start+length]
    if "_B" not in layer_name:
        if "1" in layer_name or "2" in layer_name or "3" in layer_name:
            array <<= 2 # shift weights
        else: #FC
            array <<= 4 # shift weights
    if "no" not in args.prefix:
        array &= forced_ones_memory[chip][start:start+length]  # forced ones, anything that is 0 is stuck at 0 so AND
        array |= forced_zeros_memory[chip][start:start+length] # forced zeros, anything that is 1 is stuck at 1 so OR
    write_layer_inner(layer_name, array.view(dtype=np.int8))

def write_layer_inner(layer_name, array):
    f.write(layer_name + size_to_string(array) + "  = \n")
    f.write(ndarray_to_string(array) + ";\n")

def size_to_string(n):
    return '[' + str(n.shape[0]) + ']' # assume 1-dim array

def ndarray_to_string(x):
    s = np.array2string(x, separator=',',threshold=2**32)
    s = s.replace('[', '{');
    s = s.replace(']', '}');
    return s;

params_H = {1:486, 2:2916, 3:3888, 4: {'a':4004,'b':4004,'c':4004,'d':4004,'e':3952},5:3120,6:600} 
params_B = {1:18, 2:18, 3:24, 4:52, 5:60, 6:10}

layer_to_chip = {1:0, 2:0, 3:1, 4:{'a':2, 'b':3,'c':4,'d':5,'e':6}, 5:7, 6:7}

parser = argparse.ArgumentParser(description='Extract weights for all chips running SVHN.')
parser.add_argument('--prefix',type=str,help='folder to read from',default='~/home/illusion_testing/svhn_de')
parser.add_argument('--input',type=str,help='file to read from',default='../../../fpga_endurer/memory_svhn.csv')
parser.add_argument('--time',type=str,help='time prefix for teh zero/one file',default="DAY")
parser.add_argument('--output',type=str,help='file to write model to',default='../../../c_models_test/c_svhn/model.c')
parser.add_argument('--omask',type=str,help='file to write activation mask to',default='../../../c_models_test/c_svhn/activation_mask.c')
parser.add_argument('--chips',type=int,help='number of chips',default=8)
parser.add_argument('--memory',type=int,help='size of RRAM per chip (in 16-bit increments)',default=2048)

runtimes = {}
runtimes['ZERO']= 0
runtimes['DAY'] = 6
runtimes['WEEK'] = 42-6
runtimes['MONTH'] = 180-42
runtimes['YEAR'] =  2190-180
runtimes['TENYEAR'] = 21900-2190
state_map = {}
state_map['ZERO']= 'DAY'
state_map['DAY'] = 'WEEK'
state_map['WEEK'] = 'MONTH'
state_map['MONTH'] = 'YEAR'
state_map['YEAR'] =  'TENYEAR'
state_map['TENYEAR'] = 'END'



args = parser.parse_args()
filename = args.input
N = args.chips
M = args.memory
memory = [[] for i in range(N)]

f = open(filename,'r')

for i in range(N):
   #f.readline() # skip the "physical chip #" print
   for k in range(args.memory//16): #while True:
       row = f.readline()
       row = row.split(',') # ',' for csv
       row = [int(x,16) for x in row]
       memory[i].extend(row)
       if len(memory[i])==M:
          #print("Finished parsing in-of-order memory of virtual chip",i)
          break
        
f.close()

memory = np.array(memory, dtype=np.int16)

zeros = [[] for i in range(N)]

f = open(args.prefix + "/" + str(runtimes[args.time]) + "trace_10c_0.txt",'r')

for i in range(N):
   f.readline() # skip the "physical chip #" print
   for k in range(args.memory//16): #while True:
       row = f.readline()
       row = row.split(' ') # ',' for csv
       row = [int(x,16) for x in row]
       zeros[i].extend(row)
       if len(zeros[i])==M:
          #print("Finished parsing in-of-order memory of virtual chip",i)
          break
        
f.close()


forced_zeros_memory = np.array(zeros, dtype=np.int16)#.view(dtype=np.uint8)

ones = [[] for i in range(N)]

f = open(args.prefix + "/" + str(runtimes[args.time]) + "trace_10c_1.txt",'r')

for i in range(N):
   f.readline() # skip the "physical chip #" print
   for k in range(args.memory//16): #while True:
       row = f.readline()
       row = row.split(' ') # ',' for csv
       row = [int(x,16) for x in row]
       ones[i].extend(row)
       if len(ones[i])==M: 
          #print("Finished parsing in-of-order memory of virtual chip",i)
          break
        
f.close()

forced_ones_memory = np.array(ones, dtype=np.int16)#.view(dtype=np.uint8)


# Randomized offset
#shifts = np.random.randint(0,M,(N,), dtype=np.int32)
if "no" in args.prefix: # no random offset for NR
    shifts = np.zeros((N,))
else:  #load from state dict
    f = open(args.prefix.replace("_clean","").replace("processed","raw") + "/end_states_long.txt",'r')
    states = eval(f.readline())
    state = states[state_map[args.time]]
    shifts = state[3]
    f.close()


rolled_zeros_memory = np.roll(forced_zeros_memory, shifts, 0).view(dtype=np.uint8)
rolled_ones_memory = np.roll(forced_ones_memory, shifts, 0).view(dtype=np.uint8)

# Chip shuffling
forced_ones_memory = np.zeros((N,M*2),dtype=np.uint8)
forced_zeros_memory = np.zeros((N,M*2),dtype=np.uint8)

#chips = np.random.permutation([i for i in range(N)])
if "no" in args.prefix: # no chip shufffle for NR
    chips = [i for i in range(N)]
else:
    chips = state[2]
for i in range(N):
    forced_zeros_memory[i] = rolled_zeros_memory[chips[i]]
    forced_ones_memory[i] = rolled_ones_memory[chips[i]]

f = open(args.output,'w')

chip_counter = np.zeros((N,), dtype=np.int32) 
remapped_memory = memory.view(dtype=np.uint8)

names = [[ '_H', '_B', '_BN_H', '_BN_B'],['const int8_t layer', 'const int16_t layer', 'const int32_t layer']]

for layer in range(1,7):
    sH = params_H[layer]
    sB = params_B[layer]
    if type(sH) is dict:
        for part in 'abcde': #layer_to_chip[layer]:
            chip = layer_to_chip[layer][part]
            start = chip_counter[chip]
            write_layer(names[1][0]+str(layer)+part+names[0][0], chip, start, sH[part])
            chip_counter[chip] += sH[part]
        
        #assume bias is on last chip which has this layer    
        start = chip_counter[chip]
        write_layer(names[1][0]+str(layer)+names[0][1], chip, start, sB)
        chip_counter[chip] += sB 

    else:
        chip = layer_to_chip[layer]
        start = chip_counter[chip]
        write_layer(names[1][0]+str(layer)+names[0][0], chip, start, sH)
        chip_counter[chip] += sH
        start = chip_counter[chip]
        write_layer(names[1][0]+str(layer)+names[0][1], chip, start, sB)
        chip_counter[chip] += sB 


''' 
const int8_t layer1_H[486]
const int8_t layer1_B[18]
const int8_t layer2_H[2916]
const int8_t layer2_B[18]
const int8_t layer3_H[3888]
const int8_t layer3_B[24]  = 
const int8_t layer4a_H[4004]  = 
const int8_t layer4b_H[4004]  = 
const int8_t layer4c_H[4004]  = 
const int8_t layer4d_H[4004]  = 
const int8_t layer4e_H[3952]  = 
const int8_t layer4_B[52]  = 
const int8_t layer5_H[3120]  = 
const int8_t layer5_B[60]  = 
const int8_t layer6_H[600]  = 
const int8_t layer6_B[10]




'''

f.close()


f = open(args.omask, 'w')

c=0
# masks organized by virtual id now
#if "de" in args.prefix:
#    if args.time=="TENYEAR":
#        c=6

write_layer_inner('int8_t conv1_forced_zeros',forced_zeros_memory[c,-512*2//8:].view(dtype=np.int8))
write_layer_inner('int8_t conv1_forced_ones',forced_ones_memory[c,-512*2//8:].view(dtype=np.int8))

f.close()

'''
int8_t conv1_forced_zeros[2*8*8] = {0};
 
int8_t conv1_forced_ones[2*8*8] = {
 0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
 0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
 0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
 0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
 0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
 0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
 0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
 0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
 0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
 0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
 0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
 0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
 0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
 0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
 0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
 0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF
};
'''


