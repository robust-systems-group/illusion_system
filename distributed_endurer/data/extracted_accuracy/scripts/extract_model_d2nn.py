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

#params_H = {"n1":[150,72], "n2":[1296,24,12288,3072,240], "n3":[3456,320], "q":[6912,128]}
#params_B = {"n1":[6,12], "n2":[12,24,128,24,10], "n3":[32,10], "q":[64,2]}
params_H = {"n1":[150,72], "n2":[1296,2592,[96*42,96*42,96*42,96*2],3072,240], "n3":[3456,320], "q":[[28*108,36*108],128]}
params_B = {"n1":[6,12], "n2":[12,24,[42,42,42,2],24,10], "n3":[32,10], "q":[[28,36],2]}

layer_to_chip = {"n1":[0,0], "n2":[3,3,[4,5,6,7],7,7],"n3":[2,2],"q":[[0,1],1]}

parser = argparse.ArgumentParser(description='Extract weights for all chips running D2NN.')
parser.add_argument('--prefix',type=str,help='folder to read from',default='~/home/illusion_testing/svhn_de')
parser.add_argument('--time',type=str,help='time prefix for teh zero/one file',default="DAY")
parser.add_argument('--input',type=str,help='file to read from',default='../../../fpga_endurer/memory_d2nn.csv')
parser.add_argument('--output',type=str,help='file to write model to',default='../../../c_models_test/c_d2nn/model.c')
parser.add_argument('--omask',type=str,help='file to write activation mask to',default='../../../c_models_test/c_d2nn/activation_mask.c')
parser.add_argument('--chips',type=int,help='number of chips',default=8)
parser.add_argument('--memory',type=int,help='size of RRAM per chip (in 16-bit increments)',default=2048)

args = parser.parse_args()

runtimes = {}
runtimes['ZERO']= 0*(3 if "d2nn" in args.prefix else 1)
runtimes['DAY'] = 6*(3 if "d2nn" in args.prefix else 1)
runtimes['WEEK'] = (42-6)*(3 if "d2nn" in args.prefix else 1)
runtimes['MONTH'] = (180-42)*(3 if "d2nn" in args.prefix else 1)
runtimes['YEAR'] =  (2190-180)*(3 if "d2nn" in args.prefix else 1)
runtimes['TENYEAR'] = (21900-2190)*(3 if "d2nn" in args.prefix else 1)
state_map = {}
state_map['ZERO']= 'DAY'
state_map['DAY'] = 'WEEK'
state_map['WEEK'] = 'MONTH'
state_map['MONTH'] = 'YEAR'
state_map['YEAR'] =  'TENYEAR'
state_map['TENYEAR'] = 'END'


filename = args.input
N = args.chips
M = args.memory
memory = [[] for i in range(N)]

f = open(filename,'r')

for i in range(N):
   #f.readline() # skip the "physical chip #" print
   while True:
       row = f.readline()
       row = row.split(',')
       row = [int(x,16) for x in row]
       memory[i].extend(row)
       if len(memory[i])==M:
          #print("Finished parsing in-of-order memory of virtual chip",i)
          break

memory = np.array(memory, dtype=np.int16)

f.close()

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

names = [[ '_H', '_B', '_BN_H', '_BN_B'],['const int8_t ', 'const int16_t ', 'const int32_t ']]

#ordering = ["n1","q","n2","n3"] # order of allocation
ordering = ["n1","n2","n3","q"] # order of printout

for node_type in ordering:
  for layer in range(len(params_H[node_type])):
    sH = params_H[node_type][layer]
    sB = params_B[node_type][layer]
    bias = []
    weights = []
    chip = layer_to_chip[node_type][layer]
    if type(chip) is list:
        for i in range(len(chip)):
            chip_ = chip[i] 
            sH_ = sH[i]
            sB_ = sB[i]
            start = chip_counter[chip_]

            array = remapped_memory[chip_][start:start+sH_]
            if node_type=="n1" or (node_type=="n2" and layer<2):
                array <<= 3
            else:
                array <<= 4
            if "no" not in args.prefix:
                array &= forced_ones_memory[chip_][start:start+sH_]  # forced ones, anything that is 0 is stuck at 0 so AND
                array |= forced_zeros_memory[chip_][start:start+sH_] # forced zeros, anything that is 1 is stuck at 1 so OR
            weights.append(array)

            array = remapped_memory[chip_][start+sH_:start+sH_+sB_] 
            if "no" not in args.prefix:  
                array &= forced_ones_memory[chip_][start+sH_:start+sH_+sB_]  # forced ones, anything that is 0 is stuck at 0 so AND
                array |= forced_zeros_memory[chip_][start+sH_:start+sH_+sB_] # forced zeros, anything that is 1 is stuck at 1 so OR
            bias.append(array)
            chip_counter[chip_] += sH_ + sB_ 
    else:
        start = chip_counter[chip]
        array = remapped_memory[chip][start:start+sH]
        if node_type=="n1" or (node_type=="n2" and layer<2):
            array <<= 3
        else:
            array <<= 4
        if "no" not in args.prefix:
            array &= forced_ones_memory[chip][start:start+sH]  # forced ones, anything that is 0 is stuck at 0 so AND
            array |= forced_zeros_memory[chip][start:start+sH] # forced zeros, anything that is 1 is stuck at 1 so OR
        weights.append(array)

        array = remapped_memory[chip][start+sH:start+sH+sB] 
        if "no" not in args.prefix: 
            array &= forced_ones_memory[chip][start+sH:start+sH+sB]  # forced ones, anything that is 0 is stuck at 0 so AND
            array |= forced_zeros_memory[chip][start+sH:start+sH+sB] # forced zeros, anything that is 1 is stuck at 1 so OR
        bias.append(array)
        chip_counter[chip] += sH + sB 
    write_layer_inner(names[1][0]+node_type.upper()+"_layer"+str(layer+1)+names[0][0], np.concatenate(weights).view(dtype=np.int8))
    write_layer_inner(names[1][0]+node_type.upper()+"_layer"+str(layer+1)+names[0][1], np.concatenate(bias).view(dtype=np.int8))



''' 
extern const int8_t N1_layer1_H[150];
extern const int8_t N1_layer1_B[6];
extern const int8_t N1_layer2_H[72];
extern const int8_t N1_layer2_B[12];
extern const int8_t N2_layer1_H[1296];
extern const int8_t N2_layer1_B[12];
extern const int8_t N2_layer2_H[2592];
extern const int8_t N2_layer2_B[24];
extern const int8_t N2_layer3_H[12288];
extern const int8_t N2_layer3_B[128];
extern const int8_t N2_layer4_H[3072];
extern const int8_t N2_layer4_B[24];
extern const int8_t N2_layer5_H[240];
extern const int8_t N2_layer5_B[10];
extern const int8_t N3_layer1_H[3456];
extern const int8_t N3_layer1_B[32];
extern const int8_t N3_layer2_H[320]; 
extern const int8_t N3_layer2_B[10];
extern const int8_t Q_layer1_H[6912];
extern const int8_t Q_layer1_B[64];
extern const int8_t Q_layer2_H[128];
extern const int8_t Q_layer2_B[2];
 

'''

f.close()



f = open(args.omask, 'w')
if "no" in args.prefix:
    write_layer_inner('int8_t n1_conv2_forced_zeros',forced_zeros_memory[0,1024*2:1024*2+12*6*6].view(dtype=np.int8))
    write_layer_inner('int8_t n1_conv2_forced_ones',forced_ones_memory[0,1024*2:1024*2+12*6*6].view(dtype=np.int8))
else:
    write_layer_inner('int8_t n1_conv2_forced_zeros',forced_zeros_memory[0,-12*6*6:].view(dtype=np.int8))
    write_layer_inner('int8_t n1_conv2_forced_ones',forced_ones_memory[0,-12*6*6:].view(dtype=np.int8))

f.close()


