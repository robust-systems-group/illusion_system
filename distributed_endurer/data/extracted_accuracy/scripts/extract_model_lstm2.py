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

array_list = []

def write_layer(layer_name, chip, start, length):
    array = remapped_memory[chip][start:start+length]
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

parser = argparse.ArgumentParser(description='Extract weights for all chips running LSTM.')
parser.add_argument('--prefix',type=str,help='folder to read from',default='~/home/illusion_testing/endurer_tests/lstm_de_c')
parser.add_argument('--input',type=str,help='file to read from',default='../../../fpga_endurer/memory_lstm2.csv')
parser.add_argument('--time',type=str,help='time prefix for teh zero/one file',default="DAY")
parser.add_argument('--output',type=str,help='file to write model to',default='../../../c_models_test/c_lstm2/model_chunked_LSTM.c')
parser.add_argument('--omask',type=str,help='file to write activation mask to',default='../../../c_models_test/c_lstm2/activation_mask.c')
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
       if len(memory[i])==M: # or (k+1)*16==M:
          #print("Finished parsing in-of-order memory of virtual chip",i)
          break
        
f.close()

memory = np.array(memory, dtype=np.int16)

zeros = [[] for i in range(N)]

for i in range(N):
   f = open(args.prefix + str(i) + "/" + str(runtimes[args.time]) + "trace_10c_0.txt",'r')

   for j in range(N):
       f.readline() # skip the "physical chip #" print
       for k in range(args.memory//16): #while True:
           row = f.readline()
           row = row.split(' ') # ',' for csv
           row = [int(x,16) for x in row]
           if ("no" in args.prefix): zeros[i].extend(row)  # phys chip 0 for NR
           elif (i==j): zeros[i].extend(row)  # wait for phys chip i for DE
       if "no" in args.prefix or i==j:
            f.close()
            break


forced_zeros_memory = np.array(zeros, dtype=np.int16)#.view(dtype=np.uint8)

ones = [[] for i in range(N)]

for i in range(N):
   f = open(args.prefix + str(i) + "/" + str(runtimes[args.time]) + "trace_10c_1.txt",'r')


   for j in range(N):
       f.readline() # skip the "physical chip #" print
       for k in range(args.memory//16): #while True:
           row = f.readline()
           row = row.split(' ') # ',' for csv
           row = [int(x,16) for x in row]
           if ("no" in args.prefix): ones[i].extend(row) # phys chip 0 for NR
           elif (i==j): ones[i].extend(row)
       if "no" in args.prefix or i==j:
            f.close()
            break


forced_ones_memory = np.array(ones, dtype=np.int16)#.view(dtype=np.uint8)

f.close()

# Randomized offset
#shifts = np.random.randint(0,M,(N,), dtype=np.int32)
if "no" in args.prefix: # no random offset for NR
    shifts = np.zeros((N,))
else:  #load from state dict
    shifts = []
    for i in range(N):
        f = open(args.prefix.replace("_clean","").replace("processed","raw") + str(i) + "/end_states_long.txt",'r')
        states = eval(f.readline())
        state = states[state_map[args.time]]
        shifts.append(state[3][i])
        f.close()

rolled_zeros_memory = np.roll(forced_zeros_memory, shifts, 0).view(dtype=np.uint8)
rolled_ones_memory = np.roll(forced_ones_memory, shifts, 0).view(dtype=np.uint8)

# Chip shuffling
forced_ones_memory = np.zeros((N,M*2),dtype=np.uint8)
forced_zeros_memory = np.zeros((N,M*2),dtype=np.uint8)

chips = np.random.permutation([i for i in range(N)])
if True:#"no" in args.prefix: # no chip shufffle for NR, and DE doesn't swap either
    chips = [i for i in range(N)] 
for i in range(N):
    forced_zeros_memory[i] = rolled_zeros_memory[chips[i]]
    forced_ones_memory[i] = rolled_ones_memory[chips[i]]


f = open(args.output,'w')

chip_counter = np.zeros((N,), dtype=np.int32) 
remapped_memory = memory.view(dtype=np.uint8)

names = [[ 'lstm_i_H_', 'lstm_h_H_', 'lstm_B_', 'fc_H_','fc_B'],['const int8_t ', 'const int16_t ', 'const int32_t ']]

sLHI = 400
sLHH = 3200
sLB = 40
sFH = 110
sFB = 11

for chip in range(8):   
    start = chip_counter[chip]
    write_layer(names[1][0]+names[0][0]+str(chip), chip, start, sLHI)
    chip_counter[chip] += sLHI
    start = chip_counter[chip]
    write_layer(names[1][0]+names[0][1]+str(chip), chip, start, sLHH)
    chip_counter[chip] += sLHH
    start = chip_counter[chip]
    write_layer(names[1][0]+names[0][2]+str(chip), chip, start, sLB)
    chip_counter[chip] += sLB
    start = chip_counter[chip]
    write_layer(names[1][0]+names[0][3]+str(chip), chip, start, sFH)
    chip_counter[chip] += sFH
    start = chip_counter[chip]

chip = 7 
start = chip_counter[chip]
write_layer(names[1][0]+names[0][4], chip, start, sFB)
chip_counter[chip] += sFB

#for array in sorted(array_list):
#    f.write(array)
    
'''
const int8_t lstm_i_H_0[400];
const int8_t lstm_h_H_0[3200];
const int8_t lstm_B_0[40];
const int8_t fc_H_0[110];
const int8_t lstm_i_H_1[400];
const int8_t lstm_h_H_1[3200];
const int8_t lstm_B_1[40];
const int8_t fc_H_1[110];
const int8_t lstm_i_H_2[400];
const int8_t lstm_h_H_2[3200];
const int8_t lstm_B_2[40];
const int8_t fc_H_2[110];
const int8_t lstm_i_H_3[400];
const int8_t lstm_h_H_3[3200];
const int8_t lstm_B_3[40];
const int8_t fc_H_3[110];
const int8_t lstm_i_H_4[400];
const int8_t lstm_h_H_4[3200];
const int8_t lstm_B_4[40];
const int8_t fc_H_4[110];
const int8_t lstm_i_H_5[400];
const int8_t lstm_h_H_5[3200]; 
const int8_t lstm_B_5[40];
const int8_t fc_H_5[110];
const int8_t lstm_i_H_6[400];
const int8_t lstm_h_H_6[3200];
const int8_t lstm_B_6[40];
const int8_t fc_H_6[110];
const int8_t lstm_i_H_7[400];
const int8_t lstm_h_H_7[3200];
const int8_t lstm_B_7[40];
const int8_t fc_H_7[110];
const int8_t fc_B[11];
'''
f.close()

f = open(args.omask, 'w')

for i in range(8):
    if "no" in args.prefix:
        start = int('3%d0'%i,16)*2
        write_layer_inner('int16_t chip%d_forced_zeros'%i,forced_zeros_memory[i,start:start+20].view(dtype=np.int16))
        write_layer_inner('int16_t chip%d_forced_ones'%i,forced_ones_memory[i,start:start+20].view(dtype=np.int16))
    else:
        # TODO parse out from file the chip map!!
        write_layer_inner('int16_t chip%d_forced_zeros'%i,forced_zeros_memory[i,-20:].view(dtype=np.int16))
        write_layer_inner('int16_t chip%d_forced_ones'%i,forced_ones_memory[i,-20:].view(dtype=np.int16))


f.close()


