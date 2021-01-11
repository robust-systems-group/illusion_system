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


import matplotlib.pyplot as plt
import numpy as np

N = 8
M = 2048

def get_failures(array, expected):
    failures = np.zeros((array.shape[0],16))
    for i in range(array.shape[0]):
        for j in range(16):
            if int(get_bit(array[i],j))!=expected:
                failures[i,j] = 1

    return failures.reshape(-1)

def get_bit(uint16_word, index):
    word = np.uint16(uint16_word) # force just in case
    s = bin(word)[2:]
    if len(s)<16:
        s = "0"*(16-len(s))+s
    return s[index]

def get_masks(prefix,time="19710"):
    zeros = [[] for i in range(N)]
    for i in range(N):
       if (i==5 and "no" in prefix):
           f = open(prefix + str(i) + "_v2/" + time + "trace_10c_0.txt",'r')
       else:
           f = open(prefix + str(i) + "/" + time + "trace_10c_0.txt",'r')

       for j in range(N):
           f.readline() # skip the "physical chip #" print
           for k in range(M/16): #while True:
               row = f.readline()
               row = row.split(' ') # ',' for csv
               row = [int(x,16) for x in row]
               if ("no" in prefix): zeros[i].extend(row)  # phys chip 0 for NR
               elif (i==j): zeros[i].extend(row)  # wait for phys chip i for DE
           if "no" in prefix or i==j:
                f.close()
                break
    forced_zeros_memory = np.array(zeros, dtype=np.int16).view(dtype=np.uint16)

    ones = [[] for i in range(N)]
    for i in range(N):
       if (i==5 and "no" in prefix):
           f = open(prefix + str(i) + "_v2/" + time + "trace_10c_1.txt",'r')
       else:
           f = open(prefix + str(i) + "/" + time + "trace_10c_1.txt",'r')

       for j in range(N):
           f.readline() # skip the "physical chip #" print
           for k in range(M/16): #while True:
               row = f.readline()
               row = row.split(' ') # ',' for csv
               row = [int(x,16) for x in row]
               if ("no" in prefix): ones[i].extend(row)
               elif (i==j): ones[i].extend(row)
           if "no" in prefix or i==j:
                f.close()
                break
    forced_ones_memory = np.array(ones, dtype=np.int16).view(dtype=np.uint16)
    return forced_ones_memory, forced_zeros_memory

bit_failures = np.zeros((8,2048*16))

start_addr_tmp = 0
data_length = M
fig, axs = plt.subplots(2,8,sharex=True,sharey=True,figsize=(14,10))

cmap = "gnuplot" #"brg"#"gnuplot"


start_addr_tmp = 10#0x300
data_length = 10 #10

mask_ones = np.zeros((8,2048*2),dtype=np.int16)
mask_zeros = np.zeros((8,2048*2),dtype=np.int16)

f = open("../c_models_test/c_lstm2/activation_mask_NR_TENYEAR.c","r")
for i in range(8):
    f.readline()
    mask_zeros[i,10:10+data_length] = eval("["+f.readline()[1:-3]+"]")
    f.readline()
    mask_ones[i,10:10+data_length] = eval("["+f.readline()[1:-3]+"]")
f.close() 
mask_ones = mask_ones.view(dtype=np.int16)
mask_zeros = mask_zeros.view(dtype=np.int16)

for i in range(8):
  set_failures = get_failures(mask_ones[i,start_addr_tmp:start_addr_tmp+data_length],1)
  reset_failures = get_failures(mask_zeros[i,start_addr_tmp:start_addr_tmp+data_length],0)
  bit_failures[i,100*16:100*16+16*data_length] = np.logical_or(set_failures,reset_failures)
  axs[0,i].imshow(1-bit_failures[i].reshape((128,16*16)),extent=(0,16*16,0,128),aspect="auto", cmap=cmap ,vmin=0,vmax=1)

start_addr_tmp = 0
data_length = M

bit_failures = np.zeros((8,2048*16))


mask_ones,mask_zeros = get_masks("../data/processed/lstm_de_clean_c")
mask_ones0,mask_zeros0 = get_masks("../data/processed/lstm_de_clean_c","0")
for i in range(8):
  set_failures = get_failures(mask_ones[i,start_addr_tmp:start_addr_tmp+data_length],1)
  reset_failures = get_failures(mask_zeros[i,start_addr_tmp:start_addr_tmp+data_length],0)
  bit_failures[i,0:16*data_length] = np.logical_or(set_failures,reset_failures)
  set_failures = get_failures(mask_ones0[i,start_addr_tmp:start_addr_tmp+data_length],1)
  reset_failures = get_failures(mask_zeros0[i,start_addr_tmp:start_addr_tmp+data_length],0)
  bit_failures[i,0:16*data_length] -= np.logical_or(set_failures,reset_failures)
  axs[1,i].imshow(1-bit_failures[i].reshape((128,16*16)),extent=(0,16*16,0,128),aspect="auto", cmap=cmap,vmin=0,vmax=1)

for i in range(8):
    axs[1,i].set_xlabel("Chip " +str(i),fontsize=18)
    for j in range(2):
      axs[j,i].set_xticks(np.arange(0,16*16,4*16))
      #axs[j,i].set_xticklabels(np.arange(0,16*16,4*16))
      #axs[j,i].set_xlim((0,16))
axs[0,0].set_ylabel("Without any\nResiliency Technique",fontsize=18)
axs[1,0].set_ylabel("With\nDistributed ENDURER",fontsize=18)
fig.suptitle("Permanent Bit Failures - 10-Years: LSTM",fontsize=28)
plt.show()

