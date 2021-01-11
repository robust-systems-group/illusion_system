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

parser = argparse.ArgumentParser(description='Analyze bit failures')
parser.add_argument('--prefix',type=str,help='folder to read from',default='/rsgs/pool0/zainabk/illusion_testing/endurer_tests/svhn_de')
parser.add_argument('--retry',type=int,help='how many times we cycle',default=10)
parser.add_argument('--output',type=str,help='folder to processed traces to',default='/rsgs/pool0/zainabk/illusion_testing/endurer_tests/svhn_de_clean')
parser.add_argument('--chips',type=int,help='number of chips',default=8)
parser.add_argument('--memory',type=int,help='size of RRAM per chip (in 16-bit increments)',default=2048)

def count_ones(uint16_word):
    word = np.uint16(uint16_word) # force just in case
    return bin(word).count("1")

def get_bit(uint16_word, index):
    word = np.uint16(uint16_word) # force just in case
    s = bin(word)[2:]
    if len(s)<16:
        s = "0"*(16-len(s))+s
    return s[index]

def force(dict_, time, chip, addr, bit, val):
    if val==0:
        dict_[time][chip][addr] &= 65535 - (1 << (15-bit))
    else:
        dict_[time][chip][addr] |= 1 << (15-bit)
 
def force_word(dict_, time, chip, addr, val):
    if val==0:
        dict_[time][chip][addr] = 0
    else:
        dict_[time][chip][addr] = 65535
 
args = parser.parse_args()
N = args.chips
M = args.memory

runtimes = {}
runtimes['ZERO']= 0*(3 if "d2nn" in args.prefix else 1)
runtimes['DAY'] = 6*(3 if "d2nn" in args.prefix else 1)   
runtimes['WEEK'] = (42-6)*(3 if "d2nn" in args.prefix else 1)
runtimes['MONTH'] = (180-42)*(3 if "d2nn" in args.prefix else 1)
runtimes['YEAR'] =  (2190-180)*(3 if "d2nn" in args.prefix else 1)
runtimes['TENYEAR'] = (21900-2190)*(3 if "d2nn" in args.prefix else 1)


times = ["ZERO","DAY","WEEK","MONTH","YEAR","TENYEAR"]

F1 = {}
F0 = {}

for t,time in enumerate(times):

    zeros = [[] for i in range(N)]
    f = open(args.prefix + "/" + str(runtimes[time]) + "trace_1c_0.txt",'r')
    for i in range(N):
       f.readline() # skip the "physical chip #" print
       for k in range(args.memory//16): #while True:
           row = f.readline()
           row = row.split(' ') # ',' for csv
           row = [int(x,16) for x in row]
           zeros[i].extend(row)
    f.close()
    forced_zeros_memory = np.array(zeros, dtype=np.int16).view(dtype=np.uint16)
    
    zeros = [[] for i in range(N)]
    f = open(args.prefix + "/" + str(runtimes[time]) + "trace_10c_0.txt",'r')
    for i in range(N):
       f.readline() # skip the "physical chip #" print
       for k in range(args.memory//16): #while True:
           row = f.readline()
           row = row.split(' ') # ',' for csv
           row = [int(x,16) for x in row]
           zeros[i].extend(row)
    f.close()
    forced_zeros_memory2 = np.array(zeros, dtype=np.int16).view(dtype=np.uint16)
    forced_zeros_memory &= forced_zeros_memory2
    F0[time] = forced_zeros_memory



    ones = [[] for i in range(N)]
    f = open(args.prefix + "/" + str(runtimes[time]) + "trace_1c_1.txt",'r')
    for i in range(N):
       f.readline() # skip the "physical chip #" print
       for k in range(args.memory//16): #while True:
           row = f.readline()
           row = row.split(' ') # ',' for csv
           row = [int(x,16) for x in row]
           ones[i].extend(row)
           if len(ones[i])==M: 
              break
    f.close()
    forced_ones_memory = np.array(ones, dtype=np.int16).view(dtype=np.uint16)
    
    ones = [[] for i in range(N)]
    f = open(args.prefix + "/" + str(runtimes[time]) + "trace_10c_1.txt",'r')
    for i in range(N):
       f.readline() # skip the "physical chip #" print
       for k in range(args.memory//16): #while True:
           row = f.readline()
           row = row.split(' ') # ',' for csv
           row = [int(x,16) for x in row]
           ones[i].extend(row)
    f.close()
    forced_ones_memory2 = np.array(ones, dtype=np.int16).view(dtype=np.uint16)
    forced_ones_memory |= forced_ones_memory2
    F1[time] = forced_ones_memory

    broken_zeros = np.zeros((N,))
    broken_ones = np.zeros((N,))

    for i in range(N):
        for j in range(M):
            ones_ones = count_ones(forced_ones_memory[i][j])
            ones_zeros = count_ones(forced_zeros_memory[i][j])
            broken_zeros[i] += ones_zeros
            broken_ones[i] += (16-ones_ones)


    print("Prefix: "+args.prefix)
    print("Time: "+time)
    print("Retry force: "+str(args.retry))
    print("Total broken zeros: "+str(broken_zeros))
    print("Total broken ones: "+str(broken_ones))


if "de" in args.prefix:
    times.append("FINAL")
    time = "FINAL"
    runtimes["FINAL"] = "final"

    if "d2nn" in args.prefix:
        new_prefix = "d2nn_final"
    elif "svhn" in args.prefix:
        new_prefix = "svhn_final"
        runtimes["FINAL"] = 18  # since D2NN 18 used as final for SVHN
    elif "lstm" in args.prefix:
        new_prefix = "lstm_final"
        if "c0" in args.prefix:
            new_prefix = "lstm_final_c0"

    if "/" in args.prefix:
        new_prefix = "/".join(args.prefix.split("/")[:-1]) + "/" + new_prefix

    zeros = [[] for i in range(N)]
    f = open(new_prefix + "/" + str(runtimes[time]) + "trace_1c_0.txt",'r')
    for i in range(N):
       f.readline() # skip the "physical chip #" print
       for k in range(args.memory//16): #while True:
           row = f.readline()
           row = row.split(' ') # ',' for csv
           row = [int(x,16) for x in row]
           zeros[i].extend(row)
    f.close()
    forced_zeros_memory = np.array(zeros, dtype=np.int16).view(dtype=np.uint16)
    
    zeros = [[] for i in range(N)]
    f = open(new_prefix + "/" + str(runtimes[time]) + "trace_10c_0.txt",'r')
    for i in range(N):
       f.readline() # skip the "physical chip #" print
       for k in range(args.memory//16): #while True:
           row = f.readline()
           row = row.split(' ') # ',' for csv
           row = [int(x,16) for x in row]
           zeros[i].extend(row)
    f.close()
    forced_zeros_memory2 = np.array(zeros, dtype=np.int16).view(dtype=np.uint16)
    forced_zeros_memory &= forced_zeros_memory2

    if "lstm" in args.prefix and "c0" in args.prefix: # run on diff chip, diff place
        forced_zeros_memory[0] = forced_zeros_memory[2]

    F0[time] = forced_zeros_memory



    ones = [[] for i in range(N)]
    f = open(new_prefix + "/" + str(runtimes[time]) + "trace_1c_1.txt",'r')
    for i in range(N):
       f.readline() # skip the "physical chip #" print
       for k in range(args.memory//16): #while True:
           row = f.readline()
           row = row.split(' ') # ',' for csv
           row = [int(x,16) for x in row]
           ones[i].extend(row)
           if len(ones[i])==M: 
              break
    f.close()
    forced_ones_memory = np.array(ones, dtype=np.int16).view(dtype=np.uint16)
    
    ones = [[] for i in range(N)]
    f = open(new_prefix + "/" + str(runtimes[time]) + "trace_10c_1.txt",'r')
    for i in range(N):
       f.readline() # skip the "physical chip #" print
       for k in range(args.memory//16): #while True:
           row = f.readline()
           row = row.split(' ') # ',' for csv
           row = [int(x,16) for x in row]
           ones[i].extend(row)
    f.close()
    forced_ones_memory2 = np.array(ones, dtype=np.int16).view(dtype=np.uint16)
    forced_ones_memory |= forced_ones_memory2

    if "lstm" in args.prefix and "c0" in args.prefix:  # run on diff chip, diff place
        forced_ones_memory[0] = forced_ones_memory[2]

    F1[time] = forced_ones_memory

 

# PROCESS
'''
for time in times[::-1]:

    for i in range(N):
        for j in range(M):
 
           for k in range(16):
                # fix yield errors (any bits broken at t0) for entire word forever
                if get_bit(F0[time][i][j], k)=="1":
                    if get_bit(F0["ZERO"][i][j], k)=="1":
                        force_word(F0, time, i, j, 0)
                        force_word(F1, time, i, j, 1)
                        break
                if get_bit(F1[time][i][j], k)=="0":
                    if get_bit(F1["ZERO"][i][j], k)=="0":
                        force_word(F0, time, i, j, 0)
                        force_word(F1, time, i, j, 1)
                        break

           ''
           for k in range(8):
                # fix yield errors (any bits broken at t0
                if get_bit(F0[time][i][j], k)=="1":
                    if get_bit(F0["ZERO"][i][j], k)=="1":
                        force(F0, time, i, j, k, 0)
                if get_bit(F1[time][i][j], k)=="0":
                    if get_bit(F1["ZERO"][i][j], k)=="0":
                        force(F1, time, i, j, k, 1)
           '''
    

for t,time in enumerate(times):

  for endtime in times[t+1:]:
    for i in range(N):
        for j in range(M):
            for k in range(16):
                # ignore temporary failures if bit got better later
                if get_bit(F0[time][i][j], k)=="1":
                    if get_bit(F0[endtime][i][j], k)=="0":
                        force(F0, time, i, j, k, 0)
                        force(F1, time, i, j, k, 1)

                if get_bit(F1[time][i][j], k)=="0":
                    if get_bit(F1[endtime][i][j], k)=="1":
                        force(F0, time, i, j, k, 0)
                        force(F1, time, i, j, k, 1)

 
# SAVE

for t,time in enumerate(times):
    #F1[time] = F1[time].view(dtype=np.uint16)
    #F0[time] = F0[time].view(dtype=np.uint16)

    if time=="FINAL":
        continue

    f = open(args.output + "/" + str(runtimes[time]) + "trace_" + str(args.retry) + "c_0.txt",'w')

    for i in range(N):
       f.write("Phys chip %d\n"%i)
       for k in range(M//16):
           f.write("".join(["%04x "%j for j in F0[time][i][k*16:(k+1)*16]])[:-1]+"\n")
            
    f.close()


    f = open(args.output + "/" + str(runtimes[time]) + "trace_" + str(args.retry) + "c_1.txt",'w')

    for i in range(N):
       f.write("Phys chip %d\n"%i)
       for k in range(M//16):
           f.write("".join(["%04x "%j for j in F1[time][i][k*16:(k+1)*16]])[:-1]+"\n")
            
    f.close()




