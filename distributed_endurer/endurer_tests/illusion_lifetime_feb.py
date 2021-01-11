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


import numpy as np
import string
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('workload', type=str, nargs='?',
                     help='which workload to simulate')
args = parser.parse_args()

appropriate_options = ["d2nn", "svhn", "lstm"]

if args.workload is not None and args.workload.lower() in appropriate_options:
    workload = args.workload.lower()
else: workload = 'd2nn'

print("Simulating for workload: %s" % workload.upper())

N = 8
M = 4*512

mem = np.zeros((N, M),dtype=np.int32)
NO_SHIFTS_OCCURED = True
ENDURER = False

arrays = {} # format of key -> (chip, start, len) for each software array
pattern = {} # format of key -> [....] (len from arrays above)
array_loc = {} # format of key -> (chip, new_start) 
heap = np.zeros((N,),dtype=np.int32)

def add_pattern(repeat_factor):
    for arr in arrays:
        c,start = array_loc[arr]
        _,_,L = arrays[arr]
        if start+L < M:
            mem[c][start:start+L] += pattern[arr]*repeat_factor
        else:
            leftover = start + L - M
            mem[c][start:M] += repeat_factor*pattern[arr][:L-leftover]
            mem[c][:leftover] += repeat_factor*pattern[arr][L-leftover:]

def hardware_endurer_remap():
    if ENDURER!='hard':
        return

    global mem
    NO_SHIFTS_OCCURED = False
    offset = np.random.randint(M, size=(N,), dtype=np.int32)
    for arr in arrays:
        c, loc = array_loc[arr]
        new_loc = (loc + offset[c]) % M
        array_loc[arr] = (c, new_loc)

    mem += 1 # all positions written to in remap

#HYPERPARAMETERS
TAU = .6
EPS0 = TAU
DIST_OFFSET = 8e6 
ENABLED = True

#variables
dist_step = 1
eps = EPS0

threshold = lambda writes_ : writes_.sum(1).mean()*(1+eps) + DIST_OFFSET

def time_to_distribute():
    return np.any(mem.sum(1)>threshold(mem)) and ENABLED

def distributed_endurer_remap():
    if ENDURER!='distributed':
        return
   
    global mem 
    if time_to_distribute():
       light,heavy = get_distribution_points() 
       for arr in arrays:
            c, loc = array_loc[arr]
            if (c==heavy): c = light
            elif (c==light): c = heavy
            array_loc[arr] = (c, loc)
       mem[light,:] += 1
       mem[heavy,:] += 1

    NO_SHIFTS_OCCURED = False
    offset = np.random.randint(M, size=(N,), dtype=np.int32)
    for arr in arrays:
        c, loc = array_loc[arr]
        new_loc = (loc + offset[c]) % M
        array_loc[arr] = (c, new_loc)
    
    mem += 1 # all positions get written to

def get_distribution_points():
    global eps, dist_step, chip_map, writes, estimates, switch_tracker
    if ENABLED:
        eps += TAU**dist_step
        dist_step += 1
 
        heavy = mem.sum(1).argmax()
        light = mem.sum(1).argmin()
        print("Switched chip " + str(heavy) + " with " +str(light))
        return light, heavy

# Source: https://pynative.com/python-generate-random-string/
def random_string(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def allocate_memory(chip, size, writes):
    assert NO_SHIFTS_OCCURED
    ok = False
    while not ok:
        key = random_string()
        ok = (key not in arrays)
    start = heap[chip]
    assert (start + size < M) , "Array won't fit in this chip!"
    arrays[key] = (chip, start, size)
    pattern[key] = writes
    array_loc[key] = (chip, start)  
    heap[chip] += size

def reset_array_loc():
    for arr in arrays:
        chip, start, _ = arrays[arr]
        array_loc[arr] = (chip, start)

if workload=='svhn':
    # SVHN
    writes = 64 
    allocate_memory(0, writes, np.ones((writes,), dtype=np.int32))
    inf_period = 35 # seconds
elif workload=='d2nn':
    # D2NN
    allocate_memory(0, 12*6*6/2, np.ones((12*6*6/2,), dtype=np.int32))
    inf_period = 4.8 # seconds
else: #lstm
    # LSTM
    # HDIM 80, IDIM 10, ODIM 11
    # cell -> HDIM/N
    HDIM = 80
    for i in range(N):
        allocate_memory(i, HDIM/N, np.ones((HDIM/N,), dtype=np.int32))
    inf_period = 0.66 # seconds

remap_period = 4 # hours
inf_per_period = int(3600*remap_period/inf_period) 

print("Inferences per period:",inf_per_period)

#print(pattern)
#print([a.mean() for b,a in pattern.items()])

print("MIN\tMAX\tMEAN\tSUM")

print("Simulating lifetime without Any Endurer")
# assume RP in hours evenly divisible
add_pattern(inf_per_period*(24//remap_period)*10*365)

print(mem.min(), mem.max(), mem.mean(), mem.sum())

ENDURER = 'hard'
reset_array_loc()
mem *= 0
print("Simulating lifetime with Endurer")
for day in range(10*365):
    for p in range(24//remap_period): # assume in hours evenly divisible
        add_pattern(inf_per_period)
        if p!=0 and day!=0:
            hardware_endurer_remap()
    #if (day%365==0):
    #    print("Year",day//365)


print(mem.min(), mem.max(), mem.mean(), mem.sum())

ENDURER = 'distributed'
reset_array_loc()
mem *= 0
print("Simulating lifetime with Distributed Endurer")
for day in range(10*365):
    for p in range(24//remap_period): # assume in hours evenly divisible
        add_pattern(inf_per_period)
        if p!=0 and day!=0:
            distributed_endurer_remap()
    #if (day%365==0):
    #    print("Year",day//365)


print(mem.min(), mem.max(), mem.mean(), mem.sum())


