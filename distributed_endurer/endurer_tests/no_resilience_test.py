#!/usr/bin/env python
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


import sys, os
sys.path.append("../")
import time
from illusion_testing import board, voltages
from collections import OrderedDict
 
# Sample arg: python store_trace.py trace_file addr

voltages.set_normal_voltages()
voltages.close()

IRSVER_REF =12 #uA
ISVER_REF =12 #uA
IRD_REF = 12 #uA
IBIAS_REF = 6.4 #uA
NORMAL_CLOCK_PERIOD = 1000 # ns
FAST_CLOCK_PERIOD = 100 # ns
BLDIS_TIME = 10 # ns
MAX_RETRIES = 1 #cycles

board.set_clock_period(NORMAL_CLOCK_PERIOD)
board.set_bldis(BLDIS_TIME)
board.clock_on()
for i in range(2):
    board.set_currents(i, IRD_REF, IRD_REF, IRD_REF, IBIAS_REF)

NUM_ARGS=3
print(sys.argv)
if (len(sys.argv) < NUM_ARGS):
    print('Need to specify CHIP and output dir, load trace, lstm_no?')
    exit(1)
elif not os.path.isdir(sys.argv[2]):
    print("Output dir ("+sys.argv[2]+") not valid path")
    exit(2)

chip = int(sys.argv[1])
out_dir  = sys.argv[2]
load_in = int(sys.argv[3])

lstm_no = int(sys.argv[4])
board.endurer_reset()

#trace_svhn = "/home/pi/illusion_testing/endurer_tests/trace/lstm_write_pattern_"+str(lstm_no)+"_shift.txt"
trace_svhn = "/home/pi/illusion_testing/endurer_tests/trace/svhn_write_pattern.txt"
trace_zeros = "/home/pi/illusion_testing/endurer_tests/trace/zeros.txt"
trace_ones = "/home/pi/illusion_testing/endurer_tests/trace/ones.txt"
print("Trace: "+trace_svhn)
addr_svhn = 0
addr_zeros = 1024*1024
addr_ones = 1024*1024*2

if load_in:
    print("LOADING TRACES FOR TESTING")
    print("WORKLOAD TRACE")
    length_svhn = board.endurer_send_trace(trace_svhn, addr_svhn)
    print("TRACE LOADED, LENGTH: %d"%length_svhn)
    print("ZEROS TRACE")
    length_zeros = board.endurer_send_trace(trace_zeros, addr_zeros)
    print("TRACE LOADED, LENGTH: %d"%length_zeros)
    print("ONES TRACE")
    length_ones = board.endurer_send_trace(trace_ones, addr_ones)
    print("TRACE LOADED, LENGTH: %d"%length_ones)
else:
    #length_svhn = 216400
    length_svhn = 153530
    length_zeros = 16384
    length_ones = 16384

def cycle(num_times, zeros_file, ones_file):
    for i in range(num_times - 1):
        board.endurer_run_trace(chip, addr_zeros, length_zeros)
        board.endurer_run_trace(chip, addr_ones, length_ones)
    board.endurer_run_trace(chip, addr_zeros, length_zeros)
    board.endurer_read_out_memory_raw(out_dir + "/"+zeros_file)
    board.endurer_run_trace(chip, addr_ones, length_ones)
    board.endurer_read_out_memory_raw(out_dir + "/"+ones_file)

print("SETTING ENDURER SYSTEM STATE")
board.endurer_reset()
print("TRACES LOADED, NOW INITIAL BERs")
#Set to one retry. 
print("SETTING TO SINGLE RETYR ATTEMPT")
mode, swaps, virt_to_phys, offsets, counts, counts_retry, tau, eps, c, max_retries = board.endurer_get_state()
max_retries = MAX_RETRIES
board.endurer_set_state_manual(mode, swaps, virt_to_phys, offsets, counts, counts_retry, tau, eps, c, max_retries)
mode, swaps, virt_to_phys, offsets, counts, counts_retry, tau, eps, c, max_retries = board.endurer_get_state()
cycle(1, "0trace_1c_0.txt", "0trace_1c_1.txt")
cycle(10, "0trace_10c_0.txt", "0trace_10c_1.txt")
 
 
print("NOW RUNNING TEST")
#TRACE RUNTIMEs
runtimes = OrderedDict()
runtimes['DAY'] = 6
runtimes['WEEK'] = 42
runtimes['MONTH'] = 180
runtimes['YEAR'] =  2190
runtimes['TENYEAR'] = 21900
runtimes['TWENTYYEAR'] = 21900*2
runtimes['FIFTYYEAR'] = 21900*5
runtimes['HUNDYEAR'] = 21900*10
#runtimes = OrderedDict()
#runtimes['DAY'] = 6*3
#runtimes['WEEK'] = 42*3
#runtimes['MONTH'] = 180*3
#runtimes['YEAR'] =  2190*3
#runtimes['TENYEAR'] = 21900*3
#runtimes['TWENTYYEAR'] = 21900*2*3
#runtimes['FIFTYYEAR'] = 21900*5*3
#runtimes['HUNDYEAR'] = 21900*10*3

states = OrderedDict()
for i in range(5):#len(runtimes.keys())):
    if i==0:
        previous_period = 0
    else:
        previous_period = runtimes[runtimes.keys()[i-1]]
    period = runtimes.keys()[i]
    times = runtimes[period] - previous_period

    print("NOW RUNNING PERIOD OF " + period)
    states[period] = board.endurer_get_state()
    print("RUNNING TRACE " + str(times))
    start_time = time.time()
    board.set_clock_period(FAST_CLOCK_PERIOD) #Speed up writing test
    board.set_bldis(BLDIS_TIME)
    board.clock_on()
    for i in range(times):
        if ((i% 100)==0):
            print(i) 
        board.endurer_run_trace(chip, addr_svhn, length_svhn)
    board.set_clock_period(NORMAL_CLOCK_PERIOD) #Reading requires normal clock
    board.set_bldis(BLDIS_TIME)
    board.clock_on()
    cycle(1, str(times) +"trace_1c_0.txt", str(times)+"trace_1c_1.txt")
    cycle(10, str(times)+"trace_10c_0.txt", str(times)+"trace_10c_1.txt")
    end_time = time.time()
    print("PERIOD " +str(period) +" took " + str(end_time-start_time))
 
states['END'] = board.endurer_get_state()
with open(out_dir+"/"+"end_states_long.txt",'w+') as f:
    f.write(repr(states))
    f.close()




