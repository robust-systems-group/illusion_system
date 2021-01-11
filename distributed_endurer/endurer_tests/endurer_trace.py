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


# Forms a word at a specified address and cycles them N times

import voltages
import board
import sys
import csv
import time
import numpy as np

IRSVER_REF = 5 #uA
ISVER_REF = 8.8 #uA
IRD_REF = 6.4 #uA
IBIAS_REF = 6 #uA
FORM_CLOCK_PERIOD = 120000 # ns
NORMAL_CLOCK_PERIOD = 1000 # ns
BLDIS_TIME = 10 # ns
N = 10 # cycles
MAX_RETRIES = 4 #cycles
data_mem = 1
stride = 2048

if (len(sys.argv) < 2):
    print('Need to specify trace')
    exit(1)

if (len(sys.argv) < 3):
    print('Need to specify how long to run for')
    exit(1)

if (len(sys.argv) < 4):
    print('Need to specify remap interval')
    exit(1)

if (len(sys.argv) < 5):
    print('Need to specify application runtime')
    exit(1)


trace_file = sys.argv[1]
run_duration = float(sys.argv[2])
remap_interval = float(sys.argv[3])
app_duration = float(sys.argv[4])

all_ones = [0xffff for i in range(0,stride)]
all_zeros = [0x0000 for i in range(0,stride)]

def load_dram(mem):
    board.load(0,mem)

def padded_hex(val):
    return "{0:#0{1}x}".format(val,6)

def set() :
    dram_address = 0
    board.program(dram_address,data_mem, 0, stride)

def read(address) :
    val = board.read_word(data_mem, address)
    val = board.read_word(data_mem, address)
    return val

def reset() :
    dram_address = stride
    board.program(dram_address,data_mem, 0, stride)


############################################
# Set default settings for voltages, currents, and clocks
voltages.set_safe_voltages()
voltages.set_normal_voltages()
board.set_currents(IRD_REF, IRD_REF, IRD_REF, IBIAS_REF)
#board.set_currents(IRSVER_REF, ISVER_REF, IRD_REF, IBIAS_REF)
board.set_clock_period(NORMAL_CLOCK_PERIOD)
board.clock_on()
board.set_bldis(BLDIS_TIME)

board.set_clock_period(NORMAL_CLOCK_PERIOD)
time.sleep(1)
num_set_retries = list()
num_reset_retries = list()
set_states = list()
reset_states = list()


f = open(trace_file, 'r')
csv_reader = csv.reader(f, delimiter=',')
trace = list()
for row in csv_reader:
    trace.append(int(row[0],16))
    trace.append(int(row[1],16))
    trace.append(int(row[2],16))
print("Clearing Memory")
load_dram(all_zeros)
board.program(0,1,0,stride)
print("Loading DRAM")
load_dram(trace)
print("Running ENDURER")
board.endurer(0, len(trace)/3, remap_interval, run_duration, app_duration)





