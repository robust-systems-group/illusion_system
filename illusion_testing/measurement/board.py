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


import time
import serial
import time 
import csv
import sys

ser = serial.Serial('/dev/ttyACM0', 115200)
time.sleep(1)
ser.flushInput()
FPGA_CLOCK_PERIOD_NS = 10

irsver_calib = list()
isver_calib = list()
ird_calib = list()
ibias_calib = list()

def padded_hex(val):
    return "{0:0{1}X}".format(val,4)

# Read in LUT for current sources.
with open('currents.csv', 'r') as csvfile:
  csvreader = csv.reader(csvfile)
  next(csvreader)
  for row in csvreader:
    irsver_calib.append(float(row[1])*1e6)
    isver_calib.append(float(row[1])*1e6)
    ird_calib.append(float(row[1])*1e6)
    ibias_calib.append(float(row[1])*1e6)

# Make sure connected to correct board
print('*IDN\n')
ser.write('*IDN\n')
r = ser.readline()
r = r.rstrip('\n\r')
if (r == 'LETI Test Board'):
    print('Connected to board')
else :
    exit(1)

# Set currents
def set_currents(dev, irsver, isver, ird, ibias):
    irsver_val = min(range(len(irsver_calib)), key=lambda i: abs(irsver_calib[i]-irsver))
    isver_val = min(range(len(isver_calib)), key=lambda i: abs(isver_calib[i]-isver))
    ird_val = min(range(len(ird_calib)), key=lambda i: abs(ird_calib[i]-ird))
    ibias_val = min(range(len(ibias_calib)), key=lambda i: abs(ibias_calib[i]-ibias))

    print('*CURR ' + str(dev) + ' ' + str(ibias_val) + ' ' + str(ird_val) + ' ' + str(irsver_val) + ' ' + str(isver_val))
    ser.write('*CURR ' + str(dev) + ' ' + str(ibias_val) + ' ' + str(ird_val) + ' ' + str(irsver_val) + ' ' + str(isver_val) + '\n')
    print(ser.readline())
    time.sleep(1)

# Set currents
def set_raw_currents(dev, irsver, isver, ird, ibias):
    #print('*CURR ' + str(dev) + ' ' + str(ibias) + ' ' + str(ird) + ' ' + str(irsver) + ' ' + str(isver))
    ser.write('*CURR ' + str(dev) + ' ' + str(ibias) + ' ' + str(ird) + ' ' + str(irsver) + ' ' + str(isver) + '\n')
    #print(ser.readline())
    ser.readline()
    time.sleep(1)

# Manual Mode
def manual_mode():
    print('*MODE 0')
    ser.write('*MODE 0\n')
    print(ser.readline())

# Step
def step():
    print('*STEP')
    ser.write('*STEP\n')
    print(ser.readline())

# Normal Mode
def normal_mode():
    print('*MODE 1')
    ser.write('*MODE 1\n')
    print(ser.readline())

# Select Chip
def select_chip(chip):
    print ('Selecting Chip: ' + str(chip))
    ser.write('*SEL '+str(chip)+'\n')
    #print(ser.readline())
     
# Set clock on
def clock_on():
    print('*CLK 1')
    ser.write('*CLK 1\n')
    print(ser.readline())

# Set clock on
def clock_off():
    print('*CLK 0')
    ser.write('*CLK 0\n')
    print(ser.readline())

def reset(i):
    print('*RESET '+str(i)+'\n')
    ser.write('*RESET '+str(i)+'\n')
    print(ser.readline())
    

# Set clock period (in ns)
def set_clock_period(period):
    divider = int(period/FPGA_CLOCK_PERIOD_NS/2 - 1)
    print('*CLKDIV ' + str(divider))
    ser.write('*CLKDIV ' + str(divider) + '\n')
    print(ser.readline())

# Set BLDIS time (in ns)
def set_bldis(time):
    val = int((time/FPGA_CLOCK_PERIOD_NS) - 1)
    print('*BLDIS ' + str(val))
    ser.write('*BLDIS ' + str(val) + '\n')
    print(ser.readline())

# Read word
def read_word(segment, address):
    #print('*READ ' + str(segment) + ' ' + format(address, 'x'))
    ser.write('*READ ' + str(segment) + ' ' + format(address, 'x') + '\n')
    val = ser.readline()
    val = val.rstrip('\n\r')
    return int(val,16)

def fifo_reset(reset):
    print('*FIFORESET ' + str(reset)+'\n')
    ser.write('*FIFORESET ' + str(reset)+'\n')
    print(ser.readline())
        

# Write word
def write_word(segment, address, value):
    #print('*WRITE ' + str(segment) + ' ' + format(address, 'x') + ' ' + format(value, 'x'))
    ser.write('*WRITE ' + str(segment) + ' ' + format(address, 'x') + ' ' + format(value, 'x') + '\n')
    val = ser.readline()
    val = val.rstrip('\n\r')
    return int(val)

def fifo_full():
    ser.write('*FIFOFULL\n')
    val = ser.readline().rstrip('\n\r')
    return int(val)

def wakeup(wake):
    ser.write('*WAKEUP '+str(wake)+'\n')
    print(ser.readline())

def fifo_empty():
    ser.write('*FIFOEMPTY\n')
    val = ser.readline().rstrip('\n\r')
    return int(val)

def idn():
    ser.write('*IDN\n') 
    val = ser.readline().rstrip('\n\r')
    print(val)
    return val

def read_fifo():
    #Check if FIFO has data
    #if fifo_empty() == 1: 
    #    print('No data in FIFO') 
    #else:
    #print('*FIFOREAD')
    ser.write('*FIFOREAD \n')
    s = ser.readline()
    #print(int(s,16))
    return s.rstrip('\n\r')

def write_fifo(data):
    #Check if FIFO full
    if fifo_full() == 0:
        #print('*FIFOWRITE ' + str(data))
        ser.write('*FIFOWRITE ' + str(data)+'\n')
        a = ser.readline()
        #print(a.rstrip())
    else:
        print('FIFO Full')

#Scan, returns tuple
def scan(a, b, c):
    ser.write("*SCAN " + str(a) + ' ' + str(b) + ' ' + str(c) + "\n")
    #print("*SCAN " + str(a) + ' ' + str(b) + ' ' + str(c) )
    r = ser.readline()
    #print("Returned: " + r)
    fields = r.split()
    a_r = int(fields[0],16)
    b_r = int(fields[1],16)
    c_r = int(fields[2],16)
    return a_r, b_r, c_r

# Turn the scan enable (and scan_mode) signal off and on
def scan_enable(state):
    if (state == True):
        ser.write("*SCANEN 1\n" )
    else:
        ser.write("*SCANEN 0\n" )
    ser.readline()

# Load data into DRAM
def load(address,words):
    ser.write("*LOAD " + format(address, 'x') + "\n")
    for word in words:
        ser.write(format(word,'x') + "\r\n")
    ser.write("\r\n")
    r = ser.readline()
    return r;

def program(dram_address,segment, address, num_words):
    ser.write("*PROG " + format(dram_address, 'x') + ' ' + str(segment) + ' ' + format(address, 'x') + ' ' + str(num_words) + "\n")
    r = ser.readline()
    #print(r)
    r = ser.readline()
    #print(r)

def endurer(dram_address, trace_length, remap_interval, run_duration, trace_duration):
    ser.write("*ENDURE " + format(dram_address, 'x') + ' ' + str(trace_length) + ' ' + 
        str(remap_interval) + ' ' + str(run_duration) + ' ' + str(trace_duration) + "\n")
    r = ser.readline()
    print(r)
    r = ser.readline()
    print(r)

def endurer_reset():
    ser.write("*ERESET\n")
    r = ser.readline()
    print(r)
    return r

def endurer_chmod(new_mode):
    ser.write("*ECHMOD " + str(new_mode) + "\n")
    r = ser.readline()
    print(r)
    return r

def endurer_set_state_manual(new_mode, swaps, virt_to_phys, offsets, counts, counts_retry, tau, eps, c, max_retries):
    counts_string = " ".join([str(x) for x in counts])
    counts_retry_str = " ".join([str(x) for x in counts_retry])
    offsets_string = " ".join([str(x) for x in offsets])
    virt_string = " ".join([str(x) for x in virt_to_phys])
    #counts_string = list(counts).replace(","," ")[1:-1]
    #counts_retry_string = list(counts_rety).replace(","," ")[1:-1]
    #offsets_string = list(offsets).replace(","," ")[1:-1]
    #virt_string = list(virt_to_phys).replace(","," ")[1:-1]

    ser.write("*SESTATE " + str(new_mode) + " " + str(swaps) + " " + \
                virt_string + " " + offsets_string + " " + counts_string + " " + counts_retry_str + " " + \
                format(tau, "12.6f") + " " + format(eps, "12.6f") + " " + format(c, "12.6f") + " " + str(max_retries) + "\n")
    r = ser.readline()
    print(r)
    return r

def endurer_set_state(filename):
    f = open(filename)
    total_string = ''
    r = f.readline()
    while r!='':
        total_string += r[:-1]
        r = f.read_line()
    ser.write("*SESTATE " + r + "\n")
    f.close()
    r = ser.readline()
    print(r)
    return r

def endurer_get_state():
    ser.write("*GESTATE\n")
    r = ser.readline()
    mode = int(r)
    print("Mode: " + r)
    r = ser.readline()
    swaps = int(r)
    print("Swaps: " + r)
    r = ser.readline()
    virt_to_phys = [int(x) for x in str.split(r)]
    print("Physical Mappping: " + r)
    r = ser.readline()
    offsets = [int(x) for x in str.split(r)]
    print("Offsets: " + r)
    r = ser.readline()
    counts = [int(x) for x in str.split(r)]
    print("Counts no retry: " + r)
    r = ser.readline()
    counts_retry = [int(x) for x in str.split(r)]
    print("Counts retry: " + r)
    r = ser.readline()
    params  = [float(x) for x in str.split(r)]
    tau = params[0]
    eps = params[1]
    c = params[2]
    max_retries = int(params[3])
    print("DE Params: " + r)
    return mode, swaps, virt_to_phys, offsets, counts, counts_retry, tau, eps, c, max_retries

chunk_size = 64

def endurer_send_trace(filename, trace_addr):
    lines = open(filename).readlines()
    total_lines = len(lines)
    trace_addr_high = trace_addr/0x10000
    trace_addr_low  = (trace_addr % 0x10000)
    ser.flushInput()
    ser.write("*RITRACE " + padded_hex(trace_addr_high) +" "+ padded_hex(trace_addr_low)+ " " + str(len(lines)) + '\n')
    for i in range(len(lines)):
        s  = lines[i]
        if (i%chunk_size)==0:
            r = ser.readline()
            print(i)
        ser.write(str.rstrip(s)+'\n')
    r = ser.readline()
    return int(r)

def endurer_read_out_trace(trace_addr, length, file_out):
    trace_addr_high = trace_addr/0x10000
    trace_addr_low  = (trace_addr % 0x10000)
    ser.write("*ROTRACE " + padded_hex(trace_addr_high) +" "+ padded_hex(trace_addr_low)+ " " + str(length) + "\n")
    f = open(file_out,"w+")
    for i in range(length):
        if (i%chunk_size)==0: 
            ser.write("CONTINUE\n")
            print(i)
        r = ser.readline()
        f.write(r)
        #print(r)
    r = ser.readline()
    return int(r)

def endurer_run_trace(virt_chip, trace_addr, length):
    trace_addr_high = trace_addr/0x10000
    trace_addr_low  = (trace_addr % 0x10000)
    ser.write("*RNTRACE " + padded_hex(trace_addr_high) + " " + padded_hex(trace_addr_low) \
            + " " + str(length) + " " + str(virt_chip) + "\n")
    r = ser.readline()
    print(r)
    return r

def endurer_run_all(trace_addr,trace_length, times_btwn_remap, total_times):
    trace_addr_high = trace_addr/0x10000
    trace_addr_low  = (trace_addr % 0x10000)
    ser.write("*RUNEND " + padded_hex(trace_addr_high) + " " + padded_hex(trace_addr_low) + " " + str(trace_length) + \
            " " + str(times_btwn_remap) + " " + str(total_times)  + "\n")
    r = ser.readline()
    print(r)
    return r

def endurer_run_single_chip(chip, trace_addr, trace_length, times_btwn_remap, total_times):
    trace_addr_high = trace_addr/0x10000
    trace_addr_low  = (trace_addr % 0x10000)
    ser.write("*CRUNEND " + str(chip) + " " + padded_hex(trace_addr_high) + " " + padded_hex(trace_addr_low) + " " + str(trace_length) + \
            " " + str(times_btwn_remap) + " " + str(total_times)  + "\n")
    r = ser.readline()
    print(r)
    return r

def endurer_read_out_memory(filename):
    ser.write("*RDMEM\n")
    f = open(filename,'w+')
    for i in range(8*2048/16):
        if (i % 128) == 0: 
            print("Virtual Chip %d"%(i/128)) 
            f.write("Virtual Chip %d\n"%(i/128))
        if (i % chunk_size) == 0:
            ser.write("CONTINUE\n")
            print(i)
        r = ser.readline()
        f.write(str.strip(r)+"\n")
    f.close()
    print(ser.readline()) 
    return
    

def endurer_read_out_memory_raw(filename):
    ser.write("*RRDMEM\n")
    f = open(filename,'w+')
    for i in range(8*2048/16):
        if (i % 128) == 0: 
            print("Physical Chip %d"%(i/128)) 
            f.write("Physical Chip %d\n"%(i/128))
        if  ((i % chunk_size) == 0):
            ser.write("CONTINUE\n")
            print(i)
        r = ser.readline()
        f.write(str.strip(r)+"\n")
    f.close()
    print(ser.readline()) 
    return

def endurer_set_memory(filename, virt_chip):
    ser.write("*STMEM " + str(virt_chip) + "\n")
    f = open(filename)
    ser.flushInput()
    f.readline()
    for i in range(2048/16):
        sl = str.split(f.readline())
        s = ' '.join(sl[1:]) 
        ser.write(str.rstrip(s)+'\n')
    f.close()
    r = ser.readline()
    print(r)
    return r

def endurer_set_memory_raw(filename, phys_chip):
    ser.write("*RSTMEM " + str(phys_chip) + "\n")
    f = open(filename)
    ser.flushInput()
    f.readline()
    for i in range(2048/16):
        sl = str.split(f.readline())
        s = ' '.join(sl[1:]) 
        ser.write(str.rstrip(s)+'\n')
    f.close()
    r = ser.readline()
    print(r)
    return r

def endurer_remap(chip):
    ser.write("*EREMAP " + str(chip))
    r = ser.readline()
    print(r)
    return r
def endurer_swap(chip0, chip1):
    ser.write("*C2CSWAP " + str(chip0) + " " + str(chip1) + "\n")
    r = ser.readline()
    print(r)
    return r

def select_segment(rram_chip, segment_type, chunk_id):
    ser.write("*CSEG " + str(rram_chip) + " " + str(segment_type) + " " + str(chunk_id) + "\n")
    r = ser.readline()    
    print(r)
    return r

def assign_segment(phys_chip_id, rram_chip, segment_type, chunk_id):
    ser.write("*EASEG " + str(phys_chip_id) + " " + str(rram_chip) + " " + str(segment_type) +\
                        " " + str(chunk_id) + "\n")
    r = ser.readline()    
    print(r)
    return r

def verify_segments():
    ser.write("*EVSEG\n")
    r = ser.readline()    
    print(r)
    return r

def read_out_segments():
    ser.write("*ERSEG\n")
    for i in range(8):
        print(ser.readline())

# Closes connection to the board. Not really needed
def close():
    ser.close()

