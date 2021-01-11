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


import sys,time
sys.path.append('../')
import serial
import math
import random
from testing import board
from testing import voltages
import numpy as np
from datetime import datetime
from labjack import ljm
from matplotlib import pyplot as plt


MAX_REQUESTS = 2  # The number of eStreamRead calls to perform
run_time = 1
# Open first found T7
handle = ljm.openS("T7", "ANY", "ANY")  # T7 device, Any connection, Any identifier

info = ljm.getHandleInfo(handle)
print("Opened a LabJack with Device type: %i, Connection type: %i,\n"
      "Serial number: %i, IP address: %s, Port: %i,\nMax bytes per MB: %i" %
      (info[0], info[1], info[2], ljm.numberToIP(info[3]), info[4], info[5]))

deviceType = info[0]

# Stream Configuration
scanRate = 1000
scansPerRead = int(scanRate )
ljm.eWriteName(handle, "STREAM_TRIGGER_INDEX", 0)
ljm.eWriteName(handle, "STREAM_CLOCK_SOURCE", 0)
#ljm.eWriteName(handle,"STREAM_BUFFER_SIZE_BYTES")

# All negative channels are single-ended, AIN0 and AIN1 ranges are
# +/-10 V, stream settling is 0 (default) and stream resolution index
# is 0 (default).
aNames = ["AIN_ALL_NEGATIVE_CH", "AIN_ALL_RANGE",
          "STREAM_SETTLING_US", "STREAM_RESOLUTION_INDEX"]
aValues = [1, 0.01, -1, 7]

# Write the analog inputs' negative channels (when applicable), ranges,
# stream settling time and stream resolution configuration.
numFrames = len(aNames)
ljm.eWriteNames(handle, numFrames, aNames, aValues)


clock_per = 1000 #ns,determined by board ic length more than FPGA; Can't go much faster than 2.5MHz
chip_sel = int(sys.argv[1])
currs = int(sys.argv[2])
work_sel = int(sys.argv[3])
mode = sys.argv[4]

dar_en_pmem = int(sys.argv[5])
dar_en_dmem = int(sys.argv[6])

file_out = sys.argv[7]


v12 = 62
vddsa = 20
v46 = 91
channel_base = [48, 80, 96]

#In/out data may be different lenths
in_data =[]
with open('io/lstm/mode_'+str(mode)+'_io/chip_'+str(work_sel)+'_in.txt') as fp:
    for line in fp:
        in_data.append(line.rstrip())
fp.close()
print(in_data)
out_expected = []
with open('io/lstm/mode_' +str(mode)+'_io/chip_'+str(work_sel)+'_out.txt') as fp:
    for line in fp:
        out_expected.append(line.rstrip())
fp.close()
print(out_expected)

in_data_length = len(in_data) - 1
print(in_data_length)
out_data_length = len(out_expected) - 1
print(out_data_length)
"""
Start up the system
"""
voltages.set_safe_voltages()
board.select_chip(chip_sel)
board.set_currents(chip_sel, currs, currs, currs, 6.4)

board.clock_on()
board.set_bldis(10)
board.set_clock_period(clock_per)
time.sleep(0.1)
time.sleep(0.1)

time.sleep(0.1)
board.read_word(0,2*dar_en_dmem + dar_en_pmem) #use dynamic addr remap if needed (programming errors)
time.sleep(0.1)
board.reset(1)
time.sleep(0.1)
board.normal_mode()

"""
System is ready to go
"""

"""
Function to reset both sets of FIFOs
"""
def reset_fifos():
    board.fifo_reset(1) #Set reset high
    time.sleep(1) #Need to have at least 3 slow clock cycles for correct reset
    board.fifo_reset(0) #Set reset low
    time.sleep(1) #Need to have at least 3 slow clock cycles for correct reset
    if board.fifo_empty():
        return
    else:
        print("ERROR in FIFOs")

print('Sending inputs to Chip' + str(chip_sel))
reset_fifos()

board.write_fifo('0080') #This is a quick runtime
board.write_fifo('0000') 

print('FIFO Loaded')

print('Chip Prep')
time.sleep(0.1)
board.reset(0)
time.sleep(1)

print('CHIP in Shutdown?')
data_out = [] 
while not board.fifo_empty():
    data_out.append(board.read_fifo())
print(data_out)
print(board.fifo_empty())
reset_fifos()
print(board.fifo_empty())
time.sleep(1)

"""
This resets the chip without it starting to compute
"""
curr_out = np.ones((3, scansPerRead*MAX_REQUESTS))
all_data_out = []
for channel in range(3):
    aScanListNames = ["AIN"+str(channel_base[channel]+chip_sel)] #Scan list names to stream
    numAddresses = len(aScanListNames)
    aScanList = ljm.namesToAddresses(numAddresses, aScanListNames)[0]
    
    print('Chip Running')
    reset_fifos()
    for j in range(len(in_data)): 
        board.write_fifo(in_data[j])
    print(run_time)
    #raw_input('Press any key to continue...')
    scanRate = ljm.eStreamStart(handle, scansPerRead, numAddresses, aScanList, scanRate)
    print("\nStream started with a scan rate of %0.0f Hz." % scanRate)
    start = datetime.now()
    time.sleep(1)
    board.wakeup(1)
    time.sleep(0.00001)
    board.wakeup(0)
    time.sleep(run_time) #TODO This should be based on the per-en done language but its just simpler to over-provision runtime
    
    print('Chip Done')
    
    print("\nPerforming %i stream reads." % MAX_REQUESTS)
    totScans = 0
    totSkip = 0  # Total skipped samples
    
    i = 1
    while i <= MAX_REQUESTS:
        ret = ljm.eStreamRead(handle)
        aData = ret[0]
        scans = len(aData) / numAddresses
        totScans += scans
        curr_out[channel,scans*(i-1):scans*(i)] = np.array(aData)
        # Count the skipped samples which are indicated by -9999 values. Missed
        # samples occur after a device's stream buffer overflows and are
        # reported after auto-recover mode ends.
        curSkip = aData.count(-9999.0)
        totSkip += curSkip
        #print(aData)
        print("\neStreamRead %i" % i)
        ainStr = ""
        for j in range(0, numAddresses):
            ainStr += "%s = %0.5f, " % (aScanListNames[j], aData[j])
        print("  1st scan out of %i: %s" % (scans, ainStr))
        print("  Scans Skipped = %0.0f, Scan Backlogs: Device = %i, LJM = "
              "%i" % (curSkip/numAddresses, ret[1], ret[2]))
        i += 1
    
    end = datetime.now()
    print("\nTotal scans = %i" % (totScans))
    tt = (end - start).seconds + float((end - start).microseconds) / 1000000
    print("Time taken = %f seconds" % (tt))
    print("LJM Scan Rate = %f scans/second" % (scanRate))
    print("Timed Scan Rate = %f scans/second" % (totScans / tt))
    print("Timed Sample Rate = %f samples/second" % (totScans * numAddresses / tt))
    print("Skipped scans = %0.0f" % (totSkip / numAddresses))
    
    print("\nStop Stream")
    ljm.eStreamStop(handle)
    
    data_out = [] 
    while not board.fifo_empty():
            data_out.append(board.read_fifo())
    all_data_out.append(data_out) 
    print(data_out)
    print("DATA OUT LENGTH: " + str(len(data_out)))
    print("Errors:")
    try:
        print([int(data_out[i] != out_expected[i]) for i in range(len(out_expected[0:-1]))])
    except:
        print("WARNING INCORRECT DATA RECEIVED")

board.clock_off()
voltages.close()
ljm.close(handle)

np.savez(file_out, data=curr_out, msg=np.array(all_data_out), chip=chip_sel)
# Close handle
for i in range(3):
    plt.plot(curr_out[i])
plt.show()
