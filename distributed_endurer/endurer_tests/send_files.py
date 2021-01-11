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


import sys
sys.path.append("../")
import time
from illusion_testing import board

pmem_file = sys.argv[1]
dmem_file = sys.argv[2]

IRSVER_REF = 5 #uA
ISVER_REF = 8.8 #uA
IRD_REF = 6.4 #uA
IBIAS_REF = 6 #uA
CLOCK_PERIOD = 1000 # ns
BLDIS_TIME = 10 # ns

voltages.set_safe_voltages()
voltages.set_normal_voltages()
board.set_currents(IRD_REF, IRD_REF, IRD_REF, IBIAS_REF)
board.set_clock_period(CLOCK_PERIOD)
board.clock_on()
board.set_bldis(BLDIS_TIME)

# Load the files
ser.write("*LOAD\n")
f = open(pmem_file, 'r')
for line in f:
    a = line.split()
    if (len(a) > 2) :
        ser.write(" ".join(a[1:]) + "\n")
        print(" ".join(a[1:]))
f.close()

f = open(dmem_file, 'r')
for line in f:
    a = line.split()
    if (len(a) > 2) :
        ser.write(" ".join(a[1:]) + "\n")
        print(" ".join(a[1:]))
f.close()
ser.write("\n")

# Program
print("Programming...\n")
ser.write("*PROG\n")
line = ser.readline()
print(line)
while not line.startswith("<MAP>") :
    print(line)
    line = ser.readline()

line = ser.readline()
while not line.startswith("</MAP>") :
    print(line)
    line = ser.readline()


ser.close()

