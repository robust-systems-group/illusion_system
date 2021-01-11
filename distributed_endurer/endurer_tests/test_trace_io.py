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
from illusion_testing import board

# Sample arg: python store_trace.py trace_file addr

NUM_ARGS=2
print(sys.argv)
if (len(sys.argv) < NUM_ARGS):
    print('Need to specify trace file & trace start addr in dram')
    exit(1)
elif not os.path.exists(sys.argv[1]):
    print("trace_file ("+sys.argv[1]+") not valid path")
    exit(2)
 
length = board.endurer_send_trace(sys.argv[1], int(sys.argv[2]))
print(length)
status = board.endurer_read_out_trace(int(sys.argv[2]), length)
print(status)

