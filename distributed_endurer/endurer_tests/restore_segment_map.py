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

# Sample arg: python restore_segment_map.py filename

NUM_ARGS=1

def has_error(error):
    return (error[:-1].split()[-1]!="0")

if (len(sys.argv) < NUM_ARGS):
    print('Need to specify filename')
    exit(1)
elif not os.path.exists(args[0]):
    print("state_filename ("+args[0]+") not valid path")
    exit(2)

f = open(filename)
for i in range(8):
    line = f.readline()[:-1]
    [a,b,c] = line.split(',')
    # technically maybe should make them int but will become str anyway so whatev?
    error = board.assign_segment(i, a, b, c)

    if has_error(error):
        print("Error restoring segment " + str(i)+", ABORT")
        exit(6)

error = board.verify_segments()
 
if has_error(error):
    print("Error verifying segment, ABORT")
    exit(7)



