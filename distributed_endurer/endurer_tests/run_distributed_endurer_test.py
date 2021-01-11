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

# Sample arg: python run_distributed_endurer_test.py offset length time_btwn_rmp times

NUM_ARGS=4

if (len(sys.argv) < NUM_ARGS):
    print('Need to specify offset, length, time btwn remap, times')
    exit(1)

error = board.endurer_run_all(int(args[0]), int(args[1]),
                            int(args[2]), int(args[3]))
if (error[:-1].split()[-1]!="0"):
    print("Error running endurer_all trace: ABORT")
    exit(7)

   


