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

mode = int(sys.argv[1])
mode_o, swaps, virt_to_phys, offsets, counts, counts_retry, tau, eps, c, max_retries = board.endurer_get_state()
board.endurer_set_state_manual(mode, swaps, virt_to_phys, offsets, counts, counts_retry, tau, eps, c, max_retries)
mode_n, swaps, virt_to_phys, offsets, counts, counts_retry, tau, eps, c, max_retries = board.endurer_get_state()


