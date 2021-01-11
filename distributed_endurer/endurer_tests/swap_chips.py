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

# Sample arg: python swap_chips.py ch0 ch1

NUM_ARGS=2

if (len(sys.argv) < NUM_ARGS):
    print('Need to specify the 2 chips to swap')
    exit(1)
elif int(args[0]) >7 or int(args[0])<0:
    print(args[0]+ " is not a valid chip id")
    exit(2)
elif int(args[1]) >7 or int(args[1])<0:
    print(args[1]+ " is not a valid chip id")
    exit(3)
elif args[0]==args[1]:
    print("Cannot swap onto same chip")
    exit(4)

board.endurer_swap(int(args[0]), int(args[1]))

   


