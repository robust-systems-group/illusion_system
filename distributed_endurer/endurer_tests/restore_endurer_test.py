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

# Sample arg: python restore_endurer_test.py state_filename svhn 

NUM_ARGS=2
workloads = ['svhn']

if (len(sys.argv) < NUM_ARGS):
    print('Need to specify Endurer mode & workload')
    exit(1)
elif not os.path.exists(args[0]):
    print("state_filename ("+args[0]+") not valid path")
    exit(2)
elif args[1] not in workloads:
    print("Invalid workload, must be one of: "+str(workloads))
    exit(3)

error = board.endurer_set_state(args[0])
if (error[:-1].split()[-1]!="0"):
    print("Error setting memory virt chip"+str(i))
    exit(6)

for i in range(8):
    error = board.endurer_set_memory("init/"+args[1]+str(i)+".init", 0)
    if (error[:-1].split()[-1]!="0"):
        print("Error setting memory virt chip"+str(i))
        exit(5)
   


