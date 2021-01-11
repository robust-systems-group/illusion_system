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

# Sample arg: python initialize_endurer_test.py 0 svhn 

NUM_ARGS=2
workloads = ['svhn']

def has_error(error):
    return (error[:-1].split()[-1]!="0")

def check_error(error, err_str, err_num):
    if has_error(error):
        print(error_str)
        exit(err_num)

if (len(sys.argv) < NUM_ARGS):
    print('Need to specify Endurer mode & workload')
    exit(1)
elif args[1] not in workloads:
    print("Invalid workload, must be one of: "+str(workloads))
    exit(2)

error = board.endurer_reset()
check_error(error, "Error resetting board", 4)

#define CTRL_INSTR 0x00000000
#define CTRL_DATA  0x00000002

board.endurer_chmod(args[0])
error = board.assign_segment(0, 0, 0, 0)
check_error(error, "Error in assignment segment 0", 7)
error = board.assign_segment(1, 0, 0, 1)
check_error(error, "Error in assignment segment 1", 7)
error = board.assign_segment(2, 0, 0, 2)
check_error(error, "Error in assignment segment 2", 7)
error = board.assign_segment(3, 0, 2, 0)
check_error(error, "Error in assignment segment 3", 7)
error = board.assign_segment(4, 1, 0, 0)
check_error(error, "Error in assignment segment 4", 7)
error = board.assign_segment(5, 1, 0, 1)
check_error(error, "Error in assignment segment 5", 7)
error = board.assign_segment(6, 1, 0, 2)
check_error(error, "Error in assignment segment 6", 7)
error = board.assign_segment(7, 1, 2, 0)
check_error(error, "Error in assignment segment 7", 7)

error = board.verify_segments()
check_error(error, "Error in verifying segment", 6)

for i in range(8):
    error = board.endurer_set_memory("init/"+args[1]+str(i)+".init", 0)
    check_error(error, "Error setting memory virt chip"+str(i), 5)

 
