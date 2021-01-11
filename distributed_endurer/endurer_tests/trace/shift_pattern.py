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
import os



filename_in = sys.argv[1]
filename_out = sys.argv[2]
shift = int(sys.argv[3],16)
direction = int(sys.argv[4])
chip_shift = int(sys.argv[5])
def padded_hex(val):
    return "{0:0{1}X}".format(val,4)

f_in = open(filename_in)
f_out = open(filename_out,'w+')

for line in f_in.readlines():
    a,b,c = str.split(line,',')
    if direction:
        new_b = int(b,16) + shift
    else:
        new_b = int(b,16) - shift
    f_out.write(str(chip_shift) +','+padded_hex(new_b) +','+padded_hex(int(c,16))+'\n')

f_in.close()
f_out.close()

