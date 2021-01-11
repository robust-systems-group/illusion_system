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

filename_out = sys.argv[1]
value = int(sys.argv[2],16)

def padded_hex(val):
    return "{0:0{1}X}".format(val,4)

f_out = open(filename_out,'w+')

for chip in range(8):
    for word in range(2048):
        f_out.write(str(chip) +','+padded_hex(word) +','+padded_hex(value)+'\n')

f_out.close()

