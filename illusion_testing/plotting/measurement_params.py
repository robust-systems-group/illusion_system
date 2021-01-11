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


import numpy as np
import matplotlib.pyplot as plt
import sys

#Measured resistances 
resistances = np.array([
    [60.9,61.7,61.4,61.7,62.3,61.9,62.6,61.6],
    [19.8,19.8,19.8,19.8,19.8,19.8,20.0,19.8],
    [91.2,91.2,91.2,90.5,90.5,91.6,90.3,91.0]])


voltages = np.array([1.20, 2.4, 3.6]) #Nominal
