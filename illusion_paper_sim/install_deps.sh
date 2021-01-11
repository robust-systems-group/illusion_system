#!/bin/sh
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


# create some essential folders for zsim
mkdir ops
mkdir results

# fetch and decompress deps/
tar -xvzf deps.tar.gz

# decompress multichip/nn_dataflow
cd multichip
tar -xvzf nn_dataflow.tar.gz

# decompress multichip/op_trace
tar -xvzf op_trace.tar.gz

# decompress multichip/protoio
tar -xvzf protoio.tar.gz

# return to root dir
cd ..


