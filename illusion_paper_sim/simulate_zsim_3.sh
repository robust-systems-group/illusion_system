#!/usr/bin/bash
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


# Note: This script (simulate_zsim.sh) is designed to be called after cd'ing
# to the folder that contains all the network's layer folders (e.g., called from
# within "Baseline", which contains "act_1", "conv_1", "embed1", etc.).

# Squash the annoying HDF5 version warning
export HDF5_DISABLE_VERSION_CHECK=1

# This variable is used throughout the rest of this file to provide relative
# paths to various zsim dependencies (and zsim itself).
SIM_BASE_DIR=$PWD/../../..


# The ZSIM_PATH (note the underscore) environment variable is used by zsim
# internally to locate Pin (specifically, the pinbin executable) and the zsim
# pintool (libzsim.so).
export ZSIM_PATH=$SIM_BASE_DIR/deps/ENVPATH

# Ensure zsim sees the dynamic libraries it needs; specifically, libprotobuf and
# libhdf5. (Though statically linked, zsim calls dlopen() and needs these.)
export LD_LIBRARY_PATH=$SIM_BASE_DIR/deps/ORNL-zsim-logic/lib/protobuf-master/lib:$SIM_BASE_DIR/deps/ORNL-zsim-logic/lib/hdf5-1.8.18/lib



echo "Simulating: ${PWD##*/}"

for D in $PWD/*;
do
    # (cd $D && echo "[Y] Running ${PWD##*/}" && $SIM_BASE_DIR/deps/ORNL-zsim-logic/build/opt/zsim $SIM_BASE_DIR/config/zsim/zsim_${1}_${2}.cfg &>zout.txt)
    (cd $D && echo "[Y] Running ${PWD##*/}" && $SIM_BASE_DIR/deps/ORNL-zsim-logic/build/opt/zsim ${1})
#exit
done

echo "Simulation Finished"
