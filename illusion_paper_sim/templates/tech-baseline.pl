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


#Architecture Params
$NCORES=!nNodes!;
$ncompute=!nCompute!;
$NUM_CACHE_LEVELS=2;
@NCACHE=(2*$NCORES,$NCORES);
$NCONTROLLERS=!nNodes!;
$leakage_power = !compLkgPerNode!;
$Dynamic_E_per_op = !dynEPerOp!;
$Frequency=!frequency!; #Core frequency in GHz
$reg_bit_width=16;
$reg_e_per_bit=!regEPerBit!;
#Power/Performance Parameters
@CACHE_E=(!localBufferEPerAccess!,!localBufferEPerAccess!,!globalBufferEPerAccess!);#Cache energy access nJ
#@CACHE_LKG=(0.068,0.068,gbufLkg); #mW
@CACHE_LKG=(0.068,0.068,1.6); #mW
# NOTE: mem energy calculated separately (just placeholders below?)
@MEM_E_per_bit=(0,0); #Energy per bit in memories in pJ/bit
$LINE_SIZE=256; #line size in Bits
$Do_power_map=0;
1;
