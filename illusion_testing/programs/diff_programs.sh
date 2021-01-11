#!/bin/bash
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


for b in pmem.mem #dmem.mem
do
for a in ./${1}/MODETARGET/${2}/${3}/${4}/CHIP0/${b}
do
for i in MODEILLUSION
do
    for j in CHIP0 CHIP1 CHIP2 CHIP3 CHIP4 CHIP5 CHIP6 CHIP7
    do
        echo $i
        echo $j
    diff ${a}  ./${1}/${i}/${2}/${3}/${4}/${j}/${b}
    done
done

for i in MODEILLUSIONSM
do
    for j in CHIP0 CHIP1 CHIP2 CHIP3
    do
        echo $i
        echo $j
        diff ${a}  ./${1}/${i}/${2}/${3}/${4}/${j}/${b}
    done 
done
done
done


