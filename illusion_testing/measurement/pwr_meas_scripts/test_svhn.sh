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


for i in  0  
do
    echo final_curr/svhn_target_w${i}_c${1}_0k5_${j}.npz
    ./pwr_meas_scripts/svhn_pwr_test.py $1 $2 $i target 1 1 final_curr/svhn_target_w${i}_c${1}_0k5.npz
done

for i in 1 2 3 4 5 6 7 
do
    echo final_curr/svhn_il_w${i}_c${1}_0k5.npz
    ./pwr_meas_scripts/svhn_pwr_test.py $1 $2 $i illusion 1 1 final_curr/svhn_il_w${i}_c${1}_0k5.npz
done

for i in 1 2 3
do
    echo final_curr/svhn_ilsm_w${i}_c${1}_0k5.npz
    ./pwr_meas_scripts/svhn_pwr_test.py $1 $2 $i illusion_sm 1 1 final_curr/svhn_ilsm_w${i}_c${1}_0k5.npz
done
