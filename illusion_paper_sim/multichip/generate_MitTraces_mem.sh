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


cnn=(resnet50)
lstm=()
# cnn=(alex_net)
# cnn=(vgg_net)
# lstm=(langmod)
mem=(0.25 0.5 1 2 4 8 16 32 64 128 256 512 1024 2048)

for j in "${mem[@]}"
do
for i in "${cnn[@]}"
do
for D in schedule/${i}_${2}_${1}/${j}/*;
do
	echo ${D}
	path=./trace/${i}_${2}_${1}/${j}
	python2 gen_trace_mem.py  schedule/${i}_${2}_${1}/${j}/${D##*/} ${path}/${D##*/} 
done
done

for i in "${lstm[@]}"
do
for D in  schedule/lstm_${i}_${2}_${1}/${j}/*;
do
	path=./trace/lstm_${i}_${2}_${1}/${j}
	python2 gen_trace_mem.py  schedule/lstm_${i}_${2}_${1}/${j}/${D##*/} ${path}/${D##*/} 
done
for D in  schedule/lstm_${i}_1_${2}_${1}/${j}/*;
do
	path=./trace/lstm_${i}_1_${2}_${1}/${j}
	python2 gen_trace_mem.py  schedule/lstm_${i}_1_${2}_${1}/${j}/${D##*/} ${path}/${D##*/} 
done
#for D in  mitSchedule/lstm_${i}_2_${2}_${1}/${j}/*;
#do
#	path=/scratch0/radway/multichip/lstm_${i}_2_${2}_${1}/${j}
#	python $ZSIMPATH/misc/opTrace/test/gen_trace_mem.py  mitSchedule/lstm_${i}_2_${2}_${1}/${j}/${D##*/} ${path}/${D##*/} 
#done
done
done
