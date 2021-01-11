out_dir=message_passing
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


# uncomment the workload below
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
	mkdir -p ${out_dir}/${i}_${2}_${1}/${j}
	/usr/bin/python eyeriss_search_multi_mc.py --batch $1 --word $2 --nodes 4 4 --array 16 16 --regf 256 --gbuf 2097152 --op-cost 1 --hier-cost 200 6 2 1 --hop-cost 0 --unit-static-cost 1 $i ${out_dir}/${i}_${2}_${1}/${j}/ --memsize ${j} --disable-bypass 'i' 'o' 'f' --message-passing 
done

for i in "${lstm[@]}"
do
	if [ "$i" == "langmod" ] || [ "$i" == "lm1b" ]; then
		var2=$(($1*32))
		var1=$(($1*4))
	elif [ "$i" == "captioning" ]; then
		var2=$1
		var1=$(($1*32))
	fi

	mkdir -p ${out_dir}/lstm_${i}_${2}_${1}/${j}
	mkdir -p ${out_dir}/lstm_${i}_1_${2}_${1}/${j}
#	mkdir -p ${out_dir}/lstm_${i}_2_${2}_${1}/${j}
	
	/usr/bin/python eyeriss_search_multi_mc.py --batch $var2 --word $2 --nodes 4 4 --array 16 16 --regf 256 --gbuf 2097152 --op-cost 1 --hier-cost 200 6 2 1 --hop-cost 0 --unit-static-cost 1 lstm_${i} ${out_dir}/lstm_${i}_${2}_${1}/${j}/ --memsize ${j} --disable-bypass 'i' 'o' 'f' --lstm True --message-passing
	/usr/bin/python eyeriss_search_multi_mc.py --batch $var1 --word $2 --nodes 4 4 --array 16 16 --regf 256 --gbuf 2097152 --op-cost 1 --hier-cost 200 6 2 1 --hop-cost 0 --unit-static-cost 1 lstm_${i}_1 ${out_dir}/lstm_${i}_1_${2}_${1}/${j}/ --memsize ${j} --disable-bypass 'i' 'o' 'f' --lstm True --message-passing

#	if [ "$i" == "langmod" ] || [ "$i" == "lm1b" ]; then
#		cp examples/lstm_${i}_2.py examples/lstm_${i}_temp.py
#		sed -i -e 's/batch_size1/'"$var1"'/g' examples/lstm_${i}_temp.py
#		sed -i -e 's/batch_size2/'"$var2"'/g' examples/lstm_${i}_temp.py
#	elif [ "$i" == "captioning" ]; then
#		cp examples/lstm_${i}_2.py examples/lstm_${i}_temp.py
#		sed -i -e 's/batch_size/'"$var1"'/g' examples/lstm_${i}_temp.py
#	fi
#
#       /usr/bin/python eyeriss_search_multi_mc.py --batch 1 --word $2 --nodes 2 2 --array 16 16 --regf 1024 --gbuf 2097152 --op-cost 1 --hier-cost 200 6 2 1 --hop-cost 0 --unit-static-cost 1 lstm_${i}_temp ${out_dir}/lstm_${i}_2_${2}_${1}/${j}/ --memsize ${j} --disable-bypass 'i' 'o' 'f' --message-passing
done
done
