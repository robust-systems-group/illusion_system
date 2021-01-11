#!/bin/bash

# inputs
if [ "$#" -ne 0 ]; then
    echo "Illegal number of parameters"
    exit 1
fi

MODE=0
WORKLOAD=svhn

#check workload store right
./initialize_endurer_test.py $MODE $WORKLOAD
./workload_degradation_test.sh $MODE $WORKLOAD 0

