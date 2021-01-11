#!/bin/bash

# inputs
#$1 -> endurer mode (0,1,2)
#$2 -> workload

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters (mode, workload)"
    exit 1
fi
if [[ $1 -ne 1 && $1 -ne 2 && $1 -ne 0 ]]; then
    echo "Choose an endurer mode! (0,1,2)"
    exit 2
fi
if [[ "$2" != "svhn" ]]; then
    echo "Choose valid workload! (svhn)"
    exit 3
fi



#number of remap periods in time!
DAY=6 
WEEK=42 #6*7 
MONTH=180 #6*30
YEAR=2190 #6*365
TENYEAR=21900 #6*3650

ADDR=0048

MODE=$1
WORKLOAD=$2

#check workload store right
./initialize_endurer_test.py $MODE $WORKLOAD
./workload_degradation_test.sh $MODE $WORKLOAD 0

# run first period
RUN_LENGTH=1
i=1
TIMES_TO_RUN=$((RUN_LENGTH*TIMES_BETWEEN))
 
LENGTH=`./store_trace.py "trace/$WORKLOAD" $ADDR`
./run_distributed_endurer_test.py $ADDR $LENGTH $TIMES_BETWEEN $TIMES_TO_RUN
OLD_STATE=state_$WORKLOAD\_$RUN_LENGTH\_$MODE\.txt
./backup_endurer_state.py > $OLD_STATE
./workload_degradation_test.sh $MODE $WORKLOAD $i
./memory_degradation_test.sh $MODE $WORKLOAD $i

for i in $DAY $WEEK $MONTH $YEAR $TENYEAR
do 
    RUN_LENGTH=$((i-RUN_LENGTH))
    TIMES_TO_RUN=$((RUN_LENGTH*TIMES_BETWEEN))
    echo "reload from $OLD_STATE for run length $RUN_LENGTH"
    ./restore_endurer_test.py $OLD_STATE
    ./run_distributed_endurer_test.py $ADDR $LENGTH $TIMES_BETWEEN $TIMES_TO_RUN
    OLD_STATE=state_$WORKLOAD\_$RUN_LENGTH\_$MODE\.txt
    echo "backup state to file $OLD_STATE"
    ./backup_endurer_state.py > $OLD_STATE
    ./workload_degradation_test.sh $MODE $WORKLOAD $i
    ./memory_degradation_test.sh $MODE $WORKLOAD $i
done

# Turn off supplies when finished
./off.py

