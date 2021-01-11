#!/bin/sh
#sample
#./workload_degradation_test.sh $ENDURER_MODE $WORKLOAD $RUN_LENGTH
ENDURER_MODE=$1
WORKLOAD=$2
RUN_LENGTH=$3

./read_out_memory.py > memory_map_$ENDURER_MODE\_$WORKLOAD\_$RUN_LENGTH\.txt
cd ../../illusion_system/scripts
python extract_model_$WORKLOAD\.py --input ../../illusion_testing/endurer_tests/memory_map_$ENDURER_MODE\_$WORKLOAD\_$RUN_LENGTH\.txt
cd ../c_models_test/c_$WORKLOAD/
./accuracy_test.sh > degradation_results_$ENDURER_MODE\_$RUN_LENGTH\.txt
cd ../../../illusion_testing/endurer_tests

