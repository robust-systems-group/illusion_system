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


echo "LSTM Eval: No Resiliency using memory_lstm2 for all!"

for T in ZERO DAY WEEK MONTH YEAR TENYEAR 
do
    echo "Evaluating accuracy: $T"
    cd scripts
    python extract_model_lstm2.py --prefix ../../processed/lstm_no_resilience_clean_c --time $T
    cd ../../../c_models_test/c_lstm2/
    cp model_DE_ZERO.c model_chunked_LSTM.c
    ./accuracy_test.sh
    cp activation_mask.c activation_mask_NR_$T\.c
    cp model_chunked_LSTM.c model_NR_$T\.c
    cd ../../data/extracted_accuracy
done

