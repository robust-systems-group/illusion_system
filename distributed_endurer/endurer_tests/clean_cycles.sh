#!/bin/sh

python process_bit_failure.py --prefix ../data/raw/d2nn_de --output ../data/processed/d2nn_de_clean &
python process_bit_failure.py --prefix ../data/raw/d2nn_no_resilience --output ../data/processed/d2nn_no_resilience_clean &

python process_bit_failure.py --prefix ../data/raw/svhn_de --output ../data/processed/svhn_de_clean &
python process_bit_failure.py --prefix ../data/raw/svhn_no_resilience --output ../data/processed/svhn_no_resilience_clean

for C in 0 1 2 3 4 5 6 7
do
   python process_bit_failure.py --prefix ../data/raw/lstm_de_c$C --output ../data/processed/lstm_de_clean_c$C &
   python process_bit_failure.py --prefix ../data/raw/lstm_no_resilience_c$C --output ../data/processed/lstm_no_resilience_clean_c$C &
done

