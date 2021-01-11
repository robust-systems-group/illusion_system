# Raw Measurement Data


These are the raw datafiles containing the power traces measured. First we programmed the DNNs under test (e.g. the bitstreams in `Programs`) onto our Illusion System using HW-specific programming scripts. We then verified the Illusion System operation (overall, and chip-wise vs. the golden IOs in `ios`), and then ran our measuremnt scripts to measure the power consumption of each chip). 

The scripts and utils in `../plotting/` show how to process and read out the raw measurements. 

