#Illusion System Measurement


These are the python scripts and code framework used to perform our measurment results. This contains a function interface model of the FPGA master of the Illusion System, to interface appropriately with the actual hardware. The measurement scripts used to interface with the multi-channel ADC for our power measurements are included as well.

Directory map:
- `board.py` : Functional model of the interface to the FPGA master of the Illusion System 
- `*.py` :  Other scripts to control external GPIB equipment for test, voltages, setup the labjack etc.
- `pwr_meas_scripts` : Scripts used to perform the power measurements

