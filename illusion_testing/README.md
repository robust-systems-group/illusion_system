# Testing Code &amp and Data for Illusion System Performance Measurements

This code is related to the measurements performed on the Illusion System.  

Our Illusion Mapping and Scheduling approach are given in detail in the paper's supplementary information. Here we provide the DNN model training, DNN source code, compiled executables, measurement setup, and measurement results for the four workloads presented in the paper as if they were operating on a single ideal chip, a four-chip Illusion System, and or full eight-chip Illusion System. In addition, we provide the analysis scripts used to generate the main results provided in the paper.

Directory map:
- `training` : QPyTorch [2] training scripts used to train our quantized DNNs under test
- `c_models` : C implemented golden models of the trained DNNs (including hard-coded weights partitioned). These models can be used to generate the ground truth messages between chips in Illusion Systems
- `c_models_msp` : Models coded to run on our physical hardware (the individual chips are previously decribed in [3]). Compilation uses the MSP430 open source compiler [4] with modified linker for our hardware.
- `programs` : Compiled data and program memories for the workloads, for both ideal/four-/eight-chip Illusion Systems. 
- `ios` : Refernce inputs and outputs for a single inference to be used for functionality checks
- `measurement` : Measurement framework code and scripts. Operation requires full HW setup.
- `data` : Measured performance data (e.g., power traces) for the DNNs under test
- `plotting` : Analysis and plots of final performance data, generates plots shown in [1].

[1] Radway, R. et al. "Illuison of large on-chip memory by networked computing chips for neural network inference", Nature Electronics, Jan. 2021.
[2] Zhang, T., Lin, Z., Yang, G. & De Sa, C. QPyTorch: a low-precision arithmetic simulation framework. Preprint at https://arxiv.org/abs/1910.04540 (2019).
[3] Wu, T. F. et al. 14.3-A 43-pJ/cycle non-volatile microcontroller with 4.7-μs shutdown/wake-up integrating 2.3-bit/cell resistive RAM and resilience techniques. In Proc. IEEE International Solid-State Circuits Conference (ISSCC) 226–228 (IEEE, 2019).
[4] MSP430-GCC-OPENSOURCE GCC – Open Source Compiler for MSP Microcontrollers (Texas Instruments, accessed 5 August 2020); https://www. ti.com/tool/MSP430-GCC-OPENSOURCE

Below we provide a more in-depth overview of the components of this directory and how they inter-relate.

For the SVHN CNN, KWS LSTM and D2NN, we mimicked our ideal chip and four-chip Illusion systems on one and four of our hardware chips, respectively. The physical measurement techniques are the same. The weight mapping to the chips assumed the full capacity was available (for example, 32 kBytes for the ideal chip on one chip). This results in more weights than can be compiled into the physical memory (4 kBytes capacity). These excess weights were overlapped (addresses modulo 4 kBytes) in the same physical address space using a software-defined data structure. Owing to the simple instruction set in our hardware, this mimicked the same instruction execution and memory access patterns as an ideal chip or four-chip Illusion system. The Illusion system EDP values we achieve (relative to the ideal chip measurement) are thus conservative estimates. Our small CNN (inference on MNIST) requires none of this treatment, as it fits on each on-chip memory for all the systems (for the four-chip and eight-chip Illusions, we map as if we had only 1 kByte or 0.5 kBytes of RRAM per chip). For the three DNNs discussed above and the small MNIST CNN, the results are consistent across scales, confirming that our measurement techniques for the four-chip Illusion system and ideal chip are valid on the large DNNs, as one chip is already an ideal chip for the small CNN. In addition, by using exactly the same hardware for workloads requiring one chip (MNIST CNN) up to eight chips (SVHN CNN, KWS LSTM, D2NN), we show that our Illusion systems are configurable and flexible. We achieve near-ideal EDP, regardless of the number of chips used by the DNN in the Illusion system.

Training
==============================
First we trained quantized models for the various tasks described in [1]. A quantized training software [2] so that we could run on our hardware (e.g., limited to 8-bit fixed precision operations). This folder contains scripts to generate the quantized data and model c files for use in our tests.

Golden C Models
==============================
The neural networks were then implemented in C, with the various layers hand implemented to ensure correct operation on our hardware. These models were compiled natively and the accuracy was compared to the final training results to ensure correctness. Finally, printouts were inserted at the appropriate points to generate golden references for the chip-to-chip communication that would occur on the ideal chip (e.g., just the initial input and DNN output) as well as the 4- and 8-chip Illusion Systems. These were used to test functional correctness of the programs when implemented on our hardware (see `ios`) for the 8-chip Illusion System. We could then force the reduced memory space that we have in the mimicked ideal chip and 4-chip Illusion system scenarios. 

MSP430 C Models
==============================
Our hardware implements an open-source MSP430 on each chip. We needed to modify our golden C models to properly utilize the integrated multiplier, as well as correctly flag the on-chip scheduler for proper wakeup and self-shutdown (e.g. implement the fine-grained temporal power gating described in [1]). This code also implements the appropriate message passing code for the our Illusion Systems. These were compiled and simulated on the RTL for our hardware, to ensure that the code was functionally correct when compared to the Golden C Models. Compiler flags can be used to determine if the ideal chip, 4-chip or 8-chip Illusion System code is generated.

Programs
==============================
These are the compiled programs (raw bitstream for the program and data memories on our chip) to run on our hardware that achieved the Illusion Results described.

IOs
==============================
These are the golden C model-generated input and output streams for each chip in an Illusion System (4-chip and 8-chip) as well as for our mimicked ideal chip. As our hardware's memory is still emerging, bit failures and programming errors can occur [3] describes the techniques we used to overcome these challenges. These IOs provide a golden reference to ensure our chips were programmed and operating correctly (even for the ideal chip and 4-chip case, where the weights are overlapped). 

Measurement
==============================
These are the python scripts used to perform our measurment results. This contains a function interface model of the FPGA master of the Illusion System, to interface appropriately with the actual hardware. The measurement scripts used to interface with the multi-channel ADC for our power measurements are included as well.

Data
==============================
These are the raw datafiles containing the power traces measured. First we programmed the DNNs under test (e.g. the bitstreams in `Programs`) onto our Illusion System using HW-specific programming scripts. We then verified the Illusion System operation (overall, and chip-wise vs. the golden IOs in `ios`), and then ran our measuremnt scripts to measure the power consumption of each chip). As we describe in the methods [1] 

Plotting
==============================
These scripts parse the raw datafiles in `data` and generate the final performance numbers and plots found in [1].
