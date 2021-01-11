# Illusion System: Code, Measured Data, Simulation Framework

This repository contains the code, measured data, and simulation framework used in the paper:

[1] Radway, R. et al. "Illuison of large on-chip memory by networked computing chips for neural network inference", Nature Electronics, Jan. 2021, https://www.nature.com/articles/s41928-020-00515-3.

An Illusion System consists of a network of multiple computing chips, each with a certain minimal amount of local on-chip memory and mechanisms for quick wakeup and shutdown (that is, the system contains no separate memory-only chips). For DNN inference tasks, Illusion performs like an ideal chip, with near-ideal energy, execution time and EDP. In hardware, we demonstrate an Illusion System consisting of eight computing chips, and the energy, execution time and EDP of this eight-chip Illusion System are measured to be within 1.035×, 1.025× and 1.06×, respectively, of the values of the ideal chip (which correspondingly contains eight times more memory than the individual chips used in the demonstration). 

There are three main components to this repo, each within its own directory. The first is related to the hardware measurements performed on our physical eight-chip Illusion System. The second relates the system simluations of a scaled out Illusion System using DNN Accelerators. The final is related to the Distributed ENDURER technique presented in the Illusion paper, which provides multi-chip write endurance resilience.

Illusion Testing Code and Data
==============================
Directory: `illusion_testing`

Our Illusion Mapping and Scheduling approach are given in detail in the paper's supplementary information. Here we provide the DNN model training, DNN source code, compiled executables, measurement setup, and measurement results for the four workloads presented in the paper as if they were operating on a single ideal chip, a four-chip Illusion System, and on our true eight-chip Illusion System. In addition, we provide the analysis scripts used to generate the main results provided in the paper.

Directory map:
- `training` : QPyTorch [2] training scripts used to train our quantized DNNs under test
- `c_models` : C implemented golden models of the trained DNNs (including hard-coded weights partitioned). These models can be used to generate the ground truth messages between chips in Illusion Systems
- `c_models_msp` : Models coded to run on our physical hardware (the individual chips are previously decribed in [3]). Compilation uses the MSP430 open source compiler [4] with modified linker for our hardware.
- `programs` : Compiled data and program memories for the workloads, for both ideal chip and four-/eight-chip Illusion Systems. 
- `ios` : Reference inputs and outputs for a single inference to be used for functionality checks
- `measurement` : Measurement framework code and scripts. Operation requires full HW setup.
- `data` : Measured performance data (e.g., power traces) for the DNNs under test
- `plotting` : Analysis and plots of final performance data, generates performance plots shown in [1].

[2] Zhang, T., Lin, Z., Yang, G. & De Sa, C. QPyTorch: a low-precision arithmetic simulation framework. Preprint at https://arxiv.org/abs/1910.04540 (2019).
[3] Wu, T. F. et al. 14.3-A 43-pJ/cycle non-volatile microcontroller with 4.7-μs shutdown/wake-up integrating 2.3-bit/cell resistive RAM and resilience techniques. In Proc. IEEE International Solid-State Circuits Conference (ISSCC) 226–228 (IEEE, 2019).
[4] MSP430-GCC-OPENSOURCE GCC – Open Source Compiler for MSP Microcontrollers (Texas Instruments, accessed 5 August 2020); https://www. ti.com/tool/MSP430-GCC-OPENSOURCE

Illusion Simulations and Data
=============================
Directory: `illusion_paper_sim`

Steps for reproducing the results:

0. Extract all compressed dependency libraries by running `./install_deps.sh`

1. Schedule generation:

    (under current directory) 
    `cd multichip`
    `./generateMitSchedules_mem.sh [batchsize] [wordsize]` (e.g. batchsize=1, wordsize=16 -> ./generateMitSchedules_mem.sh 1 16). You can modify the output directory and the workload by modifying the commented section in this script.
    `./generateMitSchedules_mem_mp.sh [batchsize] [wordsize]` (using the same parameters and workloads above)
    `./generate_MitTraces_mem.sh [batchsize]` [wordsize] (using the same parameters above, set LSTM=True in gen_mem_trace.py if running LSTM workload, False otherwise)
    `python3 create_workloads.py`

    Notes:
    - To change workloads, edit the corresponding mem and net names in these scripts
    - Only 1 net should be simulated at a time
    - Due to limitations of `nn_dataflow`, we need to split the `lstm_langmod` workload by each lstm cell and thus we need to run one simulation for each split workload (`lstm_langmod` and `lstm_langmod_1`)

2. Copy the workloads to the simulation directory by running:

    `cd ..`
    `./copy_multichip_workload.sh`
    `vim workloads.conf` (and enable the corresponding net to be simulated)

3. Kickoff simulation

    `python3 main.py >& log.txt`

4. Analyze results

    `cd sim_results`
    `cp results . -r` (this copies the zsim results to the current folder)
    `cp ../multichip/message_passing . -r` (this copies the chip partition and messages info to the current folder)
    `vim analyze_results.py` (and set the corresponding net name, directory paths and SPLIT (memsize for each chip))
    `python3 analyze_results.py`

    Notes:
    - The script automatically aggregates the LSTM results in the coefficient section if you put `lstm_langmod` and `lstm_langmod_1` together

Distributed ENDURER Testing Code and Data
=========================================
Directory: `distributed_endurer`

We provide a new multi-chip write endurance resilience scheme called Distributed ENDURER in [1]. We tested this scheme on three of our DNN workloads on the actual RRAM hardware. This directory contains the code for its execution.

Directory map:
- `c_models` : golden models
- `c_models_test` : Evaluates accuracy using weight & activation masks from different time points
- `endurer_tests` : FPGA scripts 
- `fpga_endurer` : FPGA version of endurer
- `data` : Measured data reported in [1]
- `plotting` : Plotting scripts from appendix for [1]



