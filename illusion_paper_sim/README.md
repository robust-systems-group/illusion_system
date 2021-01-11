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

