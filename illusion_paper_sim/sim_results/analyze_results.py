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


import os
import json
import itertools
import argparse
from collections import OrderedDict

import numpy as np

# change the factors here (only messaging factor should be changed for the content in the paper)
MESSAGING_WORSEN_FACTOR = 1
LEAKAGE_WORSEN_FACTOR = 1

# this should not be modified
BANDWIDTH = 32 / MESSAGING_WORSEN_FACTOR # GB/s
LATENCY = 100 # ns
E_PER_BYTE = 256 * MESSAGING_WORSEN_FACTOR # pJ/Byte

# change the directory names here
RESULTS_DIR = "./multichip_results_32"
OUTPUT_DIR = "./multichip_analysis"
MP_DIR = "./multichip_message_passing_32"

# uncomment the workload you want
# NETS = ["lstm_langmod", "lstm_langmod_1"]
NETS = ["resnet50",]
# NETS = ["alex_net",]
# NETS = ["vgg_net",]

WORDS = [16,]
BATCHES = [1,]
SPLIT = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

NUM_CORE = 4*4
NUM_CU = 4*4
CU_LKG = 0.000774
LKG_PWR = CU_LKG * NUM_CU * NUM_CORE / 64 * LEAKAGE_WORSEN_FACTOR

def extract_table(net, word, batch, split):
    data = {}
    data["is_lstm"] = 1 if "lstm" in net else 0

    # read corresponding message passing file
    if split >= 1:
        mp_file = open(MP_DIR + "/" + net + "_" + str(word) + "_" + str(batch) + "/" + str(split) + "/" + net + "_" + str(split) + ".0_mp.csv")
    else:
        mp_file = open(MP_DIR + "/" + net + "_" + str(word) + "_" + str(batch) + "/" + str(split) + "/" + net + "_" + str(split) + "_mp.csv")

    lines = mp_file.read().splitlines()
    mp_file.close()

    # get total number of nodes using the message passing split file
    num_nodes = int(((lines[-1].split(','))[0].split('_'))[1])

    data["num_nodes"] = num_nodes
    data["node_data"] = {}

    results_prefix = net + "_16_1_" + net + "_" + str(word) + "_" + str(batch) + "_rram_mem_"

    for node_idx in range(1, num_nodes+1):
        # read node results if it exists
        node_name = "node_" + str(node_idx)
        results_file = os.path.join(RESULTS_DIR, results_prefix + str(split) + "_" + node_name + ".csv")
        if not os.path.exists(results_file):
            continue

        data["node_data"][node_name] = {}

        with open(results_file, 'r') as f:
            lines = f.read().splitlines()
            res_header = lines[1].split(',')
            # read in all the layer info
            for line in lines[2:]:
                layer_cols = line.split(',')
                layer_name = layer_cols[0]

                # stop if no more layers
                if "Total" in layer_cols[0]:
                    break

                # load layer info col by col
                data["node_data"][node_name][layer_name] = {}
                for col_num in range(1, len(res_header)):
                    data["node_data"][node_name][layer_name][res_header[col_num][1:]] = float(layer_cols[col_num])

            # load total info
            data["node_data"][node_name]["Total"] = {}
            for col_num in range(1, len(res_header)):
                total_cols = lines[-4].split(',')
                data["node_data"][node_name]["Total"][res_header[col_num][1:]] = float(total_cols[col_num])

        if data["is_lstm"]:
            node_name = "node_" + str(node_idx) + "_dummy"
            results_file = os.path.join(RESULTS_DIR, results_prefix + str(split) + "_" + node_name + ".csv")
            if not os.path.exists(results_file):
                continue

            if "dummy_start_node" not in data:
                data["dummy_start_node"] = node_idx

            data["node_data"][node_name] = {}

            with open(results_file, 'r') as f:
                lines = f.read().splitlines()
                res_header = lines[1].split(',')
                # read in all the layer info
                for line in lines[2:]:
                    layer_cols = line.split(',')
                    layer_name = layer_cols[0]

                    # stop if no more layers
                    if "Total" in layer_cols[0]:
                        break

                    data["node_data"][node_name][layer_name] = {}

                    # load layer info col by col
                    for col_num in range(1, len(res_header)):
                        data["node_data"][node_name][layer_name][res_header[col_num][1:]] = float(layer_cols[col_num])

                # load total info
                data["node_data"][node_name]["Total"] = {}
                for col_num in range(1, len(res_header)):
                    total_cols = lines[-4].split(',')
                    data["node_data"][node_name]["Total"][res_header[col_num][1:]] = float(total_cols[col_num])

    return data


def extract_split(net, word, batch, split):
    mp_map = OrderedDict()
    mp_map["net_name"] = net
    mp_map["mapping"] = OrderedDict()

    # load message passing table
    if split >= 1:
        mp_file = open(MP_DIR + "/" + net + "_" + str(word) + "_" + str(batch) + "/" + str(split) + "/" + net + "_" + str(split) + ".0_mp.csv")
    else:
        mp_file = open(MP_DIR + "/" + net + "_" + str(word) + "_" + str(batch) + "/" + str(split) + "/" + net + "_" + str(split) + "_mp.csv")

    lines = mp_file.read().splitlines()
    mp_file.close()

    mp_map["num_nodes"] = len(lines) - 1

    mp_header = lines[0].split(',')

    for line in lines[1:]:
        line_cols = line.split(',')
        node_name = line_cols[0]

        if node_name not in mp_map["mapping"]:
            mp_map["mapping"][node_name] = OrderedDict()

        for col_idx in range(2, len(line_cols)):
            if col_idx == 2:
                mp_map["mapping"][node_name][line_cols[1]] = OrderedDict()

            mp_map["mapping"][node_name][line_cols[1]][mp_header[col_idx][1:]] = int(line_cols[col_idx])

    return mp_map




def process_net_time_stats(data):
    # calculate total compute time (non-pipe non-parallel)
    t_stats = {}
    t_stats["Total time"] = 0.0
    t_stats["Active time"] = 0.0
    t_stats["Stalled time"] = 0.0

    assert "node_1" in data["node_data"]
    last_node_idx = 1

    for node_idx in range(1, int(data["num_nodes"])+1):
        node_name = "node_" + str(node_idx)

        if node_name not in data["node_data"]:
            node_name = "node_" + str(last_node_idx)
        else:
            last_node_idx = node_idx

        t_stats["Active time"] += data["node_data"][node_name]["Total"]["Active time"]
        t_stats["Stalled time"] += data["node_data"][node_name]["Total"]["Stalled time"]
        t_stats["Total time"] += data["node_data"][node_name]["Total"]["Total time"]


    if "dummy_start_node" in data:
        last_dummy_node_idx = data["dummy_start_node"]

        for node_idx in range(data["dummy_start_node"], int(data["num_nodes"])+1):
            node_name = "node_" + str(node_idx) + "_dummy"

            if node_name not in data["node_data"]:
                node_name = "node_" + str(last_dummy_node_idx) + "_dummy"
            else:
                last_dummy_node_idx = node_idx

            t_stats["Active time"] += data["node_data"][node_name]["Total"]["Active time"]
            t_stats["Stalled time"] += data["node_data"][node_name]["Total"]["Stalled time"]
            t_stats["Total time"] += data["node_data"][node_name]["Total"]["Total time"]

    return t_stats


def process_net_energy_stats(data, t_stats):
    # add total energy and *update* the correct idle energy
    #              which uses the total time we calculated earlier
    system_wide_time = t_stats["Total time"]

    e_stats = {}
    e_stats["Compute active energy"] = 0.0
    e_stats["Register file energy"] = 0.0
    e_stats["Idle energy"] = 0.0
    e_stats["Total Cache Energy"] = 0.0
    e_stats["Total Mem Energy"] = 0.0
    e_stats["Total Energy"] = 0.0
    e_stats["System-wide Idle Energy"] = 0.0
    e_stats["System-wide Total Energy"] = 0.0


    assert "node_1" in data["node_data"]
    last_node_idx = 1

    for node_idx in range(1, int(data["num_nodes"])+1):
        node_name = "node_" + str(node_idx)

        if node_name not in data["node_data"]:
            node_name = "node_" + str(last_node_idx)
        else:
            last_node_idx = node_idx

        e_stats["Compute active energy"] += data["node_data"][node_name]["Total"]["Compute active energy"]
        e_stats["Register file energy"] += data["node_data"][node_name]["Total"]["Register file energy"]
        e_stats["Idle energy"] += data["node_data"][node_name]["Total"]["Idle Energy"]
        e_stats["Total Cache Energy"] += data["node_data"][node_name]["Total"]["Total Cache Energy"]
        e_stats["Total Mem Energy"] += data["node_data"][node_name]["Total"]["Total Mem Energy"]
        e_stats["Total Energy"] += data["node_data"][node_name]["Total"]["Total Energy"]

        node_extra_energy = LKG_PWR * (system_wide_time - data["node_data"][node_name]["Total"]["Total time"])
        e_stats["System-wide Idle Energy"] += node_extra_energy
        e_stats["System-wide Total Energy"] += node_extra_energy + data["node_data"][node_name]["Total"]["Total Energy"]


    if "dummy_start_node" in data:
        last_dummy_node_idx = data["dummy_start_node"]

        for node_idx in range(data["dummy_start_node"], int(data["num_nodes"])+1):
            node_name = "node_" + str(node_idx) + "_dummy"

            if node_name not in data["node_data"]:
                node_name = "node_" + str(last_dummy_node_idx) + "_dummy"
            else:
                last_dummy_node_idx = node_idx

            e_stats["Compute active energy"] += data["node_data"][node_name]["Total"]["Compute active energy"]
            e_stats["Register file energy"] += data["node_data"][node_name]["Total"]["Register file energy"]
            e_stats["Idle energy"] += data["node_data"][node_name]["Total"]["Idle Energy"]
            e_stats["Total Cache Energy"] += data["node_data"][node_name]["Total"]["Total Cache Energy"]
            e_stats["Total Mem Energy"] += data["node_data"][node_name]["Total"]["Total Mem Energy"]
            e_stats["Total Energy"] += data["node_data"][node_name]["Total"]["Total Energy"]

            node_extra_energy = LKG_PWR * (system_wide_time - data["node_data"][node_name]["Total"]["Total time"])
            e_stats["System-wide Total Energy"] += data["node_data"][node_name]["Total"]["Total Energy"]


    return e_stats


def process_net_mem_stats(data):
    # aggregate total mem accesses
    m_stats = {}
    m_stats["Mem Reads"] = 0
    m_stats["Mem Writes"] = 0

    assert "node_1" in data["node_data"]
    last_node_idx = 1

    for node_idx in range(1, int(data["num_nodes"])+1):
        node_name = "node_" + str(node_idx)

        if node_name not in data["node_data"]:
            node_name = "node_" + str(last_node_idx)
        else:
            last_node_idx = node_idx

        m_stats["Mem Reads"] += data["node_data"][node_name]["Total"]["Mem Reads"]
        m_stats["Mem Writes"] += data["node_data"][node_name]["Total"]["Mem Writes"]


    if "dummy_start_node" in data:
        last_dummy_node_idx = data["dummy_start_node"]

        for node_idx in range(data["dummy_start_node"], int(data["num_nodes"])+1):
            node_name = "node_" + str(node_idx) + "_dummy"

            if node_name not in data["node_data"]:
                node_name = "node_" + str(last_dummy_node_idx) + "_dummy"
            else:
                last_dummy_node_idx = node_idx

            m_stats["Mem Reads"] += data["node_data"][node_name]["Total"]["Mem Reads"]
            m_stats["Mem Writes"] += data["node_data"][node_name]["Total"]["Mem Writes"]

    return m_stats


def process_net_mp_cost(spl):
    global RESIDUAL_BYPASS

    mp_cost = {}
    mp_cost["Message Size"] = 0
    mp_cost["Total Message Size"] = 0
    mp_cost["Total Input Size"] = 0
    mp_cost["Total Output Size"] = 0
    mp_cost["Inputs"] = []
    mp_cost["Outputs"] = []
    # mp_cost["Layer Inputs Recv"] = []
    # mp_cost["Layer Outputs Recv"] = []
    # mp_cost["Layer Residual Recv"] = []
    # mp_cost["Layer Inputs Sent"] = []
    # mp_cost["Layer Outputs Sent"] = []
    # mp_cost["Layer Residual Sent"] = []
    mp_cost["is_resnet"] = False
    mp_cost["Bypassing Residual Message Size"] = 0
    mp_cost["Non-bypassing Residual Message Size"] = 0

    num_nodes = spl["num_nodes"]

    first_node = spl["mapping"]["node_1"]
    first_layer = min(first_node, key=lambda k: first_node[k]["Layer Number"])
    if "_i" in first_layer or "_o" in first_layer:
        mp_cost["Message Size"] += 0
    else:
        mp_cost["Message Size"] += first_node[first_layer]["ifmap"]

    if "resnet" in spl["net_name"]:
        end_name = None
        if "resnet50" in spl["net_name"]:
            start_name = 'conv2_1_a'
            end_name = "conv5_3_c"

        mp_cost["is_resnet"] = True

        residual = []
        for node, node_data in spl["mapping"].items():
            for layer_name, layer_data in node_data.items():
                if "conv1" in layer_name or "_c" in layer_name:
                    layer_full_name = layer_name
                    if "part" in layer_name:
                        layer_full_name = layer_name[:layer_name.index("part")-1]

                    if len(residual) == 0 or residual[-1][0] != layer_full_name:
                        residual += [[layer_full_name, layer_data["ofmap"], None, 0],] # [output_layer_name, output_Size, matching_input_layer_name, nodes_passed]
                    else:
                        residual[-1][1] += layer_data["ofmap"]

                if ("_a" in layer_name or "fc1" in layer_name) and start_name not in layer_name:
                    layer_full_name = layer_name
                    if "part" in layer_name:
                        layer_full_name = layer_name[:layer_name.index("part")-1]

                    if residual[0][2] is not None and residual[0][2] != layer_full_name:
                        residual = residual[1:]

                    residual[0][2] = layer_full_name
                    mp_cost["Non-bypassing Residual Message Size"] += residual[0][3] * residual[0][1]
                    mp_cost["Bypassing Residual Message Size"] += residual[0][1] if residual[0][3] != 0 else 0
                    print(residual[0][0] + " passed to " + layer_name + " with res size of " + str(residual[0][1]) + " with " + str(residual[0][3]) + " nodes apart.")

            for i in range(len(residual)):
                if residual[i][2] is None or residual[i][3] < 1:
                    residual[i][3] += 1


    # loop through all the nodes
    for node, node_data in spl["mapping"].items():
        if "dummy" in node:
            continue

        layers = spl["mapping"][node]
        last_layer = max(layers, key=lambda k: layers[k]["Layer Number"])
        first_layer = min(layers, key=lambda k: layers[k]["Layer Number"])

        if first_layer[-2:] == '_i' or first_layer[-2:] == '_o':
            mp_cost["Message Size"] += layers[first_layer]["ifmap"]
            if "embed" in first_layer:
                mp_cost["Message Size"] += 64

        if "embed" in first_layer and "embed" not in last_layer and "part" in first_layer:
            mp_cost["Message Size"] += 131072

        if "embed" not in last_layer:
            mp_cost["Message Size"] += layers[last_layer]["ofmap"]
            mp_cost["Total Output Size"] += layers[last_layer]["ofmap"]
            mp_cost["Outputs"] += [layers[last_layer]["ofmap"],]

        mp_cost["Total Input Size"] += layers[first_layer]["ifmap"]
        mp_cost["Inputs"] += [layers[first_layer]["ifmap"],]
        if "embed" in first_layer and "embed" not in last_layer and "part" in first_layer:
            mp_cost["Total Input Size"] += 131072
            mp_cost["Inputs"][-1] += 131072

    mp_cost["Total Message Size"] = mp_cost["Message Size"] + mp_cost["Bypassing Residual Message Size"]
    mp_cost["Time Cost"] = (LATENCY * 1 + mp_cost["Total Message Size"] / float((BANDWIDTH * 1024**3) / 1e9)) / 1e9
    mp_cost["Energy Cost"] = mp_cost["Total Message Size"] * E_PER_BYTE / 1e12
    mp_cost["Extra Shutdown Energy Cost"] = mp_cost["Time Cost"] * LKG_PWR

    return mp_cost


def preprocess_pipe_parallel_info(data):
    info = {}
    info["group_to_nodes"] = {}
    info["node_to_group"] = {}
    info["group_info"] = {}

    num_nodes = data["num_nodes"]
    node_data = data["node_data"]

    last_layer_combination = []
    max_node_time = 0
    total_node_time = 0
    last_total_time = 0
    has_dummy = False

    group_id = 0

    for node_idx in range(1, num_nodes+1):
        node_name = "node_" + str(node_idx)
        dummy_name = "node_" + str(node_idx) + "_dummy"

        if node_name not in node_data:
            total_node_time += last_total_time
            info["node_to_group"][node_name] = group_id
            info["group_to_nodes"][group_id] += [node_name,]
            if has_dummy:
                info["node_to_group"][dummy_name] = group_id
                info["group_to_nodes"][group_id] += [dummy_name,]
            continue

        last_total_time = 0

        current_layer_combination = []

        for layer, layer_data in node_data[node_name].items():
            if layer == "Total":
                continue

            layer_name = layer
            if "part" in layer:
                index = layer.find("part")
                layer_name = layer[:index-1]
            current_layer_combination += [layer_name,]

        if dummy_name in node_data:
            has_dummy = True
            for layer, layer_data in node_data[dummy_name].items():
                if layer == "Total":
                    continue

                layer_name = layer
                if "part" in layer:
                    index = layer.find("part")
                    layer_name = layer[:index-1]
                current_layer_combination += [layer_name,]

        if current_layer_combination != last_layer_combination:
            if group_id != 0:
                info["group_info"][group_id] = {}
                info["group_info"][group_id]["max_time"] = max_node_time
                max_node_time = 0
                info["group_info"][group_id]["sum_time"] = total_node_time
                total_node_time = 0

            last_layer_combination = current_layer_combination
            group_id += 1

        if group_id not in info["group_to_nodes"] and group_id != 0:
            info["group_to_nodes"][group_id] = []

        info["group_to_nodes"][group_id] += [node_name,]
        if dummy_name in node_data:
            info["group_to_nodes"][group_id] += [dummy_name,]

        max_node_time = max(node_data[node_name]["Total"]["Total time"], max_node_time)
        if dummy_name in node_data:
            max_node_time = max(node_data[dummy_name]["Total"]["Total time"] + node_data[node_name]["Total"]["Total time"], max_node_time)

        total_node_time += node_data[node_name]["Total"]["Total time"]
        last_total_time += node_data[node_name]["Total"]["Total time"]
        if dummy_name in node_data:
            total_node_time += node_data[dummy_name]["Total"]["Total time"]
            last_total_time += node_data[dummy_name]["Total"]["Total time"]

        info["node_to_group"][node_name] = group_id
        if dummy_name in node_data:
            info["node_to_group"][dummy_name] = group_id

    info["group_info"][group_id] = {}
    info["group_info"][group_id]["max_time"] = max_node_time
    info["group_info"][group_id]["sum_time"] = total_node_time

    return info


def process_net_pipe_cost(data, group_info):
    pipe_cost = {}
    pipe_cost["Per Cycle Energy"] = 0.0

    group_data = group_info["group_info"]
    max_group = max(group_data, key = lambda k : group_data[k]["sum_time"])
    max_group_time = group_data[max_group]["sum_time"]
    pipe_cost["time"] = max_group_time
    pipe_cost["throughput"] = 1.0 / max_group_time

    last_seen_node = None

    for node_idx in range(1, data["num_nodes"]+1):
        node_name = "node_" + str(node_idx)

        if node_name not in data["node_data"]:
            node_name = last_seen_node
        else:
            last_seen_node = node_name

        node_data = data["node_data"][node_name]

        pipe_cost["Per Cycle Energy"] += node_data["Total"]["Total Energy"] + (max_group_time - node_data["Total"]["Total time"]) * LKG_PWR

        dummy_name = node_name + "_dummy"

        if dummy_name in data["node_data"]:
            dummy_data = data["node_data"][dummy_name]
            pipe_cost["Per Cycle Energy"] += dummy_data["Total"]["Total Energy"]

    return pipe_cost


def process_net_parallel_cost(data, group_info):
    para_cost = {}
    para_cost["Total Energy"] = 0.0

    group_data = group_info["group_info"]

    total_time = 0.0
    for group, gdata in group_data.items():
        total_time += gdata["max_time"]

    para_cost["time"] = total_time
    para_cost["throughput"] = 1.0 / total_time

    last_seen_node = None

    for node_idx in range(1, data["num_nodes"]+1):
        node_name = "node_" + str(node_idx)

        if node_name not in data["node_data"]:
            node_name = last_seen_node
        else:
            last_seen_node = node_name

        node_data = data["node_data"][node_name]

        para_cost["Total Energy"] += node_data["Total"]["Total Energy"] + (total_time - node_data["Total"]["Total time"]) * LKG_PWR

        dummy_name = node_name + "_dummy"

        if dummy_name in data["node_data"]:
            dummy_data = data["node_data"][dummy_name]
            para_cost["Total Energy"] += dummy_data["Total"]["Total Energy"]

    return para_cost


def process_net_pipe_parallel_cost(data, group_info):
    pipepara_cost = {}
    pipepara_cost["Per Cycle Energy"] = 0.0

    group_data = group_info["group_info"]
    max_group = max(group_data, key = lambda k : group_data[k]["max_time"])
    max_group_time = group_data[max_group]["max_time"]
    pipepara_cost["time"] = max_group_time
    pipepara_cost["throughput"] = 1.0 / max_group_time

    last_seen_node = None

    for node_idx in range(1, data["num_nodes"]+1):
        node_name = "node_" + str(node_idx)

        if node_name not in data["node_data"]:
            node_name = last_seen_node
        else:
            last_seen_node = node_name

        node_data = data["node_data"][node_name]

        pipepara_cost["Per Cycle Energy"] += node_data["Total"]["Total Energy"] + (max_group_time - node_data["Total"]["Total time"]) * LKG_PWR

        dummy_name = node_name + "_dummy"

        if dummy_name in data["node_data"]:
            dummy_data = data["node_data"][dummy_name]
            pipepara_cost["Per Cycle Energy"] += dummy_data["Total"]["Total Energy"]

    return pipepara_cost


def analyze():
    e_cmp = []
    t_cmp = []
    t_lstm = {}
    e_lstm = {}
    e_stats = {}
    msg = []
    num_chips = []

    for net, word, batch in itertools.product(NETS, WORDS, BATCHES):
        for split in SPLIT:
            #outfilename = os.path.join(OUTPUT_DIR, "_".join(net, str(word), str(batch)) + ".csv")
            #fo = open(outfilename, 'a')
            # load the table containing all node info

            print("="*30)
            print("Analyzing " + net + " with (word, batchsize, memsize) = " + str((word, batch, split)))

            # #temp
            # spl = extract_split(net, word, batch, split)
            # mp_cost = process_net_mp_cost(spl)
            # print("Message Passing Cost: ")
            # print("Total Message Size: ", mp_cost["Total Message Size"])
            # print("Total Input Size: ", mp_cost["Total Input Size"])
            # print("Total Output Size: ", mp_cost["Total Output Size"])
            # print("Time: ", mp_cost["Time Cost"])
            # print("Energy: ", mp_cost["Energy Cost"])
            #
            #
            # continue


            res = extract_table(net, word, batch, split)
            spl = extract_split(net, word, batch, split)
            #print(json.dumps(res, indent=2))

            t_stats = process_net_time_stats(res)
            print("Time Cost: ", json.dumps(t_stats, indent=2))
            for entry in t_stats:
                if entry not in t_lstm:
                    t_lstm[entry] = t_stats[entry]
                else:
                    t_lstm[entry] += t_stats[entry]

            e_stats = process_net_energy_stats(res, t_stats)
            print("Energy Cost: ", json.dumps(e_stats, indent=2))
            for entry in e_stats:
                if entry not in e_lstm:
                    e_lstm[entry] = e_stats[entry]
                else:
                    e_lstm[entry] += e_stats[entry]

            m_stats = process_net_mem_stats(res)

            mp_cost = process_net_mp_cost(spl)
            print("Message Passing Cost: ", json.dumps(mp_cost, indent=2))
            msg += [mp_cost["Total Message Size"],]

            if len(e_cmp) != len(SPLIT):
                e_cmp += [e_stats["System-wide Total Energy"] + mp_cost["Energy Cost"] + mp_cost["Extra Shutdown Energy Cost"]]
            else:
                e_cmp[SPLIT.index(split)] += e_stats["System-wide Total Energy"] + mp_cost["Energy Cost"] + mp_cost["Extra Shutdown Energy Cost"]

            if len(t_cmp) != len(SPLIT):
                t_cmp += [t_stats["Total time"] + mp_cost["Time Cost"]]
            else:
                t_cmp[SPLIT.index(split)] += t_stats["Total time"] + mp_cost["Time Cost"]

            group_info = preprocess_pipe_parallel_info(res)
            #print("Group Info:", json.dumps(group_info, indent=2))
            num_chips += [len(group_info["node_to_group"])]

            pipe_info = process_net_pipe_cost(res, group_info)
            print("Pipe Cost: ", json.dumps(pipe_info, indent=2))

            para_info = process_net_parallel_cost(res, group_info)
            print("Parallel Cost: ", json.dumps(para_info, indent=2))

            pipepara_info = process_net_pipe_parallel_cost(res, group_info)
            print("PipeParallel Cost: ", json.dumps(pipepara_info, indent=2))
            
            print("========================================================")

            print("Performance Comparison (automatically aggregated for split LSTM workloads if all parts are declared in the NETS variable):")

            print("Energy Comparison: ")
            print(e_cmp)

            print("Relative Energy Coefficient (divided by the energy used by the last entry in the SPLIT list)")
            print(np.array(e_cmp) / e_cmp[-1])

            print("Time Comparison: ")
            print(t_cmp)

            print("Relative Time Coefficient (divided by the time used by the last entry in the SPLIT list)")
            print(np.array(t_cmp) / t_cmp[-1])

    return e_cmp, t_cmp, msg, num_chips, t_lstm, e_lstm


def main():
    # global BANDWIDTH
    #
    # res = []
    #
    # for i in range(11):
    #     BANDWIDTH = 32 / float(2 ** i)
    #     cmp = analyze()
    #     res.append(cmp)
    #     #res.append(list(np.array(cmp) / cmp[-1]))
    #
    # np.savetxt("temp.csv", np.array(res), delimiter=",")

    e_cmp, t_cmp, msg, num_chips, t_lstm, e_lstm = analyze()
    #print(t_lstm)
    #print(e_lstm)
    #col1 = np.array(e_cmp) / e_cmp[-1]
    #col2 = np.array(t_cmp) / t_cmp[-1]
    #cmp_csv = np.array([col1, col2]).T
    #np.savetxt("tmp/cmp.csv", cmp_csv, delimiter=",")



if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument('--bw', type=int, required=False, default=BANDWIDTH, help="bandwidth of message passing (GB/s)")
    ap.add_argument('--E_per_byte', type=int, required=False, default=E_PER_BYTE, help="energy per byte cost (pJ/Byte)")
    ap.add_argument('--latency', type=int, required=False, default=LATENCY, help="latency before first message (ns)")

    args = ap.parse_args()

    BANDWIDTH = args.bw
    E_PER_BYTE = args.E_per_byte
    LATENCY = args.latency

    main()
