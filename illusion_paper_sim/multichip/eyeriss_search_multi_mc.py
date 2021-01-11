""" $lic$
Copyright (C) 2016-2020 by The Board of Trustees of Stanford University

This program is free software: you can redistribute it and/or modify it under
the terms of the Modified BSD-3 License as published by the Open Source
Initiative.

If you use this program in your research, we request that you reference the
TETRIS paper ("TETRIS: Scalable and Efficient Neural Network Acceleration with
3D Memory", in ASPLOS'17. April, 2017), and that you send us a citation of your
work.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the BSD-3 License for more details.

You should have received a copy of the Modified BSD-3 License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>.
"""

'''
Search optimal schedule and partitioning.
'''

import argparse
import json
import multiprocessing
import numpy as np
import math
import sys
from collections import OrderedDict

from nn_dataflow import Cost
from nn_dataflow import DataCategoryEnum as de
from nn_dataflow import MemHierEnum as me
from nn_dataflow import Option
from nn_dataflow import PhyDim2
from nn_dataflow import Resource
from nn_dataflow import schedule_search
from nn_dataflow import Layer,FCLayer

from nn_dataflow import MapEyeriss

from examples import import_network_layers

SUPPRESS_PRINT = False

def do_scheduling(nname, layers, args, old_lb=None):
    '''
    Get optimal scheduling for given problem. Return a result schedule.
    '''

    #layers = import_network_layers(args.net)

    batch_size = args.batch
    word = (args.word + 7) / 8

    resource = Resource(dim_nodes=PhyDim2(*args.nodes),
                        dim_array=PhyDim2(*args.array),
                        size_gbuf=args.gbuf/word,
                        size_regf=args.regf/word*np.prod(args.array))

    hier_cost = [0] * me.NUM
    hier_cost[me.DRAM] = args.hier_cost[0]
    hier_cost[me.GBUF] = args.hier_cost[1]
    hier_cost[me.ITCN] = args.hier_cost[2]
    hier_cost[me.REGF] = args.hier_cost[3]
    cost = Cost(cost_memhier=hier_cost,
                cost_nochop=args.hop_cost,
                cost_macop=args.op_cost,
                cost_unit_static=args.unit_static_cost)

    bypass = [True] * de.NUM
    bypass[de.IFM] = 'i' not in args.disable_bypass
    bypass[de.OFM] = 'o' not in args.disable_bypass
    bypass[de.FIL] = 'f' not in args.disable_bypass
    options = Option(allow_gbuf_bypass=bypass,
                     solve_loopblocking=args.solve_loopblocking,
                     hybrid_partition2d=args.hybrid_partition2d,
                     ntops=1,
                     nprocesses=args.processes)

    # Search schedules.
    if old_lb is None:
        tops = schedule_search(layers, batch_size, resource, cost,
                            MapEyeriss.gen_nested_loop_desc, options)
    else:
        tops = schedule_search(layers, batch_size, resource, cost,
                            MapEyeriss.gen_nested_loop_desc, options, old_lb)
    #print tops
    top_mapping = tops[0]

    # Get stats.
    stats = {}
    stats['total_cost'] = top_mapping[0]

    stats['total_time'] = 0
    stats['total_noc_cost'] = 0
    stats['total_ops_per_node'] = 0
    stats['max_dram_bw_per_node'] = 0
    stats['max_dram_bw_layer'] = None
    stats['total_accesses_per_node'] = [0] * me.NUM
    for name in layers.keys():
        layer_top_mapping = top_mapping[1][name]
        layer_dict_loop = layer_top_mapping[1]
        layer_dict_part = layer_top_mapping[2]

        stats['total_time'] += layer_dict_loop['time']
        stats['total_noc_cost'] += layer_dict_part['cost']
        stats['total_ops_per_node'] += layer_dict_loop['ops']
        dram_bw_per_node = sum(layer_dict_loop['access'][me.DRAM]) \
                / float(layer_dict_loop['time'])
        if dram_bw_per_node > stats['max_dram_bw_per_node']:
            stats['max_dram_bw_per_node'] = dram_bw_per_node
            stats['max_dram_bw_layer'] = name
        stats['total_accesses_per_node'] = [
            s + a for s, a in zip(stats['total_accesses_per_node'],
                                  [sum(alist) for alist
                                   in layer_dict_loop['access']])]

    print(stats)

    stats['average_active_pes'] = stats['total_ops_per_node'] \
            / float(stats['total_time'])
    stats['total_static_cost'] = stats['total_time'] * cost.unit_static() \
            * resource.dim_nodes.size()

    sum_cost = 0
    num_nodes = resource.dim_nodes.size()
    sum_cost += stats['total_ops_per_node'] * num_nodes * cost.macop()
    sum_cost += sum([a * c * num_nodes
                     for a, c in zip(stats['total_accesses_per_node'],
                                     cost.memhier())])
    sum_cost += stats['total_static_cost']
    sum_cost += stats['total_noc_cost']
    assert abs(sum_cost / stats['total_cost'] - 1) < 0.001

    # Write results.
    res_map = OrderedDict()
    for argname in ['net', 'batch', 'word', 'nodes', 'array', 'regf', 'gbuf',
                    'op_cost', 'hier_cost', 'hop_cost', 'unit_static_cost',
                    'solve_loopblocking', 'hybrid_partition2d',
                    'disable_bypass','memsize']:
        res_map[argname] = getattr(args, argname)
    for statname in ['total_time', 'total_cost', 'total_static_cost',
                     'total_noc_cost', 'average_active_pes',
                     'max_dram_bw_per_node', 'max_dram_bw_layer',
                     'total_ops_per_node', 'total_accesses_per_node']:
        res_map[statname] = stats[statname]
    res_map['mappings'] = top_mapping[1]
    res_map['node'] = nname

    return res_map

def get_mem_req(layers, batchsize=1, wsize=2):
    mem_req = OrderedDict()
    for lname,layer in layers.items():
        i_mem_layer = batchsize*(layer.total_ifmap_size(wsize))
        o_mem_layer = batchsize*(layer.total_ofmap_size(wsize))
        f_mem_layer = layer.total_filter_size(wsize)
        mem_req[lname] = {'i': i_mem_layer, 'o': o_mem_layer, 'f': f_mem_layer}
    return mem_req

def get_lstm_mem_req(layers, batchsize=1, wsize=2):
    mem_req = OrderedDict()
    for lname, layer in layers.items():
        i_mem_layer = None
        if "embed" in lname:
            i_mem_layer = batchsize*wsize
        else:
            i_mem_layer = batchsize*(layer.total_ifmap_size(wsize))
        o_mem_layer = batchsize*(layer.total_ofmap_size(wsize))
        f_mem_layer = layer.total_filter_size(wsize)

        mem_req[lname] = {'i': i_mem_layer, 'o': o_mem_layer, 'f': f_mem_layer}
    return mem_req

def multi_chip(layers, total_buf, memsize, batchsize=1, wsize=2, splitFC=True, splitConv=True, v=True, is_lstm=False):
    nodes = OrderedDict()
    nodes_reduced = OrderedDict()

    ## Yunfeng: doing some hacking here for splitting up the lstm layers
    if is_lstm:
        print("multichip: splitting up lstm layers")
        lstm_node = OrderedDict()
        # using modified lstm mem req (different embed layer calculation)
        mem_req = get_lstm_mem_req(layers, batchsize, wsize)

        node = 0
        act_idx = 0
        node_mem_left = memsize

        if "embed" in (layers.items())[0][0]:
            act_idx = 1
            # lstm cells starting with an embedding layer is the first lstm cell
            print("multichip: splitting the first lstm cell")
            embed_layer_name = (layers.items())[0][0]
            print("splitting embedding layer " + embed_layer_name)
            # get embed layer
            embed_layer = layers[embed_layer_name]
            # calculate memory requirement for each input (input-split)
            embed_mem_per_output = embed_layer.nifm * embed_layer.filter_size(wsize) * 1024
            # calculate how many inputs a node should have
            output_per_node = int(math.floor(memsize / embed_mem_per_output))
            num_nodes = int(math.ceil(float(embed_layer.nofm) / output_per_node))
            f_nodes = [output_per_node, ] * (num_nodes - 1) + [embed_layer.nofm % output_per_node]
            print(f_nodes)
            for i in range(1, num_nodes + 1):
                node += 1
                nodes['node_' + str(node)] = OrderedDict()
                node_mem_left = memsize

                if i == 1 or i == num_nodes:
                    nodes_reduced['node_' + str(node)] = OrderedDict()
                    if num_nodes == 1:
                        nodes_reduced['node_' + str(node)][embed_layer_name] = Layer(embed_layer.nifm, f_nodes[i-1], embed_layer.sofm, embed_layer.sfil, embed_layer.strd)
                    else:
                        nodes_reduced['node_' + str(node)][embed_layer_name + '_part_' + str(i) + '_i'] = Layer(embed_layer.nifm, f_nodes[i-1], embed_layer.sofm, embed_layer.sfil, embed_layer.strd)

                if num_nodes == 1:
                    nodes['node_' + str(node)][embed_layer_name] = FCLayer(embed_layer.nifm, f_nodes[i-1], embed_layer.sfil)
                else:
                    nodes['node_' + str(node)][embed_layer_name + '_part_' + str(i) + '_i'] = FCLayer(embed_layer.nifm, f_nodes[i-1], embed_layer.sfil)

                node_mem_left -= embed_mem_per_output * f_nodes[i-1]
            ## Now try to fit the rest of the layers in
        else:
            node += 1
            nodes['node_' + str(node)] = OrderedDict()
            nodes_reduced['node_' + str(node)] = OrderedDict()

        print("splitting act & proj layer")
        # get these two layers
        act_layer_name = (layers.items())[act_idx][0]
        proj_layer_name = (layers.items())[act_idx + 1][0]
        act_layer = layers[act_layer_name]
        proj_layer = layers[proj_layer_name]
        # calculate cost for each act-proj pair
        act_mem_per_output = act_layer.nifm * act_layer.filter_size(wsize) * 4
        proj_mem_per_input = proj_layer.nofm * proj_layer.filter_size(wsize)
        # calculate per-slice information
        per_slice_mem = act_mem_per_output + proj_mem_per_input
        total_slices = proj_layer.nifm
        slices_per_node = int(math.floor(float(memsize) / per_slice_mem))
        # print(total_slices, slices_per_node, per_slice_mem)
        # store the split info for automatic conv inferring
        split_ways = []
        dummy_node_idx = []
        # if the rest of the node can fit everything, do it
        if node_mem_left >= total_slices * per_slice_mem:
            nodes_reduced['node_' + str(node)][act_layer_name] = Layer(act_layer.nifm, act_layer.nofm, act_layer.sofm, act_layer.sfil, act_layer.strd)
            nodes_reduced['node_' + str(node)][proj_layer_name] = Layer(proj_layer.nifm, proj_layer.nofm, proj_layer.sofm, proj_layer.sfil, proj_layer.strd)
            nodes['node_' + str(node)][act_layer_name] = Layer(act_layer.nifm, act_layer.nofm, act_layer.sofm, act_layer.sfil, act_layer.strd)
            nodes['node_' + str(node)][proj_layer_name] = Layer(proj_layer.nifm, proj_layer.nofm, proj_layer.sofm, proj_layer.sfil, proj_layer.strd)
            # book keeping
            split_ways += [proj_layer.nifm,]
            dummy_node_idx += [node,]
            node_mem_left -= total_slices * per_slice_mem
        else:
            print(node_mem_left)
            slices_to_fit = int(math.floor(float(node_mem_left) / per_slice_mem))
            part_num = 1
            if slices_to_fit > 0:
                nodes_reduced['node_' + str(node)][act_layer_name + '_part_' + str(part_num) + '_o'] = Layer(act_layer.nifm, slices_to_fit * 4, act_layer.sofm, act_layer.sfil, act_layer.strd)
                nodes_reduced['node_' + str(node)][proj_layer_name + '_part_' + str(part_num) + '_i'] = Layer(slices_to_fit, proj_layer.nofm, proj_layer.sofm, proj_layer.sfil, proj_layer.strd)
                nodes['node_' + str(node)][act_layer_name + '_part_' + str(part_num) + '_o'] = Layer(act_layer.nifm, slices_to_fit * 4, act_layer.sofm, act_layer.sfil, act_layer.strd)
                nodes['node_' + str(node)][proj_layer_name + '_part_' + str(part_num) + '_i'] = Layer(slices_to_fit, proj_layer.nofm, proj_layer.sofm, proj_layer.sfil, proj_layer.strd)
                split_ways += [slices_to_fit,]
                dummy_node_idx += [node,]
            # Now split the rest of the layer
            num_extra_nodes = int(math.ceil(float(total_slices - slices_to_fit) / slices_per_node))
            for i in range(1, num_extra_nodes + 1):
                part_num += 1
                node += 1
                nodes['node_' + str(node)] = OrderedDict()
                if i == 1 and act_idx == 1 and num_extra_nodes > 1:
                    nodes_reduced['node_' + str(node)] = OrderedDict()
                    nodes_reduced['node_' + str(node)][act_layer_name + '_part_' + str(part_num) + '_o'] = Layer(act_layer.nifm, slices_per_node * 4, act_layer.sofm, act_layer.sfil, act_layer.strd)
                    nodes_reduced['node_' + str(node)][proj_layer_name + '_part_' + str(part_num) + '_i'] = Layer(slices_per_node, proj_layer.nofm, proj_layer.sofm, proj_layer.sfil, proj_layer.strd)
                    nodes['node_' + str(node)][act_layer_name + '_part_' + str(part_num) + '_o'] = Layer(act_layer.nifm, slices_per_node * 4, act_layer.sofm, act_layer.sfil, act_layer.strd)
                    nodes['node_' + str(node)][proj_layer_name + '_part_' + str(part_num) + '_i'] = Layer(slices_per_node, proj_layer.nofm, proj_layer.sofm, proj_layer.sfil, proj_layer.strd)
                    split_ways += [slices_per_node,]
                    dummy_node_idx += [node,]
                elif i == num_extra_nodes:
                    slices_left = (total_slices - slices_to_fit) % slices_per_node
                    nodes_reduced['node_' + str(node)] = OrderedDict()
                    nodes_reduced['node_' + str(node)][act_layer_name + '_part_' + str(part_num) + '_o'] = Layer(act_layer.nifm, slices_left * 4, act_layer.sofm, act_layer.sfil, act_layer.strd)
                    nodes_reduced['node_' + str(node)][proj_layer_name + '_part_' + str(part_num) + '_i'] = Layer(slices_left, proj_layer.nofm, proj_layer.sofm, proj_layer.sfil, proj_layer.strd)
                    nodes['node_' + str(node)][act_layer_name + '_part_' + str(part_num) + '_o'] = Layer(act_layer.nifm, slices_left * 4, act_layer.sofm, act_layer.sfil, act_layer.strd)
                    nodes['node_' + str(node)][proj_layer_name + '_part_' + str(part_num) + '_i'] = Layer(slices_left, proj_layer.nofm, proj_layer.sofm, proj_layer.sfil, proj_layer.strd)
                    split_ways += [slices_left,]
                    dummy_node_idx += [node,]
                else:
                    nodes['node_' + str(node)][act_layer_name + '_part_' + str(part_num) + '_o'] = Layer(act_layer.nifm, slices_per_node * 4, act_layer.sofm, act_layer.sfil, act_layer.strd)
                    nodes['node_' + str(node)][proj_layer_name + '_part_' + str(part_num) + '_i'] = Layer(slices_per_node, proj_layer.nofm, proj_layer.sofm, proj_layer.sfil, proj_layer.strd)
                    split_ways += [slices_per_node,]
                    dummy_node_idx += [node,]

        # adding inferred conv layer
        for i in range(len(split_ways)):
            if i == 0 or i == 1 or i == (len(split_ways) - 1):
                nodes_reduced['node_' + str(dummy_node_idx[i]) + "_dummy"] = OrderedDict()
                nodes_reduced['node_' + str(dummy_node_idx[i]) + "_dummy"]['conv_part_' + str(i + 1)] = FCLayer(1, batchsize * split_ways[i], 1)
            nodes['node_' + str(dummy_node_idx[i]) + "_dummy"] = OrderedDict()
            nodes['node_' + str(dummy_node_idx[i]) + "_dummy"]['conv_part_' + str(i + 1)] = FCLayer(1, batchsize * split_ways[i], 1)

        return nodes, nodes_reduced

    # otherwise just split it as a non-lstm net

    mem_req = get_mem_req(layers, batchsize, wsize)

    #Check that we can fit inputs/ouputs in buffer (concerv. assume 1/2 total buffer)
    max_io_map = 0
    layer_max = ''
    for lname,layer in layers.items():
        max_io_map = max(max_io_map, mem_req[lname]['i']+mem_req[lname]['o'])
        if max_io_map == mem_req[lname]['i']+mem_req[lname]['o']:
            layer_max = lname
    if (max_io_map > total_buf):
        if not SUPPRESS_PRINT:
            print "Warning: need bigger buffer size"
        #raise NameError(layer_max + " requires: " + str(max_io_map) + " for max layer IOs, but buffer only: " + str(total_buf))

    # start splitting

    node = 1
    node_mem_left = memsize

    for lname, layer in layers.items():
        layer_mem_req = mem_req[lname]['f']

        # if layer fits in the node, everyone is happy
        if layer_mem_req <= node_mem_left:
            node_name = "node_" + str(node)

            node_mem_left -= layer_mem_req

            if node_name not in nodes:
                nodes[node_name] = OrderedDict()
            nodes[node_name][lname] = layer

            if node_name not in nodes_reduced:
                nodes_reduced[node_name] = OrderedDict()
            nodes_reduced[node_name][lname] = layer

        # otherwise we will need to split the layer
        else:
            # check conv layer
            if layer.sofm != 1:
                if splitConv:
                    if layer.nofm >= layer.nifm:
                        ofm_size = mem_req[lname]['f'] / layer.nofm
                        split_ofm = []
                        remaining_ofm = layer.nofm

                        while remaining_ofm > 0:
                            if (node_mem_left < ofm_size):
                                node += 1
                                node_mem_left = memsize

                            layer_ofm = min(int(math.floor(node_mem_left / ofm_size)), remaining_ofm)
                            node_mem_left -= layer_ofm * ofm_size
                            split_ofm += [(layer_ofm, node),]
                            remaining_ofm -= layer_ofm

                        for idx in range(len(split_ofm)):
                            lofm, node_idx = split_ofm[idx]
                            node_name = "node_" + str(node_idx)

                            if node_name not in nodes:
                                nodes[node_name] = OrderedDict()

                            layer_name = lname + "_part_" + str(idx+1) + "_o"

                            nodes[node_name][layer_name] = Layer(layer.nifm, lofm, layer.sofm, layer.sfil, layer.strd)
                            if idx <= 1 or lofm != split_ofm[idx-1][0]:
                                if node_name not in nodes_reduced:
                                    nodes_reduced[node_name] = OrderedDict()
                                nodes_reduced[node_name][layer_name] = Layer(layer.nifm, lofm, layer.sofm, layer.sfil, layer.strd)

                    else:
                        ifm_size = mem_req[lname]['f'] / layer.nifm
                        split_ifm = []
                        remaining_ifm = layer.nifm

                        while remaining_ifm > 0:
                            if (node_mem_left < ifm_size):
                                node += 1
                                node_mem_left = memsize

                            layer_ifm = min(int(math.floor(node_mem_left / ifm_size)), remaining_ifm)
                            node_mem_left -= layer_ifm * ifm_size
                            split_ifm += [(layer_ifm, node),]
                            remaining_ifm -= layer_ifm

                        for idx in range(len(split_ifm)):
                            lifm, node_idx = split_ifm[idx]
                            node_name = "node_" + str(node_idx)

                            if node_name not in nodes:
                                nodes[node_name] = OrderedDict()

                            layer_name = lname + "_part_" + str(idx+1) + "_i"

                            nodes[node_name][layer_name] = Layer(lifm, layer.nofm, layer.sofm, layer.sfil, layer.strd)
                            if idx <= 1 or lifm != split_ifm[idx-1][0]:
                                if node_name not in nodes_reduced:
                                    nodes_reduced[node_name] = OrderedDict()
                                nodes_reduced[node_name][layer_name] = Layer(lifm, layer.nofm, layer.sofm, layer.sfil, layer.strd)

                else:
                    raise NameError("Cannot fit layer " + str(lname) + " without Conv Layer Splitting")

            else:
                if splitFC:
                    if layer.nofm >= layer.nifm:
                        ofm_size = mem_req[lname]['f'] / layer.nofm
                        split_ofm = []
                        remaining_ofm = layer.nofm

                        while remaining_ofm > 0:
                            if (node_mem_left < ofm_size):
                                node += 1
                                node_mem_left = memsize

                            layer_ofm = min(int(math.floor(node_mem_left / ofm_size)), remaining_ofm)
                            node_mem_left -= layer_ofm * ofm_size
                            split_ofm += [(layer_ofm, node),]
                            remaining_ofm -= layer_ofm

                        for idx in range(len(split_ofm)):
                            lofm, node_idx = split_ofm[idx]
                            node_name = "node_" + str(node_idx)

                            if node_name not in nodes:
                                nodes[node_name] = OrderedDict()

                            layer_name = lname + "_part_" + str(idx+1) + "_o"

                            nodes[node_name][layer_name] = Layer(layer.nifm, lofm, layer.sofm, layer.sfil, layer.strd)
                            if idx <= 1 or lofm != split_ofm[idx-1][0]:
                                if node_name not in nodes_reduced:
                                    nodes_reduced[node_name] = OrderedDict()
                                nodes_reduced[node_name][layer_name] = Layer(layer.nifm, lofm, layer.sofm, layer.sfil, layer.strd)

                    else:
                        ifm_size = mem_req[lname]['f'] / layer.nifm
                        split_ifm = []
                        remaining_ifm = layer.nifm

                        while remaining_ifm > 0:
                            if (node_mem_left < ifm_size):
                                node += 1
                                node_mem_left = memsize

                            layer_ifm = min(int(math.floor(node_mem_left / ifm_size)), remaining_ifm)
                            node_mem_left -= layer_ifm * ifm_size
                            split_ifm += [(layer_ifm, node),]
                            remaining_ifm -= layer_ifm

                        for idx in range(len(split_ifm)):
                            lifm, node_idx = split_ifm[idx]
                            node_name = "node_" + str(node_idx)

                            if node_name not in nodes:
                                nodes[node_name] = OrderedDict()

                            layer_name = lname + "_part_" + str(idx+1) + "_i"

                            nodes[node_name][layer_name] = Layer(lifm, layer.nofm, layer.sofm, layer.sfil, layer.strd)
                            if idx <= 1 or lifm != split_ifm[idx-1][0]:
                                if node_name not in nodes_reduced:
                                    nodes_reduced[node_name] = OrderedDict()
                                nodes_reduced[node_name][layer_name] = Layer(lifm, layer.nofm, layer.sofm, layer.sfil, layer.strd)
                else:
                    raise NameError("Cannot fit layer " + str(lname) + " without FC Layer Splitting")

    return nodes, nodes_reduced



def printv(string, v):
    if v:
        print string

def main(args):
    ''' Main function. '''
    print "Working on Net" + str(args.net)
    # deal with input arguments
    layers = import_network_layers(args.net)
    total_buf = args.regf*args.array[0]*args.array[1]*args.nodes[0]*args.nodes[1] + args.gbuf
    wsize = int(args.word/8.0)
    # Compute total requirement so that we can know the problem size
    mem_reqs = get_mem_req(layers, args.batch, wsize)
    total_mem_req = sum(mem_reqs[lname]['f'] for lname, _ in layers.items())
    # placeholders
    num_nodes = 0
    nodes = None
    nodes_reduced = None
    # if no num_chip is set, split using chip memsize
    if (args.multi_chip is None):
        nodes,nodes_reduced = multi_chip(layers, total_buf, args.memsize*1024**2, batchsize=args.batch, wsize=wsize, v=True, is_lstm=args.lstm)
    else:
        print "Searching for #nodes = " + str(args.multi_chip)
        new_memsize = int(math.ceil(float(total_mem_req) / args.multi_chip / (1024**2)) * 1024**2)
        print "Starting with memsize = " + str(new_memsize)
        incr = int(1 * 1024**2)
        # flags: there is no solution at the incr step we are taking if the script starts incr + decr back & forth
        incremented = False
        decremented = False
        while (num_nodes != args.multi_chip):
            # set search bounds
            if new_memsize <= 0 or new_memsize > total_mem_req and args.multi_chip != 1:
                if new_memsize <= incr:
                    new_memsize = incr
                print "No split found"
                break
            # get results and see if we should incr or decr the memsize
            nodes,nodes_reduced = multi_chip(layers, total_buf, new_memsize, batchsize=args.batch, wsize=wsize, v=True, is_lstm=args.lstm)
            num_nodes = len(nodes)
            # case for exact split found
            if (num_nodes == args.multi_chip):
                print "Found split for #nodes = " + str(args.multi_chip) + ", memsize = " + str(new_memsize)
                break
            # case for memsize too small
            if (num_nodes > args.multi_chip):
                print "Incrementing memsize.."
                new_memsize += incr
                incremented = True
                # end condition to prevent jumping back and forth
                if (decremented):
                    print "Cannot achieve desired #nodes = " + str(args.multi_chip)
                    break
            # case for memsize too large
            else:
                print "Decrementing memsize.."
                new_memsize -= incr
                decremented = True
                # end condition to prevent jumping back and forth
                if (incremented):
                    print "Cannot achieve desired #nodes = " + str(args.multi_chip)
                    break
        # if the new memsize ever gets down to 0, set it back to the minimum step
        if new_memsize <= incr:
            new_memsize = incr
        # set the memsize to the result we searched for the multichip option
        args.memsize = int(new_memsize / (1024**2))
        print "Scheduling using memsize = " + str(args.memsize)
        nodes,nodes_reduced = multi_chip(layers, total_buf, args.memsize*1024**2, batchsize=args.batch, wsize=wsize, v=True, is_lstm=args.lstm)
        ## print nodes
        print "Planning complete with #nodes = " + str(len(nodes))

    print("nodes: ", nodes)
    print("nodes_reduced: ", nodes_reduced)

    if args.message_passing:
        with open(args.fp+str(args.net)+'_'+str(args.memsize)+'_mp.csv', "w+") as fo:
            fo.write("Node Name, Layer Name, Layer Number, ifmap, ofmap, fmap\n")
            for nname,node in nodes.items():
                lnum = 1
                for lname,layer in node.items():
                    fo.write(nname+','+lname+','+str(lnum)+','+str(layer.total_ifmap_size(wsize)*args.batch)+','+str(layer.total_ofmap_size(wsize)*args.batch)+','+str(layer.total_filter_size(wsize))+'\n')
                    lnum +=1
        fo.close()
        return 0

    old_lb_map = None

    if True:
        print("Retrieving the original singlechip schedule...")
        old_lb_map = {}
        old_scheduling_res = do_scheduling("old", layers, args)
        old_mapping = old_scheduling_res["mappings"]
        for lname, lmap in old_mapping.items():
            temp_map = {}
            temp_map["ti"] = lmap[1]["ti"]
            temp_map["to"] = lmap[1]["to"]
            temp_map["tb"] = None
            temp_map["orders"] = None
            old_lb_map[lname] = temp_map
        print(old_lb_map)

    for nname,node in nodes_reduced.items():
        sys.stdout.write('Scheduling ' +str(nname)+'\n')
        print(nname, node)
        with open(args.fp+nname, "w+") as fo:
            scheduling_res = do_scheduling(nname, node, args, old_lb_map)
            json.dump(scheduling_res, fo, indent=2)
            fo.close()
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()  # pylint: disable=invalid-name

    ap.add_argument('net',
                    help='network name, should be a .py file under examples')

    ap.add_argument('fp', help='File path for JSON dumps')

    ap.add_argument('--batch', type=int, required=True,
                    help='batch size')
    ap.add_argument('--word', type=int, default=16,
                    help='word size in bits')

    ap.add_argument('--nodes', type=int, nargs=2, required=True,
                    metavar=('H', 'W'),
                    help='Parallel node partitioning dimensions')
    ap.add_argument('--array', type=int, nargs=2, required=True,
                    metavar=('H', 'W'),
                    help='PE array dimensions')

    # Yunfeng: adding new multichip config here
    ap.add_argument('--multi_chip', type=int, default=None,
                    help='lower & upper bound for multichip')
    # Yunfeng: adding lstm config here
    ap.add_argument('--lstm', type=bool, default=False,
                    help='set to True when partitioning lstm cells')

    ap.add_argument('--regf', type=int, required=True,
                    help='register file size in bytes per PE')
    ap.add_argument('--gbuf', type=int, required=True,
                    help='global buffer size in bytes')

    ap.add_argument('--op-cost', type=float, default=1,
                    help='cost of arithmetic operation')
    ap.add_argument('--hier-cost', type=float, nargs=4, default=[200, 6, 2, 1],
                    metavar=('DRAM_COST', 'GBUF_COST', 'ITCN_COST',
                             'REGF_COST'),
                    help='cost of access to memory hierarchy')
    ap.add_argument('--hop-cost', type=float, default=100,
                    help='cost of access through one NoC hop')
    ap.add_argument('--unit-static-cost', type=float, default=0,
                    help='static cost for unit execution time')

    ap.add_argument('--disable-bypass', nargs='*', default=[],
                    choices=['i', 'o', 'f'],
                    help='whether disallowing gbuf bypass for i (input), o '
                         '(output), or f (filter)')

    ap.add_argument('--solve-loopblocking', action='store_true',
                    help='Use analytical solver to choose loop blocking. '
                         'Otherwise use exhaustive search.')
    ap.add_argument('--message-passing', action='store_true',help="Spit out message passing statistics")
    ap.add_argument('--hybrid-partition2d', action='store_true',
                    help='Use hybrid partition for layer for node mapping. '
                         'Otherwise use naive method based on layer type.')

    ap.add_argument('-p', '--processes', type=int,
                    default=multiprocessing.cpu_count()/2,
                    help='Number of parallel processes to use for search.')

    ap.add_argument('--memsize',type=float, default = 4*(1024**3),help='total memory per node')
    ap.add_argument('--memrange', type=int, default = 0, help="enable ranged memory auto generation")

    ap.add_argument('-v','--verbose',type=bool, default=False,help='Verbose for node mapping')

    sys.exit(main(ap.parse_args()))
