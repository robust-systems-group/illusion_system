""" $lic$
Copyright (C) 2019-2020 by The Board of Trustees of Stanford University

This program is free software: you can redistribute it and/or modify it under
the terms of the Modified BSD-3 License as published by the Open Source
Initiative.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the BSD-3 License for more details.

You should have received a copy of the Modified BSD-3 License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>.
"""

'''
Generate operation and access trace from layer mapping.
'''

import argparse
import errno
import itertools
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(HERE, 'op_trace', 'python')))
sys.path.append(os.path.abspath(os.path.join(HERE, 'protoio', 'python')))

from nn_dataflow import DataCategoryEnum as de
from nn_dataflow import Layer
from nn_dataflow import MemHierEnum as me
from nn_dataflow import ParallelEnum as pe
from nn_dataflow import Partition
from nn_dataflow import Partition2dScheme
from nn_dataflow import PhyDim2
from nn_dataflow import Util

from examples import import_network_layers
import eyeriss_search_multi_mc as esm

from op_trace import OpRecord
import protoio

LSTM=False

ADDR_BASE = 0
ADDR_NODE_ALIGN = (1 << 30)

ACCTYPE_READ = 0
ACCTYPE_WRITE = 1
ACCTYPE_UPDATE = 2

OPTYPE_COMPUTE = OpRecord.COMPUTE  # pylint: disable=no-member
OPTYPE_LOAD = OpRecord.LOAD  # pylint: disable=no-member
OPTYPE_STORE = OpRecord.STORE  # pylint: disable=no-member

ERROR = 0.2

MAX_ACC_BYTES_PER_CYCLE = 128

def access_fmap(dim_nodes, partition_addr_bases, word_bytes, nfmap, sfmap,
                partition2d, access_type, n_range, h_range, w_range, b_range,
                fhdl, op_idx0, local_coord, cnt_accessed, nhops_accessed):
    '''
    Issue accesses to the given range (`n/h/w/b_range`) of the i/ofmap.

    `fhdl` is the output trace file handler. `op_idx0` is the starting op ID.
    '''

    def range_overlap(range1, range2):
        ''' Get the overlap range of the two ranges. '''
        return (max(range1[0], range2[0]), min(range1[1], range2[1]))

    def range_len(range_):
        ''' Get length of the range. '''
        return range_[1] - range_[0]

    def range_valid(range_):
        ''' Check if the range is valid (begin < end). '''
        return range_[0] < range_[1]

    def flat_coord(coord):
        ''' Get flat 1-D coordinate. '''
        return coord.h * dim_nodes.w + coord.w

    def get_nhops(coord1, coord2):
        ''' Get number of hops from `coord1` to `coord2`. '''
        return abs(coord1.h - coord2.h) + abs(coord1.w - coord2.w)

    op_idx = op_idx0

    for index in partition2d.gen_all_indexes2d():
        # fmap range for this partition index.
        n_part_rng, h_part_rng, w_part_rng = Partition.get_layer_range(
            nfmap, sfmap, partition2d.partition2d, index)

        # Ranges to access within this partition index.
        n_overlap_rng = range_overlap(n_range, n_part_rng)
        h_overlap_rng = range_overlap(h_range, h_part_rng)
        w_overlap_rng = range_overlap(w_range, w_part_rng)
        if not (range_valid(n_overlap_rng) and range_valid(h_overlap_rng)
                and range_valid(w_overlap_rng)):
            continue

        # Change to offset within this partition index.
        n_overlap_rng = tuple([x - n_part_rng[0] for x in n_overlap_rng])
        h_overlap_rng = tuple([x - h_part_rng[0] for x in h_overlap_rng])
        w_overlap_rng = tuple([x - w_part_rng[0] for x in w_overlap_rng])

        # Physical coordinate.
        coord = partition2d.physical_coordinate2d(index)

        # Base address for this partition.
        addr_base = partition_addr_bases[flat_coord(coord)]

        # Dimension for fmap range in this partition.
        n_dim = range_len(n_part_rng)
        h_dim = range_len(h_part_rng)
        w_dim = range_len(w_part_rng)

        # Dimension for range to access within this partition.
        n_len = range_len(n_overlap_rng)
        h_len = range_len(h_overlap_rng)
        w_len = range_len(w_overlap_rng)

        # Prepare access op. Leave id and addr to set later.
        r = OpRecord()
        if access_type == ACCTYPE_READ:
            r.type = OPTYPE_LOAD
        elif access_type == ACCTYPE_WRITE or access_type == ACCTYPE_UPDATE:
            r.type = OPTYPE_STORE
        else:
            raise ValueError('Unrecognized access type {}'.format(access_type))

        str_ = ''
        cnt_accessed_part = 0
        for bidx in range(*b_range):
            flat_offset_n = bidx * n_dim

            if h_len == h_dim and w_len == w_dim:
                # Short-cut for full h & w access.
                flat_offset = (flat_offset_n + n_overlap_rng[0]) * h_dim * w_dim

                # Issue access.
                r.id = op_idx
                r.addr = addr_base + flat_offset * word_bytes
                r.size = n_len * h_dim * w_dim * word_bytes
                r.delay = Util.idivc(r.size, MAX_ACC_BYTES_PER_CYCLE)
                str_ += protoio.SerializeDelimitedToString(r)
                cnt_accessed_part += r.size / word_bytes
                op_idx += 1

                continue

            for n_offset in range(*n_overlap_rng):
                flat_offset_h = (flat_offset_n + n_offset) * h_dim

                if w_len == w_dim:
                    # Short-cut for full w access.
                    flat_offset = (flat_offset_h + h_overlap_rng[0]) * w_dim

                    # Issue access.
                    r.id = op_idx
                    r.addr = addr_base + flat_offset * word_bytes
                    r.size = h_len * w_dim * word_bytes
                    r.delay = Util.idivc(r.size, MAX_ACC_BYTES_PER_CYCLE)
                    str_ += protoio.SerializeDelimitedToString(r)
                    cnt_accessed_part += r.size / word_bytes
                    op_idx += 1

                    continue

                for h_offset in range(*h_overlap_rng):
                    flat_offset_w = (flat_offset_h + h_offset) * w_dim

                    # Partial continuous w access.
                    flat_offset = flat_offset_w + w_overlap_rng[0]

                    # Issue access.
                    r.id = op_idx
                    r.addr = addr_base + flat_offset * word_bytes
                    r.size = w_len * word_bytes
                    r.delay = Util.idivc(r.size, MAX_ACC_BYTES_PER_CYCLE)
                    str_ += protoio.SerializeDelimitedToString(r)
                    cnt_accessed_part += r.size / word_bytes
                    op_idx += 1

        assert cnt_accessed_part == range_len(b_range) * n_len * h_len * w_len
        cnt_accessed += cnt_accessed_part
        nhops_accessed += cnt_accessed_part * get_nhops(coord, local_coord)

        fhdl.write(str_)

    fhdl.flush()

    return op_idx, cnt_accessed, nhops_accessed


def access_fil(dim_nodes, partition_addr_bases, word_bytes, sfil, nifm,
               part_index, partition2d, n_range_ifm, n_range_ofm,
               fhdl, op_idx0, cnt_accessed, nhops_accessed):
    '''
    Issue accesses to the given range (`n_range_i/ofm`) of the filters.

    `fhdl` is the output trace file handler. `op_idx0` is the starting op ID.
    '''

    def flat_coord(coord):
        ''' Get flat 1-D coordinate. '''
        return coord.h * dim_nodes.w + coord.w

    op_idx = op_idx0

    coord = partition2d.physical_coordinate2d(part_index)
    addr_base = partition_addr_bases[flat_coord(coord)]

    fil_size = sfil * sfil

    str_ = ''
    for nidx_ofm in range(*n_range_ofm):

        flat_offset = (nidx_ofm * nifm + n_range_ifm[0])

        # Issue access.
        r = OpRecord()
        r.id = op_idx
        r.type = OPTYPE_LOAD  # fil is read-only.
        r.addr = addr_base + flat_offset * fil_size * word_bytes
        r.size = (n_range_ifm[1] - n_range_ifm[0]) * fil_size * word_bytes
        r.delay = Util.idivc(r.size, MAX_ACC_BYTES_PER_CYCLE)
        str_ += protoio.SerializeDelimitedToString(r)
        cnt_accessed += r.size / word_bytes
        op_idx += 1

    fhdl.write(str_)

    fhdl.flush()

    # All filters are replicated locally.
    return op_idx, cnt_accessed, nhops_accessed


# http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
def mkdir_p(path):
    ''' Command for `mkdir -p`. '''
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def gen_trace_layer(layer, batch_size, dim_nodes, word_bytes,
                    dict_loop, dict_part,
                    trace_file_prefix, output_is_update, layer_name=None): \
                            # pylint: disable=too-many-locals
    '''
    Based on mapping result `dict_loop` and `dict_part` for `layer`, generate
    trace and write to files, with each partition with physical coordinate
    `coord` in `trace_file_prefix`.`coord`.tr.

    `output_is_update` is whether to only push update to output for in-memory
    accumulation without reading the original output.
    '''

    def flat_coord(coord):
        ''' Get flat 1-D coordinate. '''
        return coord.h * dim_nodes.w + coord.w

    def get_ith_range(idx, full_range, num):
        ''' Divide `full_range` into `num`, and get the `idx`-th. '''
        full_len = full_range[1] - full_range[0]
        return (full_range[0] + idx * full_len / num,
                full_range[0] + (idx + 1) * full_len / num)

    # Loop blocking.
    ti = list(dict_loop['ti'])
    to = list(dict_loop['to'])
    tb = list(dict_loop['tb'])
    orders = list(dict_loop['orders'])
    assert len(orders) == me.NUM

    # Loop blocking at gbuf level.
    ti_outer = ti[0]
    to_outer = to[0]
    tb_outer = tb[0]
    ti_inner = np.prod(ti[1:])
    to_inner = np.prod(to[1:])
    tb_inner = np.prod(tb[1:])
    order_gbuf = orders[me.GBUF]

    def loop_gen():
        '''
        Loop generator according to the order. Always return in order i, o, b.
        '''
        # Access fil every time; access inner-fmap every time; access
        # outer-fmap only when inner-idx is 0. When one t is 1, then only
        # access it when idx of the other one is 0.
        if order_gbuf.index(de.IFM) < order_gbuf.index(de.OFM):
            # Loop ifm inside loop ofm.
            for idx_b, idx_o, idx_i in itertools.product(
                    range(tb_outer), range(to_outer), range(ti_outer)):
                yield idx_i, idx_o, idx_b, \
                        True, \
                        (idx_o == 0 if ti_outer == 1 else True), \
                        (idx_i == 0)
        else:
            assert order_gbuf.index(de.OFM) < order_gbuf.index(de.IFM)
            # Loop ofm inside loop ifm.
            for idx_b, idx_i, idx_o in itertools.product(
                    range(tb_outer), range(ti_outer), range(to_outer)):
                yield idx_i, idx_o, idx_b, \
                        True, \
                        (idx_o == 0), \
                        (idx_i == 0 if to_outer == 1 else True)

    # Partition scheme.
    plcurr = dict_part['part_lcurr']
    plprev = dict_part['part_lprev']
    part_lcurr = Partition2dScheme(plcurr[0], [PhyDim2(*p) for p in plcurr[1]])
    part_lprev = Partition2dScheme(plprev[0], [PhyDim2(*p) for p in plprev[1]])
    ## Verify.

    # Partitioned layer.
    layer_part = Layer(layer.nifm,
                       Util.idivc(layer.nofm,
                                  part_lcurr.partition2d[pe.OUTP].size()),
                       Util.idivc(layer.sofm,
                                  part_lcurr.partition2d[pe.OFMP].h),
                       layer.sfil,
                       layer.strd)

    # Total ti/to should be no more than actually nifm/nofm, given each
    # physical PE set processes multiple ifmaps/ofmaps. We do not need to
    # consider this as long as we use ti/to_outer as the gbuf blocking factors.
    assert ti_outer * ti_inner <= layer_part.nifm
    assert to_outer * to_inner <= layer_part.nofm
    cnt_ifms = Util.idivc(layer_part.nifm, (ti_outer * ti_inner))
    cnt_ofms = Util.idivc(layer_part.nofm, (to_outer * to_inner))
    modeling_overcount_ratio_nifm = cnt_ifms * ti_outer * ti_inner \
            / float(layer_part.nifm)
    modeling_overcount_ratio_nofm = cnt_ofms * to_outer * to_inner \
            / float(layer_part.nofm)
    # If total tb is larger than batch size, it means we have fold_w, i.e.,
    # vertically partitioning fmaps. This means the fmap size is reduced. We
    # always have tb_outer * tb_inner == fold_w * batch_size.
    # We consider this effect as reduction in batch size. For example, if
    # fold_w = 2, then in reality we fetch half-width fmap with batch size
    # tb_inner, and the batch loop trip count is tb_outer / fold_w. Instead, in
    # our model, we capture this as fetch full-width fmap with half of tb_inner
    # (tb_inner / fold_w) as batch size, and the batch loop trip count is
    # tb_outer.
    assert tb_outer * tb_inner >= batch_size
    fold_w = (tb_outer * tb_inner) // batch_size

    print("size, ifm, ofm: ", layer_part.filter_size(), cnt_ifms, cnt_ofms)
    print(dict_loop)

    # Verify the size for the innermost loop after loop blocking.
    usize_gbuf = [0] * de.NUM
    usize_gbuf[de.FIL] = layer_part.filter_size() * cnt_ifms * cnt_ofms
    usize_gbuf[de.IFM] = layer_part.ifmap_size() * cnt_ifms / fold_w
    usize_gbuf[de.OFM] = layer_part.ofmap_size() * cnt_ofms / fold_w
    # Filter margine for ifmap when folding.
    ratio_ifmap_fil_margine = 1 + (layer_part.sfil * (1 - 1./fold_w)) \
            / (layer_part.sifm / fold_w)
    usize_gbuf[de.IFM] *= ratio_ifmap_fil_margine
    # For ifmap with strides, there may be gaps not needed in ofmap.
    ratio_ifmap_gap = max(float(layer_part.ifmap_size())
                          / ((layer_part.sofm * layer_part.sfil)
                             * layer_part.sifm), 1)
    usize_gbuf[de.IFM] /= ratio_ifmap_gap
    usize_gbuf_mapping = list(dict_loop['unit_size'][0])
    # assert max([abs(1.*a/b - 1) for a, b
    #             in zip(usize_gbuf, usize_gbuf_mapping)]) < ERROR, \
    #        'Unit sizes for gbuf in modeling and trace gen do not match ' \
    #        '{} vs. {}'.format(usize_gbuf_mapping, usize_gbuf)

    del layer_part

    # Space allocation.
    def node_addr_base(idx_node):
        ''' Base address for node `idx_node`. '''
        return ADDR_BASE + idx_node * ADDR_NODE_ALIGN
    def data_addr_offset(denum):
        ''' Offset address for data category `denum`. Round up de.NUM to
        power-of-2 and divide ADDR_NODE_ALIGN. '''
        ilog2 = 0
        while (1 << ilog2) < de.NUM:
            ilog2 += 1
        return denum * (ADDR_NODE_ALIGN >> ilog2)
    partition_addr_bases = [[0]*dim_nodes.size() for _ in range(de.NUM)]
    for idx_node in range(dim_nodes.size()):
        for denum in range(de.NUM):
            partition_addr_bases[denum][idx_node] = \
                    node_addr_base(idx_node) + data_addr_offset(denum)

    # Process latency for consuming one gbuf data.
    lat_gbuf = dict_loop['time'] / ti_outer / to_outer / tb_outer

    total_nhops_accessed = [0] * de.NUM
    for index_lcurr in part_lcurr.gen_all_indexes2d():
        # Range for this partition.
        n_ranges = [None] * de.NUM
        h_ranges = [None] * de.NUM
        w_ranges = [None] * de.NUM

        # ofmap range.
        n_ranges[de.OFM], h_ranges[de.OFM], w_ranges[de.OFM] = \
            Partition.get_layer_range(layer.nofm, layer.sofm,
                                      part_lcurr.partition2d, index_lcurr)

        # When sofm is already small, non-dividable partitioning will result in
        # large error. E.g., 4 * 4 / (3 * 3) = 1.77. Account for this when
        # verifying.
        model_ofm_h = Util.idivc(layer.sofm, part_lcurr.partition2d[pe.OFMP].h)
        model_ofm_w = Util.idivc(layer.sofm, part_lcurr.partition2d[pe.OFMP].w)
        real_ofm_h = h_ranges[de.OFM][1] - h_ranges[de.OFM][0]
        real_ofm_w = w_ranges[de.OFM][1] - w_ranges[de.OFM][0]
        model_ifm_h = (model_ofm_h - 1) * layer.strd + layer.sfil
        model_ifm_w = (model_ofm_w - 1) * layer.strd + layer.sfil
        real_ifm_h = (real_ofm_h - 1) * layer.strd + layer.sfil
        real_ifm_w = (real_ofm_w - 1) * layer.strd + layer.sfil
        modeling_overcount_ratio_ofm = modeling_overcount_ratio_nofm \
                * model_ofm_h * model_ofm_w \
                / real_ofm_h / real_ofm_w - 1
        modeling_overcount_ratio_ifm = modeling_overcount_ratio_nifm \
                * model_ifm_h * model_ifm_w \
                / real_ifm_h / real_ifm_w - 1
        modeling_overcount_ratio = max(modeling_overcount_ratio_ofm,
                                       modeling_overcount_ratio_ifm)
        assert modeling_overcount_ratio >= -1e-4

        # ifmap range.
        # ifmap channels. All.
        n_ranges[de.IFM] = (0, layer.nifm)
        # ifmap height tiling.
        # xy_i = xy_o * stride + (0 ... sfil-1)
        h_ranges[de.IFM] = (h_ranges[de.OFM][0] * layer.strd,
                            (h_ranges[de.OFM][1]-1) * layer.strd + layer.sfil)
        assert h_ranges[de.IFM][1] <= layer.sifm
        # ifmap width tiling.
        w_ranges[de.IFM] = (w_ranges[de.OFM][0] * layer.strd,
                            (w_ranges[de.OFM][1]-1) * layer.strd + layer.sfil)
        assert w_ranges[de.IFM][1] <= layer.sifm

        # Physical coordinate.
        coord_lcurr = part_lcurr.physical_coordinate2d(index_lcurr)
        trace_file_name = '{}.{}'.format(trace_file_prefix,
                                         flat_coord(coord_lcurr))

        op_idx = 0
        cnt_accessed = [0] * de.NUM
        nhops_accessed = [0] * de.NUM
        total_lat = 0
        ## Yunfeng: adding some debugging stats
        cnt_acc_ifm = 0
        cnt_acc_ofm = 0
        with open(trace_file_name, 'wb') as fhdl:
            for idx_i, idx_o, idx_b, acc_fil, acc_ifm, acc_ofm in loop_gen():
                cnt_acc_ifm += 1 if acc_ifm else 0
                cnt_acc_ofm += 2 if acc_ofm else 0
                # Batch range. Same for ifmap and ofmap.
                b_rng = get_ith_range(idx_b % batch_size, (0, batch_size),
                                      min(tb_outer, batch_size))
                # ofmap width range.
                w_rng_ofm = get_ith_range(idx_b / batch_size, w_ranges[de.OFM],
                                          max(1, tb_outer / batch_size))
                # ifmap width range.
                w_rng_ifm = (w_rng_ofm[0] * layer.strd,
                             (w_rng_ofm[1] - 1) * layer.strd + layer.sfil)
                assert w_ranges[de.IFM][0] <= w_rng_ifm[0] \
                        and w_rng_ifm[1] <= w_ranges[de.IFM][1]

                # ofmap chn range, the idx_o-th chunk within n_ranges[de.OFM].
                n_rng_ofm = get_ith_range(idx_o, n_ranges[de.OFM], to_outer)
                # ifmap chn range, the idx_i-th chunk within n_ranges[de.IFM].
                n_rng_ifm = get_ith_range(idx_i, n_ranges[de.IFM], ti_outer)

                # First op ID that compute depends on.
                op_idx_dep_start = op_idx

                # Access fil.
                if acc_fil:
                    op_idx, cnt_accessed[de.FIL], nhops_accessed[de.FIL] = \
                        access_fil(dim_nodes, partition_addr_bases[de.FIL],
                                   word_bytes, layer.sfil, layer.nifm,
                                   index_lcurr, part_lcurr,
                                   n_rng_ifm, n_rng_ofm, fhdl, op_idx,
                                   cnt_accessed[de.FIL], nhops_accessed[de.FIL])
                # Access ifmap.
                if acc_ifm:
                    op_idx, cnt_accessed[de.IFM], nhops_accessed[de.IFM] = \
                        access_fmap(dim_nodes, partition_addr_bases[de.IFM],
                                    word_bytes, layer.nifm, layer.sifm,
                                    part_lprev, ACCTYPE_READ,
                                    n_rng_ifm, h_ranges[de.IFM],
                                    w_rng_ifm, b_rng,
                                    fhdl, op_idx, coord_lcurr,
                                    cnt_accessed[de.IFM],
                                    nhops_accessed[de.IFM])
                # Access ofmap.
                if acc_ofm:
                    if not output_is_update:
                        op_idx, cnt_accessed[de.OFM], nhops_accessed[de.OFM] = \
                            access_fmap(dim_nodes, partition_addr_bases[de.OFM],
                                        word_bytes, layer.nofm, layer.sofm,
                                        part_lcurr, ACCTYPE_READ,
                                        n_rng_ofm, h_ranges[de.OFM],
                                        w_rng_ofm, b_rng,
                                        fhdl, op_idx, coord_lcurr,
                                        cnt_accessed[de.OFM],
                                        nhops_accessed[de.OFM])

                # Compute.
                r = OpRecord()
                r.id = op_idx
                r.type = OPTYPE_COMPUTE
                r.delay = lat_gbuf
                for dep_id in range(op_idx_dep_start, op_idx+1):
                    r.dependencies.append(dep_id)

                fhdl.write(protoio.SerializeDelimitedToString(r))
                total_lat += r.delay
                op_idx += 1

                # Access ofmap.
                if acc_ofm:
                    if not output_is_update:
                        op_idx, cnt_accessed[de.OFM], nhops_accessed[de.OFM] = \
                            access_fmap(dim_nodes, partition_addr_bases[de.OFM],
                                        word_bytes, layer.nofm, layer.sofm,
                                        part_lcurr, ACCTYPE_WRITE,
                                        n_rng_ofm, h_ranges[de.OFM],
                                        w_rng_ofm, b_rng,
                                        fhdl, op_idx, coord_lcurr,
                                        cnt_accessed[de.OFM],
                                        nhops_accessed[de.OFM])
                    else:
                        op_idx, cnt_accessed[de.OFM], nhops_accessed[de.OFM] = \
                            access_fmap(dim_nodes, partition_addr_bases[de.OFM],
                                        word_bytes, layer.nofm, layer.sofm,
                                        part_lcurr, ACCTYPE_UPDATE,
                                        n_rng_ofm, h_ranges[de.OFM],
                                        w_rng_ofm, b_rng,
                                        fhdl, op_idx, coord_lcurr,
                                        cnt_accessed[de.OFM],
                                        nhops_accessed[de.OFM])


        if 'trace.0' in trace_file_name:
            print "Trace filename: "
            print(trace_file_name)
            print "Total Op Index: "
            print(op_idx)
            print "Number of OFM accesses: "
            print(cnt_acc_ofm)
            print "Number of IFM accesses: "
            print(cnt_acc_ifm)
        # Account for ifmap filter margine.
        cnt_accessed[de.IFM] *= ratio_ifmap_fil_margine
        nhops_accessed[de.IFM] *= ratio_ifmap_fil_margine
        # Account for ifmap gap.
        cnt_accessed[de.IFM] /= ratio_ifmap_gap
        nhops_accessed[de.IFM] /= ratio_ifmap_gap

        # Account for output update.
        if output_is_update:
            cnt_accessed[de.OFM] *= 2
            nhops_accessed[de.OFM] *= 2

        # Verify access count.
        cnt_accessed_mapping = list(dict_loop['access'][me.DRAM])
        # assert max([abs(1.*b/a - 1) for a, b
        #             in zip(cnt_accessed, cnt_accessed_mapping)]) \
        #        < ERROR + modeling_overcount_ratio, \
        #        'Access counts in modeling and trace gen do not match: ' \
        #        '{} vs. {}'.format(cnt_accessed_mapping, cnt_accessed)

        # Verify latency.
        # assert total_lat == dict_loop['time'], \
        #        'Latencies in modeling and trace gen do not match: ' \
        #        '{} vs. {}'.format(dict_loop['time'], total_lat)
        #
        # total_nhops_accessed = [a + b for a, b in zip(total_nhops_accessed,
        #                                               nhops_accessed)]

    # # Verify num hops.
    # try:
    #     assert abs(float(sum(total_nhops_accessed))
    #                / sum(dict_part['total_nhops']) - 1) < ERROR, \
    #            'Numbers of hops in modeling and trace gen do not match: ' \
    #            '{} vs. {}'.format(dict_part['total_nhops'],
    #                               total_nhops_accessed)
    # except ZeroDivisionError:
    #     assert int(sum(total_nhops_accessed)) == 0


def gen_trace(mapping_dict, trace_dir, output_is_update):
    '''
    Based on mapping result, generate trace and write to files.

    `mapping` is in the same format as outputed by eyeriss_search tool.

    `output_is_update` is whether to only push update to output for in-memory
    accumulation without reading the original output.
    '''
    mappings = mapping_dict['mappings']
    net = mapping_dict['net']
    print net
    node = mapping_dict['node']
    print node
    batch_size = mapping_dict['batch']
    word_bytes = mapping_dict['word'] / 8
    dim_nodes = PhyDim2(*mapping_dict['nodes'])

    net_layers = import_network_layers(net)
    nodes,nodes_reduced = esm.multi_chip(net_layers, 2097152, mapping_dict['memsize']*1024**2,batchsize=batch_size, wsize=word_bytes, is_lstm=LSTM)
    layers = nodes_reduced[node]
    print layers
    assert len(layers) == len(mappings)

    for name, layer in layers.items():
        print "="*30
        print "generating trace for layer: "
        print (layer.nifm, layer.nofm, layer.sifm, layer.sofm, layer.sfil, layer.strd)
        dict_loop = mappings[name][1]
        dict_part = mappings[name][2]
        print "dict loop: "
        print dict_loop
        print "dict part: "
        print dict_part

        trace_dir_layer = os.path.join(trace_dir, name)
        mkdir_p(trace_dir_layer)
        trace_file_prefix = os.path.join(trace_dir_layer, 'trace')

        gen_trace_layer(layer, batch_size, dim_nodes, word_bytes,
                        dict_loop, dict_part,
                        trace_file_prefix, output_is_update, name)


def main(args):
    ''' Main function. '''
    with open(args.mapping_json, 'r') as fh:
        mapping_dict = json.load(fh)
    gen_trace(mapping_dict, args.trace_dir, args.output_is_update)
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()  # pylint: disable=invalid-name

    ap.add_argument('mapping_json',
                    help='json file of the mapping result.')

    ap.add_argument('trace_dir',
                    help='output trace directory.')

    ap.add_argument('--output-is-update', action='store_true',
                    help='whether to only push output update for in-memory '
                         'accumulation without reading the original output.')

    sys.exit(main(ap.parse_args()))
