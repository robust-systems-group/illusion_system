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

import os
from pathlib import Path

SCHEDULE_DIR = './schedule'
TRACE_DIR = './trace'
OUTPUT_DIR = './workload'
BATCH_RANGE = [1,]
WORD_RANGE = [16,]

# WORKLOADS = ["resnet50",]
# WORKLOADS = ["alex_net",]
# WORKLOADS = ["vgg_net",]

# langmod workloads need to be separately simulated one at a time
# WORKLOADS = ["lstm_langmod",]
# WORKLOADS = ["lstm_langmod_1",]

NET_ALIAS = 'cnn'
# NET_ALIAS = 'lstm'

WORKLOAD_BASENAME = 'rram_mem'
TEMPLATE_SKIPLINES = 1

def create_schedule(wl_name, batchsize, word):
    net_name = wl_name + "_" + str(word) + "_" + str(batchsize)
    #schedule_subdirs = sorted([ a[0] for a in os.walk(SCHEDULE_DIR + '/' + net_name + '/')][1:])
    # print(schedule_subdirs)
    schedule_subdirs = [SCHEDULE_DIR + '/' + net_name + '/',]

    paths = []
    for subdir in schedule_subdirs:
        for r, d, f in os.walk(subdir, topdown=True):
            for filename in sorted(f):
                if filename[-3:] == 'csv':
                    continue
                path_to_file = os.path.join(r, filename)
                paths.append(path_to_file)

    schedule_path = os.path.join(OUTPUT_DIR, 'schedules')
    Path(schedule_path).mkdir(parents=True, exist_ok=True)

    for pth in paths:
        split_names = pth.split('/')
        task_name = wl_name + "_" + str(word) + "_" + str(batchsize) + "_" + WORKLOAD_BASENAME + '_' + split_names[-2] + '_' + split_names[-1]

        task_path = os.path.join(schedule_path, task_name, NET_ALIAS)
        Path(task_path).mkdir(parents=True, exist_ok=True)
        cmd = 'cp -r ' + pth + ' ' + task_path + '/' + wl_name
        print(cmd)
        os.system(cmd)


def create_trace(wl_name, batchsize, word):
    net_name = wl_name + "_" + str(word) + "_" + str(batchsize)
    #trace_subdirs = sorted([ a[0] for a in os.walk(TRACE_DIR + '/' + net_name + '/')][1:])
    #print(trace_subdirs)
    trace_subdirs = [TRACE_DIR + '/' + net_name + '/',]

    paths = []
    for subdir in trace_subdirs:
        for r, d, f in os.walk(subdir, topdown=True):
            for filename in sorted(f):
                if filename[-3:] == 'csv':
                    continue
                path_to_file = os.path.join(r, filename)
                paths.append(path_to_file)

    trace_path = os.path.join(OUTPUT_DIR, 'traces')
    Path(trace_path).mkdir(parents=True, exist_ok=True)

    for pth in paths:
        split_names = pth.split('/')
        task_name = wl_name + "_" + str(word) + "_" + str(batchsize) + "_" + WORKLOAD_BASENAME + '_' + split_names[-4] + '_' + split_names[-3]

        task_path = os.path.join(trace_path, task_name, NET_ALIAS, split_names[-2])
        Path(task_path).mkdir(parents=True, exist_ok=True)
        cmd = 'cp -r ' + pth + ' ' + task_path + '/' + split_names[-1]
        print(cmd)
        os.system(cmd)


def create_config(wl_name, batchsize, word, print_header):
    net_name = wl_name + "_" + str(word) + "_" + str(batchsize)
    #trace_subdirs = sorted([ a[0] for a in os.walk(TRACE_DIR + '/' + net_name + '/')][1:])
    # print(schedule_subdirs)
    trace_subdirs = [TRACE_DIR + '/' + net_name + '/',]

    paths = []
    for subdir in trace_subdirs:
        for r, d, f in os.walk(subdir, topdown=True):
            for filename in sorted(f):
                if filename[-3:] == 'csv':
                    continue
                path_to_file = os.path.join(r, filename)
                paths.append(path_to_file)

    with open('configs.conf.template', 'r') as conf_t:
        conf_path = os.path.join(OUTPUT_DIR, 'configs.conf')

        if print_header:
            os.system('rm ' + conf_path)

        with open(conf_path, 'a') as conf_w:
            for i in range(TEMPLATE_SKIPLINES):
                l = conf_t.readline()
                if print_header:
                    conf_w.write('    '.join(l.split()) + '\n')

            l = conf_t.readline()
            template_fields = l.split()

            task_names = {}

            for pth in paths:
                split_names = pth.split('/')
                task_name = wl_name + "_" + str(word) + "_" + str(batchsize) + "_" + WORKLOAD_BASENAME + '_' + split_names[-4] + '_' + split_names[-3]
                task_names[task_name] = 1

            for key, _ in sorted(task_names.items()):
                template_fields[0] = key
                template_fields[1] = key
                conf_w.write('    '.join(template_fields) + '\n')


def main():
    os.system('rm -r ' + OUTPUT_DIR + '/*')
    is_first = True
    for wl_name in WORKLOADS:
        for batch in BATCH_RANGE:
            for word in WORD_RANGE:
                create_schedule(wl_name, batch, word)
                create_trace(wl_name, batch, word)
                create_config(wl_name, batch, word, is_first)
                is_first = False

if __name__ == '__main__':
    main()
