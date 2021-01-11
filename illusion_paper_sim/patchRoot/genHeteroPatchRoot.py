#!/usr/bin/python
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


import os, sys
import argparse
import shutil
import string


class XTemplate(string.Template):
    delimiter = '$'
    escaped = '$$'

longMaskFmt=False
maskSize = 256
def getMask(start, end):
    cur = 0
    l = []
    for i in range(maskSize):
        j = i % 32
        if i >= start and i < end: cur |= 1 << j
        if (i + 1) % 32 == 0:
            l.append(cur)
            cur = 0
    l.reverse()
    return ','.join('%08x' % n for n in l)

def getList(start, end):
    if end - start == 1:
        return str(start)
    else:
        return str(start) + '-' + str(end-1)


parser = argparse.ArgumentParser(
        description='Generate patch root for heterogeneous system')

parser.add_argument('--bc', type=int, default=1,
        help='Number of big cores')
parser.add_argument('--lc', type=int, default=0,
        help='Number of little cores')
parser.add_argument('--bn', type=int, default=1,
        help='Number of NUMA nodes for big cores')
parser.add_argument('--ln', type=int, default=0,
        help='Number of NUMA nodes for little cores')

parser.add_argument('--no-little-core-memory', action='store_true',
        help='Whether the little cores share the same memory as big cores, '
        'i.e., no normal memory for little cores')

parser.add_argument('--dir', '-d', type=str, default='patchRoot',
        help='Destination directory')


args = parser.parse_args()

bcores = args.bc
lcores = args.lc
bnodes = args.bn
lnodes = args.ln
no_little_core_memory = args.no_little_core_memory
root = os.path.abspath(args.dir)

ncores = bcores + lcores
nnodes = bnodes + lnodes

progDir = os.path.dirname(os.path.abspath(__file__))

if ncores < 1:
    print 'ERROR: Need >= 1 cores!'
    sys.exit(1)

if ncores > maskSize:
    print 'WARN: These many cpus have not been tested, x2APIC systems may be different...'
    if ncores > 2048:
        print 'ERROR: Too many cores, currently support up to 2048'
        sys.exit(1)
    print 'WARN: Switch to long mask format, up to 2048 cores'
    longMaskFmt = True
    maskSize = 2048

if bcores != 0 and bcores % bnodes != 0:
    print 'ERROR: {} big cores must be evenly distributed among {} NUMA nodes!'.format(bcores, bnodes)
    sys.exit(1)
bcpern = bcores / bnodes if bcores != 0 else 0

if lcores != 0 and lcores % lnodes != 0:
    print 'ERROR: {} little cores must be evenly distributed among {} NUMA nodes!'.format(lcores, lnodes)
    sys.exit(1)
lcpern = lcores / lnodes if lcores != 0 else 0

if os.path.exists(root):
    print 'ERROR: Directory {} already exists, aborting'.format(root)
    sys.exit(1)

os.makedirs(root)
if not os.path.exists(root):
    print 'ERROR: Could not create {}, aborting'.format(root)
    sys.exit(1)

print 'Will produce a tree for {}/{} big/little cores with {}/{} NUMA nodes in {}'.format(
        bcores, lcores, bnodes, lnodes, root)


## /proc

rootproc = os.path.join(root, 'proc')
os.makedirs(rootproc)

# cpuinfo
cpuinfoBigTemplate = XTemplate(open(os.path.join(progDir, 'cpuinfo.template'), 'r').read())
try:
    cpuinfoLittleTemplate = XTemplate(open(os.path.join(progDir, 'cpuinfo.little.template'), 'r').read())
except:
    # Use the same cpuinfo template for big and little cores.
    cpuinfoLittleTemplate = XTemplate(open(os.path.join(progDir, 'cpuinfo.template'), 'r').read())

with open(os.path.join(rootproc, 'cpuinfo'), 'w') as fh:
    for cpu in range(bcores):
        print >>fh, cpuinfoBigTemplate.substitute({'CPU' : str(cpu), 'NCPUS' : ncores}),
    for cpu in range(bcores, ncores):
        print >>fh, cpuinfoLittleTemplate.substitute({'CPU' : str(cpu), 'NCPUS' : ncores}),

# stat
statTemplate = XTemplate(open(os.path.join(progDir, 'stat.template'), 'r').read())

cpuAct = [int(x) for x in '665084 119979939 9019834 399242499 472611 20 159543 0 0 0'.split(' ')]
totalAct = [x * ncores for x in cpuAct]

with open(os.path.join(rootproc, 'stat'), 'w') as fh:
    cpuStat = 'cpu  ' + ' '.join([str(x) for x in totalAct])
    for cpu in range(ncores):
        cpuStat += '\ncpu{} '.format(cpu) + ' '.join([str(x) for x in cpuAct])
    print >>fh, statTemplate.substitute({'CPUSTAT' : cpuStat}),

# self/status
os.makedirs(os.path.join(rootproc, 'self'))
with open(os.path.join(rootproc, 'self', 'status'), 'w') as fh:
    # FIXME: only for CPU/memory list
    print >>fh, '...'
    print >>fh, 'Cpus_allowed:\t' + getMask(0, ncores)
    print >>fh, 'Cpus_allowed_list:\t' + getList(0, ncores)
    print >>fh, 'Mems_allowed:\t' + getMask(0, nnodes)
    print >>fh, 'Mems_allowed_list:\t' + getList(0, nnodes)
    print >>fh, '...'

## /sys

rootsys = os.path.join(root, 'sys')
os.makedirs(rootsys)

# cpus
cpuDir = os.path.join(rootsys, 'devices', 'system', 'cpu')
os.makedirs(cpuDir)

for f in ['online', 'possible', 'present']:
    with open(os.path.join(cpuDir, f), 'w') as fh:
        print >>fh, getList(0, ncores)
with open(os.path.join(cpuDir, 'offline'), 'w') as fh:
    print >>fh, ''
with open(os.path.join(cpuDir, 'sched_mc_power_savings'), 'w') as fh:
    print >>fh, 0
with open(os.path.join(cpuDir, 'kernel_max'), 'w') as fh:
    print >>fh, maskSize-1

for (cores, nodes, cpern) in [(range(bcores), range(bnodes), bcpern),
        (range(bcores, ncores), range(bnodes, nnodes), lcpern)]:
    for cpu in cores:
        c = cpu - cores[0]  # cid within group
        n = c / cpern       # nid within group
        node = n + nodes[0]
        coreSiblings = (cores[0] + n*cpern, cores[0] + (n+1)*cpern)

        d = os.path.join(cpuDir, 'cpu{}'.format(cpu))
        td = os.path.join(d, 'topology')
        os.makedirs(d)
        os.makedirs(td)

        with open(os.path.join(td, 'core_id'), 'w') as fh:
            print >>fh, cpu
        with open(os.path.join(td, 'physical_package_id'), 'w') as fh:
            print >>fh, node
        with open(os.path.join(td, 'core_siblings'), 'w') as fh:
            print >>fh, getMask(*coreSiblings)
        with open(os.path.join(td, 'core_siblings_list'), 'w') as fh:
            print >>fh, getList(*coreSiblings)
        with open(os.path.join(td, 'thread_siblings'), 'w') as fh:
            # FIXME: assume single-thread core
            print >>fh, getMask(cpu, cpu+1)
        with open(os.path.join(td, 'thread_siblings_list'), 'w') as fh:
            # FIXME: assume single-thread core
            print >>fh, getList(cpu, cpu+1)
        with open(os.path.join(d, 'online'), 'w') as fh:
            print >>fh, 1

# nodes
nodeDir = os.path.join(rootsys, 'devices', 'system', 'node')
os.makedirs(nodeDir)

for f in ['online', 'possible']:
    with open(os.path.join(nodeDir, f), 'w') as fh:
        print >>fh, getList(0, nnodes)
with open(os.path.join(nodeDir, 'has_normal_memory'), 'w') as fh:
    if no_little_core_memory:
        print >>fh, getList(0, bnodes)
    else:
        print >>fh, getList(0, nnodes)
with open(os.path.join(nodeDir, 'has_cpu'), 'w') as fh:
    print >>fh, getList(0, nnodes) if nnodes > 1 else ''

meminfoTemplate = XTemplate(open(os.path.join(progDir, 'nodeFiles', 'meminfo.template'), 'r').read())

for (cores, nodes, cpern) in [(range(bcores), range(bnodes), bcpern),
        (range(bcores, ncores), range(bnodes, nnodes), lcpern)]:
    for node in nodes:
        n = node - nodes[0]  # nid within group
        coreSiblings = (cores[0] + n*cpern, cores[0] + (n+1)*cpern)

        d = os.path.join(nodeDir, 'node{}'.format(node))
        os.makedirs(d)

        for cpu in range(*coreSiblings):
            os.symlink(os.path.relpath(os.path.join(cpuDir, 'cpu{}'.format(cpu)), d), os.path.join(d, 'cpu{}'.format(cpu)))

        for f in ['numastat', 'scan_unevictable_pages', 'vmstat']:
            shutil.copy(os.path.join(progDir, 'nodeFiles', f), d)

        with open(os.path.join(d, 'cpumap'), 'w') as fh:
            print >>fh, getMask(*coreSiblings)
        with open(os.path.join(d, 'cpulist'), 'w') as fh:
            print >>fh, getList(*coreSiblings)
        with open(os.path.join(d, 'meminfo'), 'w') as fh:
            print >>fh, meminfoTemplate.substitute({'NODE' : str(node)}),
        with open(os.path.join(d, 'distance'), 'w') as fh:
            for node2 in range(nnodes):
                print >>fh, ('10' if node2 == node else '20') + ' ',
            print >>fh, ''

# misc
os.makedirs(os.path.join(rootsys, 'bus', 'pci', 'devices'))


## make read-only

for (p, ds, fs) in os.walk(root):
    for d in ds:
        os.chmod(os.path.join(p, d), 0555)
    for f in fs:
        os.chmod(os.path.join(p, f), 0444)

