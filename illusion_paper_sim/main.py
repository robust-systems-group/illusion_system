#!/usr/bin/env py50n

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

import time
import sys, os
from template import *
from parallel import SSHPool
from collections import OrderedDict
from util import *

# add the modules dir, which contains nn_dataflow, to PYTHONPATH for subprocs
# that we spawn
modulesDir = os.getcwd()+'/modules'
prevPYTHONPATH = os.environ.get('PYTHONPATH', '')
os.environ['PYTHONPATH'] = prevPYTHONPATH+':'+modulesDir
#sys.path += [cwd+'/modules']
PWD = os.getcwd()

# TODO put in defines.py
CNN_PARTS = ['']
LSTM_PARTS = ['']
GEN_STRING_BATCH_SIZE = 1   # NOTE: really handle "unroll size" in the multiplier

# TODO: to coarse-parallelize entire process, transpose `for c in configs` loops
# across generate fns; then, handle one config on one worker

sshPool = SSHPool(8)

# TODO: parameterize multipliers


# returns a dict-of-dicts mapping a configuration's name to its associated params
def parseConfigsTXT(txt):
    with open(txt, 'r') as txtFile:
        lines = txtFile.read().splitlines() # gets rid of \n
    fieldNames = lines[0].split()
    entries = lines[1:]

    # use an OrderedDict to maintain the order of entries in configs file
    configs = OrderedDict()
    for eStr in entries:
        eArr = eStr.split()
        eTyped = []
        # if commented out, skip.
        if eArr[0][0] == '#':
            continue

        # convert parameters to int types as necessary
        for i, e in enumerate(eArr):
            try:
                te = float(e)       # assume fractional
                if te % 1 == 0:
                    te = int(te)    # was an int
                eTyped.append(te)
            except ValueError:
                if ',' in e:
                    l = e.split(',')                # was a list
                    if l[-1] == '':                 # chop off any trailing ''
                        l = l[:-1]
                    eTyped.append(l)
                else:
                    eTyped.append(e)                # was a str

        params = {fieldNames[i]: eTyped[i] for i in range(len(fieldNames))}
        configs[params['name']] = params

    return configs



def buildScheduleGenString(batchSize, wordSize, nodeXY, arrayXY,
        localBufferSize, globalBufferSize, networkName):
    s = "/usr/bin/python2 modules/nn_dataflow/eyeriss_search.py --batch %d"\
        " --word %d --nodes %d %d --array %d %d --regf %d --gbuf %d --op-cost 1"\
        " --hier-cost 200 6 2 1 --hop-cost 0 --unit-static-cost 1 %s"\
        " --disable-bypass 'i' 'o' 'f'" % (batchSize, wordSize, nodeXY,
        nodeXY, arrayXY, arrayXY, localBufferSize, globalBufferSize, networkName)

    return s

#python $ZSIMPATH/misc/opTrace/test/gen_trace.py $1 $2
def buildTraceGenString(scheduleFile, tracesDir):
    s = "/usr/bin/python2 deps/ORNL-zsim-logic/misc/opTrace/test/gen_trace.py"\
        " %s %s" % (scheduleFile, tracesDir)

    return s

# TODO redo _temp to include batch size dependency
def generateSchedules(configs, workloads):
    for c in configs:
        p = configs[c]      # params

        for w in workloads:
            W = workloads[w]

            schedulesDir = './schedules/%s/%s' % (p['name'], W['alias'])
            touchDir(schedulesDir)

            for part in W['parts']:
                ## Before generating schedule, if need be: Generate the template
                ## and fill in batch size dependencies for final lstm layer
                if part == 'lstm_lm1b_2' or part == 'lstm_langmod_2':
                    templateFile = './templates/%s_TEMPLATE.py' % part
                    filledFile = './modules/examples/%s.py' % part  # TODO non-parallelizable
                    subsMap = {'BATCHSIZEX4':  str(GEN_STRING_BATCH_SIZE * 4),
                               'BATCHSIZEX32': str(GEN_STRING_BATCH_SIZE * 32)}
                    fillTemplate(templateFile, filledFile, subsMap, None)
                ##
                ##

                scheduleFile = schedulesDir + '/' + part
                if not os.path.exists(scheduleFile):
                    cmd = buildScheduleGenString(GEN_STRING_BATCH_SIZE, p['wordSize'],
                            p['nodeXY'], p['arrayXY'], p['localBufferSize'],
                            p['globalBufferSize'], part)
                    fullCmd =  cmd + ' > ' + scheduleFile
                    t = time.time()
                    os.system(fullCmd)
                    t = time.time() - t
                    print(p['name'] + "::" + part + " took %.2f s." % t)


def generateOps(configs, workloads):
    for c in configs:
        p = configs[c]

        for w in workloads:
            W = workloads[w]
            scheduleFile = './schedules/%s/%s/%s' % (p['name'], W['alias'], W['name'])
            print(scheduleFile)
            opsFile = './ops/%s_%s_ops.pl' % (p['name'], W['name'])
            numNetworkParts = len(W['parts'])   # for historical reasons
            os.system('./Extract_operations.pl %s %s %s' % (scheduleFile,
                    opsFile, numNetworkParts))


def generateTraces(configs, workloads):
    for c in configs:
        p = configs[c]      # params

        for w in workloads:
            W = workloads[w]

            tracesDir = './traces/%s/%s' % (p['name'], W['alias'])

            if not os.path.isdir(tracesDir):
                touchDir(tracesDir)
                for part in W['parts']:
                    scheduleFile = './schedules/%s/%s/%s' % (p['name'],
                            W['alias'], part)

                    cmd = buildTraceGenString(scheduleFile, tracesDir)
                    t = time.time()
                    os.system(cmd)
                    t = time.time() - t
                    print(p['name'] + "::" + part + " took %.2f s." % t)


# template-fill the zsim and tech config files
# NOTE: must make the distinction between baseline and n3xt, as their templates
# are different (for both zsim .cfg and tech config)
def generateSimConfigs(configs):
    for c in configs:
        p = configs[c]
        name = p['name']

        if 'baseline' in name:
            zsimTemplateFile = './templates/zsim-baseline.cfg'
            techTemplateFile = './templates/tech-baseline.pl'
        else:
            zsimTemplateFile = './templates/zsim-non-baseline.cfg'
            techTemplateFile = './templates/tech-non-baseline.pl'

        cfgFilesDir = './cfg/' + name
        if True:#not os.path.isdir(cfgFilesDir):
            touchDir(cfgFilesDir)

            zsimFile = cfgFilesDir + '/zsim.cfg'
            #techFile = cfgFilesDir + '/tech.pl'
            # NOTE must follow Perl script convention for now...
            techFile = PWD + '/config/tech/Config_%s_28nm_1_16.pl' % name

            # Compute some parameters needed to fill the templates.
            nNodes = p['nodeXY']**2
            nCompute = p['arrayXY']**2
            patchRoot = PWD + '/patchRoot/patchRoot_bc%d_bn%d' % (nNodes, nNodes)

            assert(p['globalBufferSize'] % nNodes == 0)
            gbufPerUnitSize = p['globalBufferSize'] / nNodes
            if 'baseline' in name:
                memBWPerUnit = None # won't be filled in
            else:
                memBWPerUnit = p['memBWPerChannel'] * p['memNChannels'] // nNodes
            compLkgPerNode = p['compLkgPerCU'] * nCompute   # not necessarily an int here!


            # Fill templates for zsim...
            # NOTE: for baseline templates which don't have latency, etc.
            # macros (since they use DDR), the template-fill will just ignore
            # unneded subs in subsMap.
            zsimSubsMap = {'nNodes':          str(nNodes),
                           'frequency':       str(p['frequency']),
                           'gbufPerUnitSize': str(int(gbufPerUnitSize)),
                           'globalBufferLat': str(p['globalBufferLat']),
                           # non-baseline-only subs. below
                           'memRDLat':        str(p['memRDLat']),
                           'memWRLat':        str(p['memWRLat']),
                           'memBWPerUnit':    str(memBWPerUnit),
                           'patchRoot':       patchRoot}
            # field(s) specific to non-baseline configs
            if 'baseline' not in name:
                zsimSubsMap['memNChannels'] = str(p['memNChannels'])

            fillTemplate(zsimTemplateFile, zsimFile, zsimSubsMap, None, '!')

            # ...and tech.
            techSubsMap = {'nNodes':    str(nNodes),
                           'frequency': str(float(p['frequency'])/1000),
                           'nCompute':  str(nCompute),
                           'compLkgPerNode':   str(compLkgPerNode),
                            # TODO beware scientific notation for too-small inputs!
                           'dynEPerOp': str(float(p['dynEPerOp'])/1000),
                           'regEPerBit':        str(p['regEPerBit']),
                           'localBufferEPerAccess':     str(p['localBufferEPerBit']*256/1000),
                           'globalBufferEPerAccess': str(p['globalBufferEPerBit']*256/1000),
                           'memRDEPerBit':      str(p['memRDEPerBit']),
                           'memWREPerBit':      str(p['memWREPerBit'])}
            # field(s) specific to non-baseline configs
            if 'baseline' not in name:
                techSubsMap['memLkg'] = str(int(p['memLkg']*1000))

            fillTemplate(techTemplateFile, techFile, techSubsMap, None, '!')
            os.system('chmod a+x %s' % techFile)   # make the .pl file executable


def createWorkingDirs(configs, workloads):
    for c in configs:
        p = configs[c]      # params

        for w in workloads:
            W = workloads[w]

            traceSrcSubDir = './traces/%s/%s/*' % (p['name'], W['alias'])
            workingSubDir = './working/%s/%s' % (p['name'], W['alias'])
            os.system('mkdir -p %s' % workingSubDir)   # create the subdir
            linkclone_dir(traceSrcSubDir, workingSubDir) # clone the trace directory

def generatePatchRoots(configs):
    for c in configs:
        p = configs[c]
        nNodes = p['nodeXY']**2
        patchRootDir = './patchRoot/patchRoot_bc%d_bn%d' % (nNodes, nNodes)
        if not os.path.exists(patchRootDir):
            os.system('./patchRoot/genPatchRoot.py -n %d --nodes %d'
                    ' -d patchRoot/patchRoot_bc%d_bn%d' % (nNodes, nNodes,
                    nNodes, nNodes))
            os.system('chmod -R 755 ./patchRoot/patchRoot_bc%d_bn%d' % (nNodes, nNodes))


# TODO prevent re-run if zsim.out exists?
def runZsimSimulations(configs, workloads, usePool=False):
    for c in configs:
        p = configs[c]      # params
        print('****Zsim for %s' % p['name'])
        for w in workloads:
            W = workloads[w]
            workingSubDir = PWD + '/working/%s/%s' % (p['name'], W['alias'])
            zsimCfgFullPath = PWD + '/cfg/%s/zsim.cfg' % p['name']
            zsimCmd = 'cd %s && %s/simulate_zsim_3.sh %s' %\
                    (workingSubDir, PWD, zsimCfgFullPath)
            if usePool:
                sshPool.submit(zsimCmd)
            else:
                os.system(zsimCmd)

    if usePool:
        sshPool.join()

def createPostprocStructure(configs, workloads):
    for i, c in enumerate(configs):
        p = configs[c]
        for w in workloads:
            W = workloads[w]
            workingSubDir = PWD + '/working/%s/%s' % (p['name'], W['alias'])
            postprocSubDir = PWD + '/%s/16/1/%s' % (W['name'], p['name'])
            touchDir(postprocSubDir)
            os.system('ln -sf %s/* %s' % (workingSubDir, postprocSubDir))


# call Parse_all_results.pl on all the simulated files
def runPostproc(configs, workloads):
    os.system('mkdir -p ./results') # done anyway
    for c in configs:
        p = configs[c]
        for w in workloads:
            W = workloads[w]
            cmdStr = './Parse_all_results.pl %s %s %s %s' % (W['name'], str(1),
                    p['baseline'], p['name'])
            print(cmdStr)
            os.system(cmdStr)

def spreadsheetResults(configs, workloads):
    for w in workloads:
        W = workloads[w]
        print('-'*10 + ' ' + W['alias'] + ' ' + '-'*10)
        for c in configs:
            resultsFileName = './results/%s_16_1_%s.csv' % (W['name'], c)
            with open(resultsFileName, 'r') as f:
                resLine = f.readlines()[-1]
                toks = resLine.split(',')
                # align and print the names and columns
                print(c + ',' + ' '*(30-len(c)) + ','.join(toks[1:7+1]))
        print('') # just a separating line


if __name__ == '__main__':
    # parse the configs and workloads
    configs = parseConfigsTXT('./configs.conf')
    workloads = parseConfigsTXT('./workloads.conf')

    SKIP_SCHED_TRACE = True

    if not SKIP_SCHED_TRACE:
        print('-'*30 + 'schedules' + '-'*30)
        generateSchedules(configs, workloads)

    print('-'*30 + 'ops' + '-'*30)
    generateOps(configs, workloads)

    if not SKIP_SCHED_TRACE:
        print('-'*30 + 'traces' + '-'*30)
        generateTraces(configs, workloads)

    print('-'*30 + 'patchroots' + '-'*30)
    generatePatchRoots(configs)

    print('-'*30 + 'sim-configs' + '-'*30)
    generateSimConfigs(configs)

    print('-'*30 + 'working-dirs' + '-'*30)
    createWorkingDirs(configs, workloads)

    DO_ZSIM=True
    if DO_ZSIM:
        print('-'*30 + 'zsim' + '-'*30)
        runZsimSimulations(configs, workloads, False)

    DO_POSTPROC=True
    if DO_POSTPROC:
        print('-'*30 + 'pre-postproc' + '-'*30)
        createPostprocStructure(configs, workloads)
        print('-'*30 + 'postproc' + '-'*30)
        runPostproc(configs, workloads)

    DO_DISPLAY=True
    if DO_DISPLAY:
        print('-'*30 + 'spreadsheet-results' + '-'*30)
        spreadsheetResults(configs, workloads)

    sys.exit(0)
