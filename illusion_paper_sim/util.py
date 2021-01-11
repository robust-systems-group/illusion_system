# utility/helper functions

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

import subprocess
import sys, os


def warn(msg):
    print('\033[1;33m' + str(msg) + '\033[0;0m')

def die(msg):
    print('\033[1;31m' + str(msg) + '\033[0;0m')
    sys.exit(1)

def touchDir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

# Synchronously runs a command on the local system, and returns a list of lines that that command sent to STDOUT.
def run_cmd(cmd, cwd=None):
    lines = []
    if sys.version_info >= (2,7,16):
        lines = str(subprocess.check_output(cmd, shell=True, cwd=cwd), 'utf-8').split('\n')
    else:
        lines = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, cwd=cwd).communicate()[0]
        lines = lines.split('\n')

    return list(filter(lambda l: l != '', lines))

# Takes in a directory, and makes a copy of it by copying the directory
# structure, but hardlinking the non-directory contents instead of copying them.
# TODO: use symlinks (cp -as?) instead? (might be faster?)
def linkclone_dir(srcDir, destDir):
    # -a: archive
    # -l: hardlink
    # -n: skip if destination file exists
    run_cmd('cp -aln %s %s' % (srcDir, destDir))
