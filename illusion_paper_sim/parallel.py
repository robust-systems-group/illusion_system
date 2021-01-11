#!/usr/bin/env python3

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

####
## Primary parallel driver. Round-robins tasks to available workers in the
## cluster.
####

from multiprocessing import Pool
import sys, os
import random



class SSHPool:

    def __init__(self, slotsPerHost=4):
        self.hosts = sorted(['rsg%d' % x for x in range(16, 16+1)])
        self.slotsPerHost = slotsPerHost # uniform for now
        self.hostPools = {h: Pool(self.slotsPerHost) for h in self.hosts}
        self.totalSlots = len(self.hosts) * self.slotsPerHost
        self.jobsSubmitted = 0



    # Submits asynchronously via SSH to a node (round-robin).
    def submit(self, cmd):
        host = self.hosts[self.jobsSubmitted % len(self.hosts)]
        pool = self.hostPools[host]

        sshCmd = 'ssh -t %s -- "bash -c \'%s\'"' % (host, cmd)
        pool.apply_async(os.system, [sshCmd])
        self.jobsSubmitted += 1




    def join(self):
        for p in self.hostPools.values():
            p.close()
        # could probably do this in one loop
        for p in self.hostPools.values():
            p.join()








if __name__ == '__main__':
    sshPool = SSHPool()


    for i in range(100):
        sshPool.submit('uptime && sleep 8')

    sshPool.join()
