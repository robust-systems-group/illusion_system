#!/usr/bin/env python
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



from datetime import datetime
import sys
from labjack import ljm
import time

'''
Example for how to setup the labjack are read out the correct channel data

This is originally from ljm website
'''





MAX_REQUESTS = 1  # The number of eStreamRead calls to perform
# Open first found T7
handle = ljm.openS("T7", "ANY", "ANY")  # T7 device, Any connection, Any identifier

info = ljm.getHandleInfo(handle)
print("Opened a LabJack with Device type: %i, Connection type: %i,\n"
      "Serial number: %i, IP address: %s, Port: %i,\nMax bytes per MB: %i" %
      (info[0], info[1], info[2], ljm.numberToIP(info[3]), info[4], info[5]))

deviceType = info[0]



# Stream Configuration
aScanListNames = ["AIN10"] #Scan list names to stream
numAddresses = len(aScanListNames)
aScanList = ljm.namesToAddresses(numAddresses, aScanListNames)[0]
scanRate = 10000
scansPerRead = int(scanRate / 2)

try:
    # When streaming, negative channels and ranges can be configured for
    # individual analog inputs, but the stream has only one settling time and
    # resolution.

    # LabJack T7 and other devices configuration
    # Ensure triggered stream is disabled.
    ljm.eWriteName(handle, "STREAM_TRIGGER_INDEX", 0)
    # Enabling internally-clocked stream.
    ljm.eWriteName(handle, "STREAM_CLOCK_SOURCE", 0)
    ljm.eWriteName(handle,"STREAM_BUFFER_SIZE_BYTES", 32768)
    # All negative channels are single-ended, AIN0 and AIN1 ranges are
    # +/-10 V, stream settling is 0 (default) and stream resolution index
    # is 0 (default).
    aNames = ["AIN_ALL_NEGATIVE_CH", "AIN_ALL_RANGE",
              "STREAM_SETTLING_US", "STREAM_RESOLUTION_INDEX"]
    aValues = [1, 0.01, 10, 1]
    
    # Write the analog inputs' negative channels (when applicable), ranges,
    # stream settling time and stream resolution configuration.
    numFrames = len(aNames)
    ljm.eWriteNames(handle, numFrames, aNames, aValues)

    # Configure and start stream
    scanRate = ljm.eStreamStart(handle, scansPerRead, numAddresses, aScanList, scanRate)
    print("\nStream started with a scan rate of %0.0f Hz." % scanRate)
    
    print("\nPerforming %i stream reads." % MAX_REQUESTS)
    start = datetime.now()
    totScans = 0
    totSkip = 0  # Total skipped samples
    i = 1
    while i <= MAX_REQUESTS:
        ret = ljm.eStreamRead(handle)

        aData = ret[0]
        scans = len(aData) / numAddresses
        totScans += scans

        # Count the skipped samples which are indicated by -9999 values. Missed
        # samples occur after a device's stream buffer overflows and are
        # reported after auto-recover mode ends.
        curSkip = aData.count(-9999.0)
        totSkip += curSkip
        #print(aData)
        print("\neStreamRead %i" % i)
        ainStr = ""
        for j in range(0, numAddresses):
            ainStr += "%s = %0.5f, " % (aScanListNames[j], aData[j])
        print("  1st scan out of %i: %s" % (scans, ainStr))
        print("  Scans Skipped = %0.0f, Scan Backlogs: Device = %i, LJM = "
              "%i" % (curSkip/numAddresses, ret[1], ret[2]))
        i += 1

    end = datetime.now()

    print("\nTotal scans = %i" % (totScans))
    tt = (end - start).seconds + float((end - start).microseconds) / 1000000
    print("Time taken = %f seconds" % (tt))
    print("LJM Scan Rate = %f scans/second" % (scanRate))
    print("Timed Scan Rate = %f scans/second" % (totScans / tt))
    print("Timed Sample Rate = %f samples/second" % (totScans * numAddresses / tt))
    print("Skipped scans = %0.0f" % (totSkip / numAddresses))
except ljm.LJMError:
    ljme = sys.exc_info()[1]
    print(ljme)
except Exception:
    e = sys.exc_info()[1]
    print(e)

try:
    print("\nStop Stream")
    ljm.eStreamStop(handle)
except ljm.LJMError:
    ljme = sys.exc_info()[1]
    print(ljme)
except Exception:
    e = sys.exc_info()[1]
    print(e)

# Close handle
ljm.close(handle)
