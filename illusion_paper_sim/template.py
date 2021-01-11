#!/usr/bin/env python

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

# Simple template/macro library.
# Anything between @ signs, e.g. @KEYWORD@, gets replaced with simple
# substitution.
#
# Anything between double @@ signs, e.g. @@KEYWORD@@, gets expanded with the
# text generator callback function supplied for KEYWORD.

DELIM_CHAR = '@'


def fillTemplate(srcFileName, destFileName, subsMap, callbackMap=None, delimChar=DELIM_CHAR):
    destChars = []

    with open(srcFileName, 'r') as src:
        data = src.read()
        i = 0
        while (i < len(data)):
            char = data[i]
            if char == delimChar:
                if data[i+1] == delimChar:     # double delim; do callback expansion
                    if callbackMap == None:
                        print("ERROR: requested callback expansion, but no callbacks supplied.")
                        return

                    endDelimIdx = data.index(delimChar, i+2)
                    callbackName = data[i+2:endDelimIdx]
                    destChars.append(callbackMap[callbackName]())

                    i += 2 + len(callbackName) + 2

                else:                          # single delim; do simple substitution
                    if subsMap == None:
                        print("ERROR: requested variable expansion, but no variables supplied.")
                        return

                    endDelimIdx = data.index(delimChar, i+1)
                    varName = data[i+1:endDelimIdx]
                    print(subsMap)
                    destChars.append(subsMap[varName])

                    i += 1 + len(varName) + 1

            else:       # just a regular character
                destChars.append(char)

                i += 1

    with open(destFileName, 'w') as dest:
        dest.write(''.join(destChars))
