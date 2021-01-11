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


import pyvisa
import time
import usb

'''
Old supplies for SPF Setup
supplies = {"VDD_LOW": [4, 0], "VDD_4V6":[4, 1], 
            "V_PG_SET": [3, 0], "V_PG_RESET": [3, 1],
            "V_WL_SET": [5, 0], "V_WL_RESET": [5, 1],
            "VDD": [6, 0], "VDD_1V2": [6, 1],
            "VDDSA": [1, 0], "VREAD": [1, 1],
            "5V": [23,0]}
'''
#Current setup in Gates 325
supplies = { "VDD_4V6":[1, 1], "VDD_LOW": [1, 2], "VREAD": [1, 3], "VDDSA": [1, 4], 
            "V_PG_RESET": [2, 1], "V_PG_SET": [2, 2], "V_WL_SET": [2, 3], "V_WL_RESET": [2, 4],
            "VDD_1V2": [3, 1], "VDD": [3, 2]}


def open():
    rm = pyvisa.ResourceManager("@py")
    instruments = [
            rm.open_resource('GPIB0::1::INSTR'),
            rm.open_resource('GPIB0::2::INSTR'),
            rm.open_resource('GPIB0::3::INSTR'),
            ]
    for instrument in instruments:
        instrument.timeout = 5000
        instrument.write_termination = ';'
        instrument.write('ID?')
        print("Instrument ID?"+instrument.read())
    return rm, instruments

rm, instruments = open()

def close():
    for instrument in instruments:
        instrument.close()

def reset_usb_hub():
    usb_dev = usb.core.find(idVendor=0x3923, idProduct =0x709B)
    usb_dev.reset()

def reset_all():
    reset_supply("VDD_1V2")
    reset_supply("VDD_4V6")
    reset_supply("V_PG_RESET")

def reset_supply(supply):
    ID = supplies[supply]
    instrument = instruments[ID[0]-1]    
    instrument.write('CLR')

def set_voltage(supply, voltage):
    ID = supplies[supply]
    instrument = instruments[ID[0]-1]    
    instrument.write( 'VSET '+str(ID[1])+','+str(voltage))
    instrument.write( 'OUT '+str(ID[1])+','+str(1))
    instrument.write( 'OUT? ' +str(ID[1]))
    print("Channel Enabled?"+instrument.read() + " Val: " + str(voltage))


def set_voltage_curr(supply, voltage, curr):
    ID = supplies[supply]
    instrument = instruments[ID[0]-1]    
    instrument.write( 'VSET '+str(ID[1])+','+str(voltage))
    instrument.write( 'ISET '+str(ID[1])+','+str(curr))
    instrument.write( 'OUT '+str(ID[1])+','+str(1))
    instrument.write( 'OUT? ' +str(ID[1]))
    print("Channel Enabled?"+instrument.read())

def disable_voltage(supply):
    ID = supplies[supply]
    instrument = instruments[ID[0]-1]    
    instrument.write( 'OUT '+str(ID[1])+','+str(0))
    instrument.write( 'OUT? ' +str(ID[1]))
    print("Channel Disabled?" + instrument.read())



# Use safe voltages. RRAM cells should not be able to be set or reset
def set_safe_voltages():
    set_voltage('VDD_1V2', 1.2)
    set_voltage('VDD_LOW', 1.2)
    set_voltage_curr('VDD', 3.3, 0.2) #3.3
    set_voltage('VDD_4V6', 3.6)
    set_voltage('V_PG_SET', 1.0)
    set_voltage('V_PG_RESET', 1.0)
    set_voltage('V_WL_SET', 1.6)
    set_voltage('V_WL_RESET', 3.0)
    set_voltage('VDDSA', 2.4)
    set_voltage('VREAD', 0.2)

# Use forming voltages. RRAM cells won't be able to be reset. The set voltage
# is much higher. Only use for forming
def set_form_voltages():
    set_voltage('VDD_1V2', 1.2)
    set_voltage('VDD_LOW', 1.2)
    set_voltage_curr('VDD', 3.3, 0.2) #3.3
    set_voltage('VDD_4V6', 3.6)
    set_voltage('V_PG_RESET', 1.0)
    set_voltage('V_WL_RESET', 3.0)
    set_voltage('VDDSA', 2.4)
    set_voltage('VREAD', 0.2)
    set_voltage('V_PG_SET', 3) #4
    set_voltage('V_WL_SET', 1.1) #0.9

# Use regular voltages. RRAM cells should be able to be set and reset reliably.
def set_normal_voltages():
    set_voltage('VDD_1V2', 1.2)
    #set_voltage('VDD', 3.3333) # 3.3
    set_voltage_curr('VDD', 3.3, 0.2) #3.3
    set_voltage('VDD_LOW', 1.2)
    set_voltage('VDD_4V6', 3.6)
    set_voltage('VDDSA', 2.4)
    set_voltage('VREAD', 0.2)
    set_voltage('V_PG_SET', 1.55) # 1.55From testing
    set_voltage('V_WL_SET', 1.6) # 1.6From testing
    #set_voltage('V_PG_SET', 2.00) # From Elisa
    #set_voltage('V_WL_SET', 1.3) # From Elisa
    set_voltage('V_PG_RESET', 2.5) #2.5
    set_voltage('V_WL_RESET', 3.2)

# 2-bit per cell
# Manually determined set voltages through trial & error
def set_2bit_voltages(distr): 
    set_voltage('VDD_4V6', 4.8)
    set_voltage('VREAD', 0.2)
    set_voltage('V_PG_SET', 2.00)
    if (distr == 0):
        set_voltage('V_WL_SET', 1.48) 
    elif (distr == 1):
        set_voltage('V_WL_SET', 1.1) 
    else : #2 or 3
        set_voltage('V_WL_SET', 0.9) 
    set_voltage('V_PG_RESET', 2.6)
    set_voltage('V_WL_RESET', 3.3)

