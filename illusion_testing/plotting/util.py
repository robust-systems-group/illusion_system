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


from measurement_params import *
import numpy as np
import matplotlib.pyplot as plt

datas_norm = []
datas_norm.append(np.load('final_curr_measurements/mnist/mnist_target_w0_c0_0k5.npz')['data'])
datas_norm.append(np.load('final_curr_measurements/mnist/mnist_target_w0_c1_0k5.npz')['data'])
datas_norm.append(np.load('final_curr_measurements/mnist/mnist_target_w0_c2_0k5.npz')['data'])
datas_norm.append(np.load('final_curr_measurements/mnist/mnist_target_w0_c3_0k5.npz')['data'])
datas_norm.append(np.load('final_curr_measurements/mnist/mnist_target_w0_c4_0k5.npz')['data'])
datas_norm.append(np.load('final_curr_measurements/mnist/mnist_target_w0_c5_0k5.npz')['data'])
datas_norm.append(np.load('final_curr_measurements/mnist/mnist_target_w0_c6_0k5.npz')['data'])
datas_norm.append(np.load('final_curr_measurements/mnist/mnist_target_w0_c7_0k5.npz')['data'])
datas_norm = np.array(datas_norm)

def get_powers(data_in, chip_ids=range(8), plot=False, volt=voltages):
    #Chip 6 leakage issue, biased off by 240uV
    datas = np.copy(data_in)
    for i in range(len(chip_ids)):
        if chip_ids[i]==6:
            datas[i][0] = datas[i][0]-0.00240    
    res_used = resistances[:,chip_ids]
    currents = datas / res_used.T[:, :, np.newaxis]
    powers = currents * volt[np.newaxis, :, np.newaxis]
    if plot: plot_powers(powers,chips)
    return powers

def get_runtimes(powers_in, on_thresholds=[0.00008,0.0003,0.00015]):
    filters = powers_in > np.array(on_thresholds)[np.newaxis,:,np.newaxis]
    runtimes = np.average(np.sum(filters, axis=2),axis=1)
    return runtimes,filters

def get_energies(powers_in, meas_per=1.0/500):
    runtimes,filters = get_runtimes(powers_in)
    on_data = np.zeros(powers_in.shape)
    on_data = np.where(filters, powers_in, on_data)
    energies = np.sum(on_data,axis=(2))*meas_per
    return energies 

def get_normalization_factors(powers_in, plot=False):
    energies = get_energies(powers_in)
    average_energy = np.average(energies,axis=0)
    c_norm = average_energy/energies
    return c_norm

def normalize_powers(powers_in, c_norm, chips=range(8), plot=False):
    powers = powers_in*c_norm[chips,:,np.newaxis]
    if plot: plot_powers(powers,chips)
    return powers

def get_leakage(powers_in, chips, meas_per=1.0/500):
    leakage_power = powers_in[1:,0,0:300]
    leakage = np.mean(leakage_power,axis=1)*meas_per
    return leakage

def get_datas(mode, chips, workload, path, loc='final_curr_measurements/', mtype='_0k5.npz'):
    datas = []
    datas.append(np.load(loc+'/'+workload+'/'+workload+'_target_w0'+path+'_c'+str(chips[0])+mtype)['data'])
    if mode =='ilsm':
        if workload !='d2nn':
            for i in range(4):
                chip = chips[i+1]
                datas.append(np.load(loc+'/'+workload+'/'+workload+'_ilsm_w'+str(i)+path+'_c'+str(chip)+mtype)['data'])
        else:
            if path == 'h':
                for i in range(4):
                    chip = chips[i+1]
                    datas.append(np.load(loc+'/'+workload+'/'+workload+'_ilsm_w'+str(i)+path+'_c'+str(chip)+mtype)['data'])
            elif path == 'l':
                wls = [0,1,-1,-1]
                for i in range(4):
                    chip = chips[i+1]
                    if wls[i] == -1:
                        temp_data = np.load(loc+'/'+workload+'/'+workload+'_ilsm_w'+str(0)+path+'_c'+str(chip)+mtype)['data']
                        temp_data[:,:] = temp_data[:,0][:,np.newaxis]
                        datas.append(temp_data)
                    else:
                        datas.append(np.load(loc+'/'+workload+'/'+workload+'_ilsm_w'+str(i)+path+'_c'+str(chip)+mtype)['data'])
    elif mode =='il':
        if workload !='d2nn':
            for i in range(8):
                chip = chips[i+1]
                datas.append(np.load(loc+'/'+workload+'/'+workload+'_il_w'+str(i)+path+'_c'+str(chip)+mtype)['data'])
        else:
            if path == 'h':
                wls = [0,1,-1,3,4,5,6,7]
            elif path == 'l':
                wls = [0,1,2,-1,-1,-1,-1,-1]
            for i in range(8): 
                chip = chips[i+1]
                if wls[i] == -1:
                    temp_data = np.load(loc+'/'+workload+'/'+workload+'_il_w'+str(0)+path+'_c'+str(chip)+mtype)['data']
                    temp_data[:,:] = temp_data[:,0][:,np.newaxis]
                    datas.append(temp_data)
                else:
                    datas.append(np.load(loc+'/'+workload+'/'+workload+'_il_w'+str(i)+path+'_c'+str(chip)+mtype)['data'])
    elif mode =='all':
        if workload !='d2nn':
            for i in range(4):
                chip = chips[i+1]
                datas.append(np.load(loc+'/'+workload+'/'+workload+'_ilsm_w'+str(i)+path+'_c'+str(chip)+mtype)['data'])
            for i in range(8):
                chip = chips[i+5]
                datas.append(np.load(loc+'/'+workload+'/'+workload+'_il_w'+str(i)+path+'_c'+str(chip)+mtype)['data'])
        else:
            typ =['_ilsm_w','_il_w','_il_w0']
            if path == 'h':
                wls = [0,1,2,3,0,1,-1,3,4,5,6,7]
            elif path == 'l':
                wls = [0,1,-1,-1,0,1,2,-1,-1,-1,-1,-1]
            for i in range(12): 
                if i < 4:
                    typ_wl = typ[0]+str(i)
                    chip = chips[i+1]
                else:
                    typ_wl = typ[1] + str(i-4)
                    chip = chips[i+1]
                if wls[i] == -1:
                    temp_data = np.load(loc+'/'+workload+'/'+workload+typ[2]+path+'_c'+str(chip)+mtype)['data']
                    temp_data[:,:] = temp_data[:,0][:,np.newaxis]
                    datas.append(temp_data)
                else:
                    datas.append(np.load(loc+'/'+workload+'/'+workload+typ_wl+path+'_c'+str(chip)+mtype)['data'])
    datas=np.array(datas)
    return datas


def get_illusion(datas, chips, meas_per=1.0/500, c_norm=None, plot=False):
    powers = get_powers(datas,chip_ids=chips,plot=plot)
    runtimes,filters = get_runtimes(powers)
    if c_norm is not None: powers = normalize_powers(powers, c_norm, chips)
    energies_c = get_energies(powers, meas_per=meas_per)
    leakage = get_leakage(powers, chips, meas_per=meas_per)
    runtime_illusion = np.sum(runtimes[1:])
    shutdown_times = runtime_illusion - runtimes[1:]
    leakage_energy = np.sum(leakage*shutdown_times)
    illusion_delay = np.sum(runtimes[1:])*meas_per
    target_delay = runtimes[0]*meas_per
    illusion_energy = np.sum(energies_c[1:]) + leakage_energy
    target_energy = np.sum(energies_c[0])
    return [[target_delay, target_energy],[illusion_delay, illusion_energy, leakage_energy]]

def plot_powers(powers, chips=range(8)):
    fig, axs = plt.subplots(3,figsize=(8,8))
    for i in range(len(chips)):
        for j in range(3):
            axs[j].plot(powers[i,j], label="chip " + str(chips[i]) + " ch " + str(voltages[j]))
            axs[j].legend(bbox_to_anchor=(1,0.5),loc='center left')
    plt.show()

