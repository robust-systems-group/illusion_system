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


import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import sys
import random
from collections import OrderedDict
from measurement_params import *
from util import *
#import seaborn as sns
#sns.set()
random.seed(238)
#Parse inputs
wl = sys.argv[1]
normalize = int(sys.argv[2])
if len(sys.argv)>=4:
    num_samples = int(sys.argv[3])
else: num_samples = 64
if len(sys.argv)>=5:
    plot = int(sys.argv[4])
else: plot = 0

#LSTM was sampled at a higher rate
if wl =='lstm':
    MEAS_RATE=1000
else:
    MEAS_RATE=500

MEAS_PER=1.0/MEAS_RATE

#Weightings for the various "paths"
if wl =='d2nn':
    configs = ['ilsm_high','ilsm_low','il_high','il_low']
    path = ['h','l','h','l']
    wl_nums = [4,4,8,8]
    W_HIGH=0.869
    W_LOW=(1 - W_HIGH)
elif wl=='lstm':
    configs = ['ilsm_first','ilsm_final','il_first','il_final']
    path = ['','f','','f']
    wl_nums = [4,4,8,8]
    W_HIGH=39
    W_LOW=(40 - W_HIGH)
else:
    configs = ['ilsm_a','ilsm_b','il_a','il_b'] #Just easier to BS and weight on with 0
    path = ['','','','']
    wl_nums = [4,4,8,8]
    W_HIGH=1
    W_LOW=0

#Normalize chip-to-chip variations
if normalize:
    powers_norm = get_powers(datas_norm)
    c_norm = get_normalization_factors(powers_norm)
else: c_norm = None

results = OrderedDict()
results['target_d'] = np.zeros(num_samples)
results['ilsm_d']   = np.zeros(num_samples)
results['il_d']     = np.zeros(num_samples)
results['target_e'] = np.zeros(num_samples)
results['ilsm_e']   = np.zeros(num_samples)
results['ilsm_m']   = np.zeros(num_samples)
results['ilsm_l']   = np.zeros(num_samples)
results['il_e']     = np.zeros(num_samples)
results['il_m']     = np.zeros(num_samples)
results['il_l']     = np.zeros(num_samples)

chips = []
datas = []
raw_results = []
for i in range(num_samples):
    chips.append({})
    datas.append({})
    raw_results.append({})
    t_chip = [i % 8]
    all_chip = [x for x in range(8)]
    for j in range(len(configs)):
        config = configs[j]
        sampled_chips = random.sample(all_chip,wl_nums[j])
        random.shuffle(sampled_chips)
        chips[i][config] = np.array(t_chip + sampled_chips)
        datas[i][config] = get_datas(config.split('_')[0], chips[i][config], wl, path[j])
        raw_results[i][config] = get_illusion(datas[i][config], chips[i][config], meas_per=MEAS_PER, c_norm=c_norm)
    #Process data into something more manageable. [config][baseline vs. test][parameter]
    results['target_d'][i] =  W_HIGH*raw_results[i][configs[0]][0][0] + W_LOW*raw_results[i][configs[1]][0][0]
    results['target_e'][i] =  W_HIGH*raw_results[i][configs[0]][0][1] + W_LOW*raw_results[i][configs[1]][0][1]
    
    results['ilsm_d'][i] =    W_HIGH*raw_results[i][configs[0]][1][0] + W_LOW*raw_results[i][configs[1]][1][0]
    results['ilsm_e'][i] =    W_HIGH*raw_results[i][configs[0]][1][1] + W_LOW*raw_results[i][configs[1]][1][1]
    results['ilsm_l'][i] =    W_HIGH*raw_results[i][configs[0]][1][2] + W_LOW*raw_results[i][configs[1]][1][2]
    results['ilsm_m'][i] =    W_HIGH*raw_results[i][configs[0]][1][1] + W_LOW*raw_results[i][configs[1]][1][1] -\
                              W_HIGH*raw_results[i][configs[0]][1][2] - W_LOW*raw_results[i][configs[1]][1][2] -\
                              W_HIGH*raw_results[i][configs[0]][0][1] - W_LOW*raw_results[i][configs[1]][0][1]
    
    results['il_d'][i] =      W_HIGH*raw_results[i][configs[2]][1][0] + W_LOW*raw_results[i][configs[3]][1][0]
    results['il_e'][i] =      W_HIGH*raw_results[i][configs[2]][1][1] + W_LOW*raw_results[i][configs[3]][1][1]
    results['il_l'][i] =      W_HIGH*raw_results[i][configs[2]][1][2] + W_LOW*raw_results[i][configs[3]][1][2]
    results['il_m'][i] =      W_HIGH*raw_results[i][configs[2]][1][1] + W_LOW*raw_results[i][configs[3]][1][1] -\
                              W_HIGH*raw_results[i][configs[2]][1][2] - W_LOW*raw_results[i][configs[3]][1][2] -\
                              W_HIGH*raw_results[i][configs[2]][0][1] - W_LOW*raw_results[i][configs[3]][0][1]

print("Raw %s Performance: " % wl)
for key in results.keys():
    if '_d' not in key:
        s = stats.bayes_mvs(results[key]*1000) #Units mJ
    else:
        s = stats.bayes_mvs(results[key]) #Units mJ
    print("%10s, %07.4f, %07.4f, %07.4f" % (key ,s[0][0], s[0][1][0], s[0][1][1]))
print("%s Illusion Percentages: " % wl)
for key in results.keys():
    if 'target' in key:
        continue
    elif '_d' in key:
        s = stats.bayes_mvs(results[key]/results['target_d'])
        print("%10s Exec. Time, %07.6f, %07.6f, %07.6f" % (key ,s[0][0], s[0][1][0], s[0][1][1]))
    else:
        s = stats.bayes_mvs(results[key]/results['target_e'])
        print("%10s     Energy, %08.6f, %08.6f, %08.6f" % (key ,s[0][0], s[0][1][0], s[0][1][1]))

plot_headding = [' Workload', ' Parallelized', ' Pipelined', ' PipeParallel']
if plot >0:
    cnum = 0
    fig, axs = plt.subplots(13,figsize = (6,9),sharex=True)
    if wl=='lstm':
        fig.suptitle("LSTM"+plot_headding[plot-1],fontsize=20,fontweight='bold')
        chips_a = np.array([cnum]+[(x+cnum)%8 for x in range(4)]+[(x+cnum)%8 for x in range(8)])
        datas_a = get_datas('all',chips_a,'lstm','')
        powers_a = get_powers(datas_a,chips_a)
        if c_norm is not None: powers_a = normalize_powers(powers_a, c_norm, chips_a)
        runtimes_a,filters_a = get_runtimes(powers_a)
        
        chips_b = np.array([cnum]+[(x+cnum)%8 for x in range(4)]+[(x+cnum)%8 for x in range(8)])
        datas_b = get_datas('all',chips_b,'lstm','f')
        powers_b = get_powers(datas_b,chips_b)
        if c_norm is not None: powers_b = normalize_powers(powers_b, c_norm, chips_b)
        runtimes_b,filters_b = get_runtimes(powers_b)
        plt_start = int(1*MEAS_RATE)
        plt_end = int(1.75*MEAS_RATE)
        label_str = ['',' Final Inp.']
        parallel = [
                [[1,1,1,1],[1,1,1,1,1,1,1,1]],
                [[0,0,0,0],[0,0,0,0,0,0,0,0]]]
    if wl=='svhn':
        fig.suptitle("SVHN"+plot_headding[plot-1],fontsize=20,fontweight='bold')
        chips_a = np.array([cnum]+[(x+cnum)%8 for x in range(4)]+[(x+cnum)%8 for x in range(8)])
        datas_a = get_datas('all',chips_a,'svhn','')
        powers_a = get_powers(datas_a,chips_a)
        if c_norm is not None: powers_a = normalize_powers(powers_a, c_norm, chips_a)
        runtimes_a,filters_a = get_runtimes(powers_a)
        plt_start = int(1*MEAS_RATE)
        plt_end = int(35*MEAS_RATE)
        label_str = ['','']
        parallel = [
                [[0,1,0,0],[0,0,1,1,1,1,0,0]],
                [[0,0,0,0],[0,0,0,0,0,0,0,0]]]
    if wl=='mnist':
        fig.suptitle("MNIST"+plot_headding[plot-1],fontsize=20,fontweight='bold')
        chips_a = np.array([cnum]+[(x+cnum)%8 for x in range(4)]+[(x+cnum)%8 for x in range(8)])
        datas_a = get_datas('all',chips_a,'mnist','')
        powers_a = get_powers(datas_a,chips_a)
        if c_norm is not None: powers_a = normalize_powers(powers_a, c_norm, chips_a)
        runtimes_a,filters_a = get_runtimes(powers_a)
        plt_start = int(1*MEAS_RATE)
        plt_end = int(6*MEAS_RATE)
        label_str = ['','']
        parallel = [
                [[0,1,0,0],[0,0,1,1,1,0,1,0]],
                [[0,0,0,0],[0,0,0,0,0,0,0,0]]]
    if wl=='d2nn':
        fig.suptitle("D2NN"+plot_headding[plot-1],fontsize=20,fontweight='bold')
        chips_a = np.array([cnum]+[(x+cnum)%8 for x in range(4)]+[(x+cnum)%8 for x in range(8)])
        datas_a = get_datas('all',chips_a,'d2nn','h')
        powers_a = get_powers(datas_a,chips_a)
        if c_norm is not None: powers_a = normalize_powers(powers_a, c_norm, chips_a)
        runtimes_a,filters_a = get_runtimes(powers_a)
        
        chips_b = np.array([cnum]+[(x+cnum)%8 for x in range(4)]+[(x+cnum)%8 for x in range(8)])
        datas_b = get_datas('all',chips_b,'d2nn','l')
        powers_b = get_powers(datas_b,chips_b)
        if c_norm is not None: powers_b = normalize_powers(powers_b, c_norm, chips_b)
        runtimes_b,filters_b = get_runtimes(powers_b)
        plt_start = int(1*MEAS_RATE)
        plt_end = int(6*MEAS_RATE)
        label_str = [' High',' Low']
    plt_stride = plt_end - plt_start
    if plot==4:
        in_lb = ["\nInput "+str(i) for i in range(1,0,-1)] + ["\nInput "+str(4-sum(parallel[0][0]) - sum([1-parallel[0][0][j] for j in range(i)])) for i in range(4)] + ["\nInput "+str(8-sum(parallel[0][1])  - sum([1-parallel[0][1][j] for j in range(i)])) for i in range(8)] 
        print(in_lb)
    elif plot == 3:
        if wl == 'lstm':
            in_lb = [" Input "+str(i) for i in range(1,0,-1)] + [" Input "+str(i) for i in range(4,0,-1)] + [" Input "+str(i) for i in range(8,0,-1)] 
            in_lb_b = [''  for i in range(13)]
        else:
            in_lb = ["\nInput "+str(i) for i in range(1,0,-1)] + ["\nInput "+str(i) for i in range(4,0,-1)] + ["\nInput "+str(i) for i in range(8,0,-1)] 
    else:
        in_lb = ['' for i in range(13)]
    for i in range(datas_a.shape[0]):
        x = [x*MEAS_PER for x in range(plt_stride)]
        if 0<i<=4:
            if i==4:
                axs[12-i].set_title("4 Chip Illusion System",fontsize=16,fontweight='bold')
            else:
                axs[12-i].set_title("")
            if plot==2:
                rl = int(sum([runtimes_a[j] if parallel[0][0][j-1]!=1 else 0 for j in range(1,i)]))
                ya = np.roll(powers_a[i,0]*1000,rl)[plt_start:plt_end]
                if wl == 'lstm' or wl == 'd2nn': 
                    rl = int(sum([runtimes_b[j] if parallel[1][0][j-1]!=1 else 0 for j in range(1,i)]))
                    yb = np.roll(powers_b[i,0]*1000,rl)[plt_start:plt_end]
            elif plot>=3:
                ya = (powers_a[i,0]*1000)[plt_start:plt_end]
                if wl == 'lstm' or wl == 'd2nn': 
                    yb = (powers_b[i,0]*1000)[plt_start:plt_end]
            else: 
                ya = np.roll(powers_a[i,0]*1000,int(sum(runtimes_a[5:i])))[plt_start:plt_end]
                if wl == 'lstm' or wl == 'd2nn': yb = np.roll(powers_b[i,0]*1000,int(sum(runtimes_b[1:i])))[plt_start:plt_end]
            axs[12-i].plot(x, ya, label="Chip "+ str(i)+ label_str[0]+in_lb[i],color='red')
            if wl == 'lstm' or wl == 'd2nn': axs[12-i].plot(x, yb,label="Chip "+ str(i) + label_str[1]+in_lb_b[i],color='magenta')

            axs[12-i].set_ylim([0,.2])
            axs[12-i].grid(True, axis='x')
            axs[12-i].legend(bbox_to_anchor=(1,0.25),loc="center left",labelspacing=0.1,fontsize=14)
        elif 4<i<=13:
            if i==12:
                axs[12-i].set_title("8 Chip Illusion System",fontsize=16,fontweight='bold')
            else:
                axs[12-i].set_title("")
            
            if plot==2:
                rl = int(sum([runtimes_a[j] if parallel[0][1][j-5]!=1 else 0 for j in range(5,i)]))
                ya = np.roll(powers_a[i,0]*1000,rl)[plt_start:plt_end]
                if wl == 'lstm' or wl == 'd2nn': 
                    rl = int(sum([runtimes_b[j] if parallel[1][1][j-5]!=1 else 0 for j in range(5,i)]))
                    yb = np.roll(powers_b[i,0]*1000,rl)[plt_start:plt_end]
            elif plot>=3:
                ya = (powers_a[i,0]*1000)[plt_start:plt_end]
                if wl == 'lstm' or wl == 'd2nn': 
                    yb = (powers_b[i,0]*1000)[plt_start:plt_end]
            else: 
                ya = np.roll(powers_a[i,0]*1000,int(sum(runtimes_a[5:i])))[plt_start:plt_end]
                if wl == 'lstm' or wl == 'd2nn': yb = np.roll(powers_b[i,0]*1000,int(sum(runtimes_b[5:i])))[plt_start:plt_end]
            
            axs[12-i].plot(x, ya,label="Chip "+ str(i-4)+ label_str[0]+in_lb[i],color='blue')
            if wl == 'lstm' or wl == 'd2nn': axs[12-i].plot([x*MEAS_PER for x in range(plt_stride)],np.roll(powers_b[i,0]*1000,int(sum(runtimes_b[5:i])))[plt_start:plt_end], label="Chip "+ str(i-4) + label_str[1]+in_lb_b[i],color='cyan')
            
            axs[12-i].set_ylim([0,.2])
            axs[12-i].grid(True, axis='x')
            axs[12-i].legend(bbox_to_anchor=(1,0.25),loc="center left",labelspacing=0.1,fontsize=14)
        elif i==0:
            axs[12-i].set_title("Ideal Target System",fontsize=16,fontweight='bold')
            axs[12-i].plot([x*MEAS_PER for x in range(plt_stride)],powers_a[i,0][plt_start:plt_end]*1000,label="Chip " + str(i+1)+ label_str[0]+in_lb[i],color='green')
            try:
                axs[12-i].plot([x*MEAS_PER for x in range(plt_stride)],powers_b[i,0][plt_start:plt_end]*1000,label="Chip " + str(i+1) + label_str[1]+in_lb_b[i],color='black')
            except: pass
            axs[12-i].set_ylim([0,.2])
            axs[12-i].grid(True, axis='x')
            axs[12-i].legend(bbox_to_anchor=(1,0.25),loc="center left",labelspacing=0.1,fontsize=14)
            axs[12-i].set_xlabel("Time (s)",fontsize=16,fontweight='bold')
    plt.autoscale(enable=True, axis='x', tight=True)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9,hspace=0.95)
    #plt.subplot_tool()
    plt.show()


