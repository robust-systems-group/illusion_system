// 
// Copyright (C) 2020 by The Board of Trustees of Stanford University
// This program is free software: you can redistribute it and/or modify it under
// the terms of the Modified BSD-3 License as published by the Open Source
// Initiative.
// If you use this program in your research, we request that you reference the
// Illusion paper, and that you send us a citation of your work.
// This program is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
// PARTICULAR PURPOSE. See the BSD-3 License for more details.
// You should have received a copy of the Modified BSD-3 License along with this
// program. If not, see <https://opensource.org/licenses/BSD-3-Clause>.
//  
//

/*
 * This file defines a driver for Endurer 
 */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "endurer_top.h"

extern u16 dram_buf [DRAM_BUFFER_SIZE];


/* read_in_trace(trace (virtual chip ID, word, value), trace_length) -> up to you if you split writes by chip in the code or not, but we should minimize physical chip-to-chip switching as it is a slow process. */
int run_trace(int virt_chip, int trace_offset, int trace_length) {
    // assumes select_chip already called on virt chip!!
    e_trace* trace = (e_trace*)(dram_buf + trace_offset);
    if (trace_offset>DRAM_BUFFER_SIZE) return 1;
    for (int i=0; i<trace_length && trace_offset+i*sizeof(e_trace)<DRAM_BUFFER_SIZE; i++) {
        if (trace[i].chip==virt_chip) endurer_write_word(trace[i].address, trace[i].data);
    }
    if (trace_offset+sizeof(e_trace)*trace_length > DRAM_BUFFER_SIZE) return 2;
    return 0;
}


/* read_out_trace() -> spits out the trace in DRAM to make sure we are doing the right thing */
int read_out_trace(int trace_offset, int trace_length) {
    e_trace* trace = (e_trace*) (dram_buf + trace_offset);
    if (trace_offset>DRAM_BUFFER_SIZE) return 1;
    for (int i=0; i<trace_length && trace_offset+i*sizeof(e_trace)<DRAM_BUFFER_SIZE; i++) {
        printf("%d,%x,%x\n", trace[i].chip, trace[i].address, trace[i].data);
    }
    if (trace_offset+sizeof(e_trace)*trace_length > DRAM_BUFFER_SIZE) return 2;
    return 0;
}

/* run_endurer(Number of times to run the trace) -> Should error if there isn't a trace present, should return what state was run */
int run_endurer(int trace_offset, int trace_length, int times_before_remap, int total_times) {
    e_uint heavy, light, error;
    for (int t=0; t<total_times; t++) {
        for (int c=0; c<NUM_CHIPS; c++) {
            select_chip(c);
            error = run_trace(c, trace_offset, trace_length);
            if (error) return error;
        
            if (t!=0 && (t%times_before_remap==0)) {
                remap();
            }
        }
        if (time_to_shift_chips(&heavy, &light)) {
            distributed_remap(heavy, light);
        }
    } 
    return 0;
}

/* If we must manually do each chip once & do all remaps for it before goign to next chip */
int run_endurer_single_chip(int chip, int trace_offset, int trace_length, int times_before_remap, int total_times) {
    int error;
    for (int t=0; t<total_times; t++) {
        error = run_trace(chip, trace_offset, trace_length);
        if (error) return error;
    
        if (t!=0 && (t%times_before_remap==0)) {
            remap();
        }
    } 
    return 0;
}


/* read_out_memory -> Should read out the data based on the endurer settings */
void read_out_memory() {
    // in order, virtual chip 0 onwards
    // virt addr 0 onwards
    for (int i=0; i<NUM_CHIPS; i++) {
        select_chip(i);
        for (int j1=0; j1<M/16; j1++) {
            for (int j2=0; j2<15; j2++) {
                printf("%4x,",endurer_read_word(j1*16+j2));
            } printf("%4x\n",endurer_read_word(j1*16+15));
        } 
    }
}

/* read_out_memory_raw -> Should read out the data as is */
void read_out_memory_raw() {
    // in order, virtual chip 0 onwards
    // virt addr 0 onwards
    for (int i=0; i<NUM_CHIPS; i++) {
        select_chip(get_virtual_chip_id(i));
        for (int j1=0; j1<M/16; j1++) {
            for (int j2=0; j2<15; j2++) {
                printf("%4x,",read_word_helper(j1*16+j2));
            } printf("%4x\n",read_word_helper(j1*16+15));
        } 
    }
}

// Returns trace length, or -1 if trace overflows
int send_trace_pattern(int trace_addr) {
    e_trace* trace = (e_trace*)(dram_buf+trace_addr);
    int a;
    int d;
    int c;
    int i = 0;
    while (scanf("%d,%x,%d\n",&c,&a,&d) != EOF) {
        if (trace_addr+sizeof(e_trace)*i>=DRAM_BUFFER_SIZE) return -1;
        trace[i].chip = c;
        trace[i].address = a;
        trace[i].data = d;
        i++;
    }
    return i;
}
 
int stream_initial_chip_memory(int c) {
    int addr, data, error, i;
    select_chip(c);
    while (scanf("\n") != EOF) {
        error = scanf("@%x",&addr);
        if (error==EOF) {
            return 1;
        }

        for (i = 0; i<16; i++) {
            error = scanf("%x", &data);
            if (error==EOF) {
                return 1;
            }
            endurer_write_word(addr + i, data);
        }
    }
    return 0;

}

