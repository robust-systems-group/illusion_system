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
 * This file defines a top driver for Endurer 
 */

#include "mock_board.h"
#include "endurer.h"

/* Select a chunk for R/W (NO RESILIENCY) */
int select_segment(e_uint rram_chip, e_uint segment_type, e_uint chunk);

/* Assign a 'physical' chip to an rram segment chunk (END/DE) */
int assign_physical_segment(e_uint phys_chip, e_uint rram_chip, e_uint segment_type, e_uint chunk);

/* Verify that the physical chip to rram segment map is possible for distributed endurer */ 
int verify_segments();

/* Read out which rram chip & segment & chunk the 'physical chip' of the system is on */ 
void read_out_physical_segments();
 
/* Endurer state function are actually defined in endurer.c but defined here */
/* reset_endurer_system() ->return if successful */
int reset_endurer_state();

/* only changes mode, does not reset anything */
void enable_distributed_endurer();
void enable_endurer();
void disable_endurer();

/* set_endurer_system(offsets,phsyical-to-virtual chip, mode, write counts, whatever else is stateful) -> should return if successful */
int set_endurer_state(e_uint mode, e_count swap_count, e_uint virt_to_phys_map[NUM_CHIPS], e_uint offsets[NUM_CHIPS],
                    e_count counts[NUM_CHIPS], double TAU_, double EPS_, double C_);
 
/* read_endurer_system() -> returns all the inputs of 2 above, just but also needs to read out the number of swaps performed (to check that we match with simulation) */
void get_endurer_state(e_uint *mode, e_count *swap_count, e_uint* virt_to_phys_map, 
                    e_uint* phys_to_virt_map, e_uint* offsets, e_count* counts,
                    double* TAU_, double* EPS_, double* C_);

/* Process the initial trace file over stream until EOF */
// Returns trace length, or -1 if trace overflows
int send_trace_pattern(int trace_addr);
 
/* Defined in endurer_top.c */
/* read_in_trace(trace (virtual chip ID, word, value), trace_length) -> up to you if you split writes by chip in the code or not, but we should minimize physical chip-to-chip switching as it is a slow process. */
// select which chip's traces to run
int run_trace(int virt_chip, int trace_offset, int trace_length);

/* read_out_dram() -> spits out the trace in DRAM to make sure we are doing the right thing */
void read_out_dram();

/* read_out_trace() -> spits out the trace in DRAM to make sure we are doing the right thing */
int read_out_trace(int trace_offset, int trace_length);

/* run_endurer(Number of times to run the trace) -> Should error if there isn't a trace present, should return what state was run */
int run_endurer(int trace_offset, int trace_length, int times_before_remap, int total_times);
 
/* If we must manually do each chip once & do all remaps for it before goign to next chip */
int run_endurer_single_chip(int chip, int trace_offset, int trace_length, int times_before_remap, int total_times);
 
/* read_out_memory -> Should read out the data based on the endurer settings */
// streams out memory chip by chip 
void read_out_memory();

// Reads memory without using offsets
// Uses the phys_to_virt to select the right physical chip!
// But tries to read phys chip 0, phys chip 1
void read_out_memory_raw();
 
/* When sending dmem.mem file, tell which virtual chip is being streamed; does till EOF */
int stream_initial_chip_memory(int c);

// Do distributed remap manually
// Input : physical chip ids
void distributed_remap(e_uint, e_uint);

