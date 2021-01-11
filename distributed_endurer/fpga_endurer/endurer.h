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

#ifndef ENDURER_H
#define ENDURER_H
//#include "xil_types.h"
// TODO @Robert uncomment top line, comment out next line
#include "mock_types.h"

/*
 * ENDUReR RRAM endurance mitigation mechanism, embedded implementation.
 *
 * This file defines the write_word() and read_word() methods, which are interfaces
 * into the RRAM+SRAM memory area.
 *
 * Symbols for segment size, remapping period, etc. are consistent with the paper.
 */

typedef u16 e_address;
typedef u16 e_data;
typedef u16 e_uint;
typedef u64 e_count;

typedef struct {
	u16 chip;
	u16 address;
	u16 data;
} e_trace;

typedef enum {
    NO_RESILIENCY,
    ENDURER,
    DISTRIBUTED_ENDURER
} e_endurer_mode;

#define M 2048
#define S 4
#define TW 4

#define NUM_CHIPS 8

void initialize(const e_uint _segment_size);
void flush_and_write_back(e_uint virt);
void remap();
void distributed_remap(e_uint, e_uint);
e_data endurer_read_word(const e_address address);
e_uint endurer_write_word(const e_address address, const e_data data);
e_uint teardown(e_uint virt);
e_endurer_mode get_endurer_mode();

/*
 * Getters and setters for the internal state of the ENDUReR mechanism.
 */
e_address get_relative_shift(e_uint virt);
e_count get_count(e_uint phys);
void enable_distributed_endurer();
void enable_endurer();
void disable_endurer();
void reset_endurer();
e_uint select_chip(e_uint virt);
e_uint time_to_shift_chips(e_uint *heavy, e_uint *light);
 
e_address get_virtual_chip_id(e_uint phys);
e_address get_physical_chip_id(e_uint virt);
e_address get_current_physical_chip();
e_address get_current_virtual_chip();

#endif
