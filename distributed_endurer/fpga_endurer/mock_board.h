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
 * mock_board.h
 *
 *  Created on: March 11, 2020
 *      Author: zainabk
 */

#ifndef SRC_LETI_BOARD_H
#define SRC_LETI_BOARD_H

#include "mock_types.h"

#define DRAM_BUFFER_SIZE 1024*1024

#define CTRL_START 0x00000001
#define CTRL_INSTR 0x00000000
#define CTRL_DATA  0x00000002

#define M 2048
#define NUM_CHIPS 8

int write_word(const unsigned short , const unsigned short , const unsigned int);
u16 read_word(const unsigned short, const unsigned int );
void select_rram_chip(int );
u16 get_rram_chip();

#endif /* SRC_LETI_BOARD_H */
