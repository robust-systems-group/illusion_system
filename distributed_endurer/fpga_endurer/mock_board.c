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
 * mock_board.c
 *
 *  Created on: March 11, 2020
 *      Author: zainabk
 */


#include "mock_board.h"
#include <stdio.h>

static u16 DATA_RRAM[2][M];
static u16 INST_RRAM[2][M*3];
short rram_phys_chip_id;

int write_word(const unsigned short addr, const unsigned short val, const unsigned int segment){
//	printf("Writing %x to %x\n", val, addr);
    //RRAM[rram_phys_chip_id][addr] = val;
    if (segment==CTRL_INSTR) INST_RRAM[rram_phys_chip_id][addr] = val;
    else DATA_RRAM[rram_phys_chip_id][addr] = val;
    
    return 1;
}

u16 read_word(const unsigned short addr, const unsigned int segment){
//	printf("Read from %04x\n", addr);
	//return (u16) RRAM[rram_phys_chip_id][addr];
	if (segment==CTRL_INSTR) return (u16) INST_RRAM[rram_phys_chip_id][addr];
    else return DATA_RRAM[rram_phys_chip_id][addr];
}

/* Set physical rram chip */
void select_rram_chip(int new_phys_chip) {
    rram_phys_chip_id = new_phys_chip;
}

/* Get physical chip */
u16 get_rram_chip() {
    return (u16) rram_phys_chip_id;
}

