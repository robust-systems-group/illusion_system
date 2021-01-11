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
 * leti_board.h
 *
 *  Created on: May 15, 2018
 *      Author: tonyfwu
 */

#ifndef SRC_LETI_BOARD_H
#define SRC_LETI_BOARD_H

#include "leti_chip_board.h"
#include "multichip_interrupt_registers.h"
#include "xil_types.h"
#include "xil_io.h"

#define DRAM_BUFFER_SIZE 1024*1024

// AXI for output


#define CTRL_REG LETI_CHIP_BOARD_S00_AXI_SLV_REG0_OFFSET
#define ADDR_REG LETI_CHIP_BOARD_S00_AXI_SLV_REG1_OFFSET
#define DATA_REG LETI_CHIP_BOARD_S00_AXI_SLV_REG2_OFFSET
#define CLOCK_REG LETI_CHIP_BOARD_S00_AXI_SLV_REG3_OFFSET

#define LETI_CTRL_REG 0x43C00000

#define MULTI_CHIP_REG 0x43c10000
#define CHIP_SEL_REG MULTICHIP_INTERRUPT_REGISTERS_S00_AXI_SLV_REG1_OFFSET

#define RRAM_DONE_MASK 0x0002
#define RRAM_SUCCESS_MASK 0x0001

#define CTRL_START 0x00000001
#define CTRL_INSTR 0x00000000
#define CTRL_DATA  0x00000002
#define CTRL_WRITE 0x00000004
#define CTRL_MANUAL 0x00000000
#define CTRL_NORMAL 0x00000008
#define CTRL_SCAN  0x00000010
#define CTRL_SCAN_ENABLE 0x00000200
#define CTRL_CLK_ON 0x0000020
#define CTRL_CLK_OFF 0x0000000
#define CTRL_VERIFY_MODE_00 0x0000000
#define CTRL_VERIFY_MODE_01 0x0000040
#define CTRL_VERIFY_MODE_11 0x00000C0
#define CTRL_WAKEUP 0x0000100
#define CTRL_STEP 0x0000400
#define CTRL_RESET 0x0000800

#define WRITE_WORD (CTRL_START | CTRL_WRITE | CTRL_MANUAL | CTRL_CLK_ON)
#define READ_WORD (CTRL_START | CTRL_MANUAL | CTRL_CLK_ON)
#define SCAN (CTRL_START | CTRL_NORMAL | CTRL_SCAN | CTRL_CLK_ON | CTRL_SCAN_ENABLE)
#define STEP (CTRL_CLK_ON | CTRL_STEP)

void set_clock_div(int );
void set_clock(int);
void set_bldis(int );
void wakeup(int);
u64 scan(const unsigned short , const unsigned short , const unsigned short );
int write_word(const unsigned short , const unsigned short , const unsigned int);
u16 read_word(const unsigned short, const unsigned int );
void set_mode(int);
void scan_enable(int);
void step();
void reset_chip(int);

int check_result();

#endif /* SRC_LETI_BOARD_H */
