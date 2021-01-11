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
 * leti_board.c
 *
 *  Created on: May 15, 2018
 *      Author: tonyfwu
 */


#include "leti_board.h"
#include <stdio.h>

u16 current_ctrl_reg = 0;

void set_clock_div(int divider){
	u32 val;
	val = LETI_CHIP_BOARD_mReadReg(LETI_CTRL_REG, CLOCK_REG);
	val = (val & 0xFFFF0000) | (divider & 0x0000FFFF);
	LETI_CHIP_BOARD_mWriteReg(LETI_CTRL_REG, CLOCK_REG, val);
}

void set_bldis(int time){
	u32 val;
	val = LETI_CHIP_BOARD_mReadReg(LETI_CTRL_REG, CLOCK_REG);
	val = (((u32)time) << 16) | (val & 0x0000FFFF);
	LETI_CHIP_BOARD_mWriteReg(LETI_CTRL_REG, CLOCK_REG, val);
}


void set_clock(int state){
	if (state) {
		current_ctrl_reg = current_ctrl_reg | CTRL_CLK_ON;
		LETI_CHIP_BOARD_mWriteReg(LETI_CTRL_REG, CTRL_REG, current_ctrl_reg);
	} else {
		current_ctrl_reg = current_ctrl_reg & (~CTRL_CLK_ON);
		LETI_CHIP_BOARD_mWriteReg(LETI_CTRL_REG, CTRL_REG, current_ctrl_reg);
	}
}

void wakeup(int state){
	if (state) {
		current_ctrl_reg = current_ctrl_reg | CTRL_WAKEUP;
		LETI_CHIP_BOARD_mWriteReg(LETI_CTRL_REG, CTRL_REG, current_ctrl_reg);
	} else {
		current_ctrl_reg = current_ctrl_reg & (~CTRL_WAKEUP);
		LETI_CHIP_BOARD_mWriteReg(LETI_CTRL_REG, CTRL_REG, current_ctrl_reg);
	}
}

int write_word(const unsigned short addr, const unsigned short val, const unsigned int segment){
//	printf("Writing %x to %x\n", val, addr);
	LETI_CHIP_BOARD_mWriteReg(LETI_CTRL_REG, ADDR_REG, addr);
	LETI_CHIP_BOARD_mWriteReg(LETI_CTRL_REG, DATA_REG, ((u32) val) << 16);
	LETI_CHIP_BOARD_mWriteReg(LETI_CTRL_REG, CTRL_REG, WRITE_WORD | segment);
	return check_result();
}

u64 scan(const unsigned short val0, const unsigned short val1, const unsigned short val2){
	LETI_CHIP_BOARD_mWriteReg(LETI_CTRL_REG, ADDR_REG, ((u32) val0) << 16 | (u32) val1);
	LETI_CHIP_BOARD_mWriteReg(LETI_CTRL_REG, DATA_REG, ((u32) val2) << 16);
	LETI_CHIP_BOARD_mWriteReg(LETI_CTRL_REG, CTRL_REG, SCAN);
	check_result();
	u64 result;
	u32 val;
//	printf("Scan Results\n");

	val = LETI_CHIP_BOARD_mReadReg(LETI_CTRL_REG, ADDR_REG);
//	printf("%0lx\n", val);
	result = ((u64) val) << 16;
	val = LETI_CHIP_BOARD_mReadReg(LETI_CTRL_REG, DATA_REG);
//	printf("%0lx\n", val);
	result |= val >> 16;
//	result |= val;

	return result;
}

u16 read_word(const unsigned short addr, const unsigned int segment){
	LETI_CHIP_BOARD_mWriteReg(LETI_CTRL_REG, ADDR_REG, addr);
	LETI_CHIP_BOARD_mWriteReg(LETI_CTRL_REG, CTRL_REG, READ_WORD | segment);
//	printf("Read from %04x\n", addr);
	check_result();

	int val = LETI_CHIP_BOARD_mReadReg(LETI_CTRL_REG, DATA_REG);
	val = LETI_CHIP_BOARD_mReadReg(LETI_CTRL_REG, DATA_REG);
	return (u16) (val & 0xFFFF);
}

void set_mode(int mode) {
	if (mode) { // normal mode
		current_ctrl_reg = current_ctrl_reg | CTRL_NORMAL;
		LETI_CHIP_BOARD_mWriteReg(LETI_CTRL_REG, CTRL_REG, current_ctrl_reg);
	} else {
		current_ctrl_reg = current_ctrl_reg & (~CTRL_NORMAL);
		LETI_CHIP_BOARD_mWriteReg(LETI_CTRL_REG, CTRL_REG, current_ctrl_reg);
	}
}

void reset_chip(int mode) {
	if (mode) { // Reset high
		current_ctrl_reg = current_ctrl_reg | CTRL_RESET;
		LETI_CHIP_BOARD_mWriteReg(LETI_CTRL_REG, CTRL_REG, current_ctrl_reg);
	} else { // Reset Low
		current_ctrl_reg = current_ctrl_reg & (~CTRL_RESET);
		LETI_CHIP_BOARD_mWriteReg(LETI_CTRL_REG, CTRL_REG, current_ctrl_reg);
	}
}

void step() {
	LETI_CHIP_BOARD_mWriteReg(LETI_CTRL_REG, CTRL_REG, current_ctrl_reg | STEP);
}

void scan_enable(int mode) {
	if (mode) { // enable
		current_ctrl_reg = current_ctrl_reg | CTRL_SCAN_ENABLE;
		LETI_CHIP_BOARD_mWriteReg(LETI_CTRL_REG, CTRL_REG, current_ctrl_reg);
	} else {
		current_ctrl_reg = current_ctrl_reg & (~CTRL_SCAN_ENABLE);
		LETI_CHIP_BOARD_mWriteReg(LETI_CTRL_REG, CTRL_REG, current_ctrl_reg);
	}
}

// Blocking function to check if control register is done
int check_result(){
	int r;
	r = LETI_CHIP_BOARD_mReadReg(LETI_CTRL_REG, CTRL_REG);
	while (!(r & RRAM_DONE_MASK)) {
		r = LETI_CHIP_BOARD_mReadReg(LETI_CTRL_REG, CTRL_REG);
	}
	return r & RRAM_SUCCESS_MASK;

}
