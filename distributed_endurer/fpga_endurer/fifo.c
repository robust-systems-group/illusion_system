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

/*
 * fifo.c
 *
 *  Created on: Feb 10, 2019
 *      Author: tonyfwu
 */


#include "fifo.h"
#include <stdio.h>
#include <unistd.h>


int read_fifo_is_full() {
	u32 val = FIFO_CONTROLLER_mReadReg(FIFO_BASE_ADDR, READ_FIFO_STATUS);
	if (val & 0x00000002) {
		return 1;
	} else {
		return 0;
	}
}

int write_fifo_is_empty(){
	u32 val = FIFO_CONTROLLER_mReadReg(FIFO_BASE_ADDR, WRITE_FIFO_STATUS);
	if (val & 0x00000002) {
		return 1;
	} else {
		return 0;
	}
}

void read_fifo_write(u16 val) {
	FIFO_CONTROLLER_mWriteReg(FIFO_BASE_ADDR, READ_FIFO, (u32) val);
	u32 status = FIFO_CONTROLLER_mReadReg(FIFO_BASE_ADDR, READ_FIFO_STATUS);
	FIFO_CONTROLLER_mWriteReg(FIFO_BASE_ADDR, READ_FIFO_STATUS, (u32) (status | 0x00000001));
}

u16 write_fifo_read() {
	return FIFO_CONTROLLER_mReadReg(FIFO_BASE_ADDR, WRITE_FIFO);
}
void reset_fifos(u16 r_flag) {
	FIFO_CONTROLLER_mWriteReg(FIFO_BASE_ADDR, WRITE_FIFO_STATUS, r_flag << 2 );
	FIFO_CONTROLLER_mWriteReg(FIFO_BASE_ADDR, READ_FIFO_STATUS, r_flag << 2 );
}
//void reset_fifos() {
//	FIFO_CONTROLLER_mWriteReg(FIFO_BASE_ADDR, WRITE_FIFO_STATUS, 0x04);
//	//u32 status = FIFO_CONTROLLER_mReadReg(FIFO_BASE_ADDR, READ_FIFO_STATUS);
//	FIFO_CONTROLLER_mWriteReg(FIFO_BASE_ADDR, READ_FIFO_STATUS, 0x04);
//	//FIFO_CONTROLLER_mWriteReg(FIFO_BASE_ADDR, READ_FIFO_STATUS, status | 0x04);
//	usleep(100);
//	FIFO_CONTROLLER_mWriteReg(FIFO_BASE_ADDR, WRITE_FIFO_STATUS, 0x00);
//	//status = FIFO_CONTROLLER_mReadReg(FIFO_BASE_ADDR, READ_FIFO_STATUS);
//	FIFO_CONTROLLER_mWriteReg(FIFO_BASE_ADDR, READ_FIFO_STATUS, 0x00);
//	//FIFO_CONTROLLER_mWriteReg(FIFO_BASE_ADDR, READ_FIFO_STATUS, status & ~0x04);
//}

