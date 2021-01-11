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
 * fifo.h
 *
 *  Created on: Feb 10, 2019
 *      Author: tonyfwu
 */

#ifndef SRC_FIFO_H_
#define SRC_FIFO_H_

#include "FIFO_Controller.h"
#include "xil_types.h"
#include "xil_io.h"

#define FIFO_BASE_ADDR 0x43c20000
#define WRITE_FIFO FIFO_CONTROLLER_S00_AXI_SLV_REG0_OFFSET
#define WRITE_FIFO_STATUS FIFO_CONTROLLER_S00_AXI_SLV_REG1_OFFSET
#define READ_FIFO FIFO_CONTROLLER_S00_AXI_SLV_REG2_OFFSET
#define READ_FIFO_STATUS FIFO_CONTROLLER_S00_AXI_SLV_REG3_OFFSET

int read_fifo_is_full();
int write_fifo_is_empty();

void reset_fifos(u16 );


void read_fifo_write(u16 );
u16 write_fifo_read();



#endif /* SRC_FIFO_H_ */
