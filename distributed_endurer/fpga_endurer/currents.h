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
 * currents.h
 *
 *  Created on: May 15, 2018
 *      Author: tonyfwu
 */

#ifndef SRC_CURRENTS_H_
#define SRC_CURRENTS_H_


#include "leti_chip_board.h"
#include "xil_types.h"
#include "xil_io.h"
//#include "xspips.h"
#include "xparameters.h"


#define SPI_BASEADDR 0x41200000
#define SPI_CS 0
#define SPI_MOSI 1
#define SPI_CLK 2

#define NUM_DEVICES 8


// I2C device addresses. TODO: Double check this
#define SPI_IBIAS_ADDR (0x0003)
#define SPI_IRD_ADDR (0x0002)
#define SPI_IRSVER_ADDR (0x0001)
#define SPI_ISVER_ADDR (0x0000)

#define IIC_SCLK_RATE 100000

// Reference Current Addresses
#define ISVER_REF_REG_0 0xF0
#define ISVER_REF_REG_1 0xF1
#define ISVER_REF_REG_2 0xF2

#define IRD_REF_REG_0 0xF3
#define IRD_REF_REG_1 0xF4
#define IRD_REF_REG_2 0xF5

#define IRSVER_REF_REG_0 0xF0
#define IRSVER_REF_REG_1 0xF1
#define IRSVER_REF_REG_2 0xF2

#define IBIAS_REF_REG_0 0xF3
#define IBIAS_REF_REG_1 0xF4
#define IBIAS_REF_REG_2 0xF5

// Initializes SPI
void init_spi();

// Set Reference Currents via SPI
void set_currents(char, char , char , char , char );

#endif /* SRC_CURRENTS_H_ */
