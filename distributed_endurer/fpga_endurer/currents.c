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
 * currents.c
 *
 *  Created on: May 15, 2018
 *      Author: tonyfwu
 */


#include "currents.h"

/*
 * main.c
 *
 *  Created on: May 7, 2018
 *      Author: tonyfwu
 */


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "leti_chip_board.h"
#include "xil_types.h"
#include "xil_io.h"
#include "currents.h"
//#include "xspips.h"
#include <sleep.h>
#include "xgpio.h"

union Buffer {
	u8 e1[16];
	u64 e2[2];
};

static u8 IBIAS_DAC_DATA[NUM_DEVICES * 10];
static u8 IRD_DAC_DATA[NUM_DEVICES * 10];
static u8 IRSVER_DAC_DATA[NUM_DEVICES * 10];
static u8 ISVER_DAC_DATA[NUM_DEVICES * 10];

static XGpio Gpio;


void reverse(union Buffer *in, union Buffer *out) {
	int i;
	for (i = 0; i < 16; i++){
		out->e1[i] = in->e1[15-i];
	}
}

void init_spi(){



	XGpio_Initialize(&Gpio, SPI_BASEADDR);
	XGpio_SetDataDirection(&Gpio, 1, 0);

	int i;
	for (i = 0; i < NUM_DEVICES*10; i++){
		IBIAS_DAC_DATA[i] = 0;
		IRD_DAC_DATA[i] = 0;
		IRSVER_DAC_DATA[i] = 0;
		ISVER_DAC_DATA[i] = 0;
	}
	for (i = 0; i < NUM_DEVICES; i++){
		IBIAS_DAC_DATA[(i+1)*10-1] = 1;
		IBIAS_DAC_DATA[(i+1)*10-2] = 1;

		IRD_DAC_DATA[(i+1)*10-1] = 1;
		IRD_DAC_DATA[(i+1)*10-2] = 0;

		IRSVER_DAC_DATA[(i+1)*10-1] = 0;
		IRSVER_DAC_DATA[(i+1)*10-2] = 1;

		ISVER_DAC_DATA[(i+1)*10-1] = 0;
		ISVER_DAC_DATA[(i+1)*10-2] = 0;

	}
//	Xil_Out32(SPI_BASEADDR, 1);


}

void unpack_bits(char a, u8 *out) {
	int i;
	u8 temp = (u8) a;
	for (i = 0; i < 8; i++) {
		out[i] = temp & 0x01;
		temp >>= 1;
	}
}

void send_spi(u8 *out, int len) {
	u32 data = 0;
	int i;
	// CS Low
	data = 0x00;
	XGpio_DiscreteWrite(&Gpio, 1, data);
//	printf("%x\n", data);

	usleep(1);

	for (i = len-1; i >= 0; i--){
		// Set data
		data ^= (-out[i] ^ data) & (1 << SPI_MOSI);
		data &= ~(1 << SPI_CLK);
		XGpio_DiscreteWrite(&Gpio, 1, data);
//		printf("%x\n", data);

		usleep(1);

		// Pulse clock
		data |= (1 << SPI_CLK);
		XGpio_DiscreteWrite(&Gpio, 1, data);
//		printf("%x\n", data);

		usleep(1);

	}
	usleep(1);
	XGpio_DiscreteWrite(&Gpio, 1, 1);
//	printf("%x\n", data);

}

void set_currents(char dev, char ibias, char ird, char irsver, char isver) {


	unpack_bits(ibias, IBIAS_DAC_DATA + dev*10);
	send_spi(IBIAS_DAC_DATA, NUM_DEVICES*10);

	usleep(100);

	unpack_bits(ird, IRD_DAC_DATA + dev*10);
	send_spi(IRD_DAC_DATA, NUM_DEVICES*10);

	usleep(100);

	unpack_bits(irsver, IRSVER_DAC_DATA + dev*10);
	send_spi(IRSVER_DAC_DATA, NUM_DEVICES*10);

	usleep(100);

	unpack_bits(isver, ISVER_DAC_DATA + dev*10);
	send_spi(ISVER_DAC_DATA, NUM_DEVICES*10);

}

