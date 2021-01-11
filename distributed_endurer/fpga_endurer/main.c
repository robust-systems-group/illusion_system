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
 * main.c
 *
 *  Created on: May 7, 2018
 *      Author: tonyfwu
 */


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "xil_types.h"
#include "xil_io.h"
#include "currents.h"
#include "leti_board.h"
#include "multichip_interrupt_registers.h"
#include "fifo.h"

int clock_divider = 4; //Current clock divider
int bldis = 0; // Current bldis
char buffer [1024]; // Buffer for serial reading

char tokens[128][1024];

// Shadow copy of the RRAM
u16 dram_buf [DRAM_BUFFER_SIZE];


unsigned char IBIAS = 229;
unsigned char IRD = 229;
unsigned char ISVER = 229;
unsigned char IRSVER = 229;

// List of commands
#define CMD_SET_CLOCK "*CLKDIV"
#define CMD_SET_BLDIS "*BLDIS"
#define CMD_CLOCK "*CLK"
#define CMD_LOAD "*LOAD"
#define CMD_PROG "*PROG"
#define CMD_SET_CURR "*CURR"
#define CMD_READ "*READ"
#define CMD_WRITE "*WRITE"
#define CMD_SCAN "*SCAN"
#define CMD_SCAN_EN "*SCANEN"
#define CMD_MODE "*MODE"
#define CMD_IDN "*IDN"
#define CMD_WAKEUP "*WAKEUP"
#define CMD_STEP "*STEP"
#define CMD_SEL "*SEL"
#define CMD_FIFO_WRITE "*FIFOWRITE"
#define CMD_FIFO_READ "*FIFOREAD"
#define CMD_FIFO_EMPTY "*FIFOEMPTY"
#define CMD_FIFO_FULL "*FIFOFULL"
#define CMD_FIFO_RESET "*FIFORESET"
#define CMD_CHIP_RESET "*RESET"
#define CMD_CHIP_CHSEGMENT "*CHSEGMENT"
#define CMD_ENDR_ASSIGNSEGMENT "*ENDURERASSIGNSEGMENT"
#define CMD_ENDR_VERIFYSEGMENTS "*ENDURERVERIFYSEGMENTS"
#define CMD_ENDR_READOUTSEGMENTS "*ENDURERREADOUTSEGMENTS"
#define CMD_ENDR_RESET "*ENDURERRESET"
#define CMD_ENDR_CHMOD "*ENDURERCHMOD"
#define CMD_ENDR_SETSTATE "*SETENDURERSTATE"
#define CMD_ENDR_GETSTATE "*GETENDURERSTATE"
#define CMD_ENDR_SENDTRACE "*SENDTRACE"
#define CMD_ENDR_RUNTRACE "*RUNTRACE"
#define CMD_ENDR_READOUTTRACE "*READOUTTRACE"
#define CMD_ENDR_RUNENDURER "*RUNENDURER"
#define CMD_ENDR_RUNENDURERCHIP "*RUNENDURERCHIP"
#define CMD_ENDR_READMEM "*READMEM"
#define CMD_ENDR_READMEMRAW "*READMEMRAW"
#define CMD_ENDR_SETMEM "*SETMEM"
#define CMD_ENDR_SWAP "*ENDURERSWAP"

int select_segment(e_uint rram_chip, e_uint segment_type, e_uint chunk);

/* Assign a 'physical' chip to an rram segment chunk (END/DE) */
int assign_physical_segment(e_uint phys_chip, e_uint rram_chip, e_uint segment_type, e_uint chunk);

/* Verify that the physical chip to rram segment map is possible for distributed endurer */ 
int verify_segments();

/* Read out which rram chip & segment & chunk the 'physical chip' of the system is on */ 
void read_out_physical_segments();
 
#define hex2i(val) (u16) strtol(val, NULL, 16)

void setup(){
	init_spi();
	for (int i = 0; i < 8; i++)
		set_currents(i, IBIAS, IRD, IRSVER, ISVER);
	set_clock(clock_divider);
}

char ** get_args(char * buf, size_t *num_tokens){
	char *temp, *token;
	int i;
	temp = buf;
	token = strtok(temp, " ");
	i = 0;
	while (token != NULL && strcmp(token[0], "")) {
		token = strtok(NULL, " ");
		strcpy(tokens[i], token);
		i++;
	}
	(*num_tokens) = i;
	return tokens;
}

char ** split(char * buf, size_t *num_tokens){
	char *temp, *token;
	int i;
	temp = buf;
	token = strtok(temp, " ");
	strcpy(tokens[0], token);
	i = 1;
	while (token != NULL && strcmp(token[0], "")) {
		token = strtok(NULL, " ");
		strcpy(tokens[i], token);
		i++;
	}
	(*num_tokens) = i;
	return tokens;
}

void select_rram_chip(int dev) {
    // Decode (because its not implemented in hardware.... oops)
    u16 val;
    if (dev >= 0 && dev < 8){
        val = 0x0001 << dev;
        val = ~val;
    } else {
        val = 0xFFFF;
    }

    MULTICHIP_INTERRUPT_REGISTERS_mWriteReg(MULTI_CHIP_REG,
            CHIP_SEL_REG,
            val);
}

int get_rram_chip() {
    u16 val,out;
    MULTICHIP_INTERRUPT_REGISTERS_mReadReg(MULTI_CHIP_REG,
            CHIP_SEL_REG,
            &val);
    // Decode
    val = ~val;
    switch(val) {
        case 0x0001 : out = 0; break;
        case 0x0002 : out = 1; break;
        case 0x0004 : out = 2; break;
        case 0x0008 : out = 3; break;
        case 0x0010 : out = 4; break;
        case 0x0020 : out = 5; break;
        case 0x0040 : out = 6; break;
        case 0x0080 : out = 7; break;
        default : out = -1; break;
    } 
    return out;
}

#define FORCE_NUM_TOKENS(x) ({\
                if (num_tokens<x) {\
                    printf("Not enough args (required %d)\n", x);\
                    continue;\
                }\
               })

// Main Function
int main() {
	printf("Setup\n");
	setup();

	int i;
	int currents[4];
	int dev;
	size_t num_tokens;

	while (1) {
		gets(buffer);
		if (strncmp(buffer, CMD_IDN , strlen(CMD_IDN)) == 0) {
			printf("LETI Test Board\n");
		}
		// Chip Select
		else if (strncmp(buffer, CMD_SEL , strlen(CMD_SEL)) == 0) {
            get_args(buffer, &num_tokens);
            int dev;
            dev = atoi(tokens[0]);
            select_rram_chip(dev);
            printf("Selected %d \n", dev);
        }
		// Set Clock
		else if (strncmp(buffer, CMD_SET_CLOCK , strlen(CMD_SET_CLOCK)) == 0) {
			get_args(buffer, &num_tokens);
			clock_divider = atoi(tokens[0]);
			set_clock_div(clock_divider);
			printf("Clock Divider set to %d \n", clock_divider);
		}
		// Reset chip
		else if (strncmp(buffer, CMD_CHIP_RESET , strlen(CMD_CHIP_RESET)) == 0) {
			get_args(buffer, &num_tokens);
			int val = atoi(tokens[0]);
			reset_chip(val);
			printf("RST set to %d \n", val);
		}
		// Set Bitline Discharge pulse width
		else if (strncmp(buffer, CMD_SET_BLDIS , strlen(CMD_SET_BLDIS)) == 0) {
			get_args(buffer, &num_tokens);
			bldis = atoi(tokens[0]);
			set_bldis(bldis);
			printf("BLDIS set to %d ns\n", 10*(bldis+1));
		}
		// Send 1 clock pulse for debugging
		else if (strncmp(buffer, CMD_STEP , strlen(CMD_STEP)) == 0) {
			step();
			printf("Step\n");
		}
		// Turn clock on/off
		else if (strncmp(buffer, CMD_CLOCK , strlen(CMD_CLOCK)) == 0) {
			get_args(buffer, &num_tokens);
			i = atoi(tokens[0]);
			set_clock(i);
			if (i) printf("Setting clock to on\n");
			else printf("Setting clock to off\n");
		}
		// Select DMA (manual) mode or normal mode
		else if (strncmp(buffer, CMD_MODE , strlen(CMD_MODE)) == 0) {
			get_args(buffer, &num_tokens);
			i = atoi(tokens[0]);
			set_mode(i);
			if (i) printf("Setting mode to normal\n");
			else printf("Setting mode to manual\n");
		}
		// Assert wakeup pin
		else if (strncmp(buffer, CMD_WAKEUP , strlen(CMD_WAKEUP)) == 0) {
			get_args(buffer, &num_tokens);
			i = atoi(tokens[0]);
			wakeup(i);
			if (i) printf("Wakeup High\n");
			else printf("Wakeup Low\n");
		}
		// Scan enable
		else if (strncmp(buffer, CMD_SCAN_EN , strlen(CMD_SCAN_EN)) == 0) {
			get_args(buffer, &num_tokens);
			i = atoi(tokens[0]);
			scan_enable(i);
			if (i) printf("Setting scan_en to 1\n");
			else printf("Setting scan_en to 0\n");
		}
		// Load DRAM
		else if (strncmp(buffer, CMD_LOAD , strlen(CMD_LOAD)) == 0) {
			u32 start_address, address;
			get_args(buffer, &num_tokens);
			start_address = strtol(tokens[0], NULL, 16);
			address = start_address;
			int j;
			gets(buffer);
			buffer[strlen(buffer)-1] = '\0'; // Remove \n
			while (buffer != NULL && strcmp(buffer, "") != 0) {
				split(buffer, &num_tokens);
				j = 0;
				while (j < num_tokens-1) {
					dram_buf[address] = hex2i(tokens[j]);
					address++;
					j++;
				}
				gets(buffer);
				buffer[strlen(buffer)-1] = '\0'; // Remove \n
			}
			printf("Done loading\n");
		}
		// Program Chip
		else if (strncmp(buffer, CMD_PROG , strlen(CMD_PROG)) == 0) {
			printf("Programming\n");
			u32 dram_start_address, dram_address;
			u16 rram_start_address, rram_address;
			int rram_segment;
			u16 num_words;

			get_args(buffer, &num_tokens);
			dram_start_address = strtol(tokens[0], NULL, 16);
			rram_segment = atoi(tokens[1]);
			rram_start_address = (u16) strtol(tokens[2], NULL, 16);
			num_words = atoi(tokens[3]);

			dram_address = dram_start_address;
			int r;
			for (rram_address = rram_start_address; rram_address < rram_start_address + num_words; rram_address++){
//				printf("Writing %x to %x\n", imem_buf[i], i);
				if (rram_segment == 0) {
					r = read_word(rram_address, CTRL_INSTR);
					r = write_word(rram_address, dram_buf[dram_address], CTRL_INSTR);
				}
				else {
					r = read_word(rram_address, CTRL_DATA);
					r = write_word(rram_address, dram_buf[dram_address], CTRL_DATA);
				}
				dram_address++;
			}
			printf("Done\n");
		}
		// Set board reference currents
		else if (strncmp(buffer, CMD_SET_CURR , strlen(CMD_SET_CURR)) == 0) {
			get_args(buffer, &num_tokens);
			dev = atoi(tokens[0]);
			for (i = 1; i < 5; i++) currents[i-1] = atoi(tokens[i]);
			printf("Setting currents on device %d to %d %d %d %d\n", dev, currents[0], currents[1], currents[2], currents[3]);
			set_currents(dev, currents[0], currents[1], currents[2], currents[3]);
		}
		// Read from RRAM
		else if (strncmp(buffer, CMD_READ , strlen(CMD_READ)) == 0) {
			get_args(buffer,  &num_tokens);
			u16 addr = hex2i(tokens[1]);
			if (atoi(tokens[0]) == 0) printf("%04x\n", read_word(addr, CTRL_INSTR));
			else printf("%04x\n", read_word(addr, CTRL_DATA));
		}
		// Write to RRAM
		else if (strncmp(buffer, CMD_WRITE , strlen(CMD_WRITE)) == 0) {
			get_args(buffer,  &num_tokens);
			u16 addr = hex2i(tokens[1]);
			u16 data = hex2i(tokens[2]);
			if (atoi(tokens[0]) == 0) printf("%d\n", write_word(addr, data, CTRL_INSTR));
			else printf("%d\n", write_word(addr, data, CTRL_DATA));
		}
		// Scan 16 bits
		else if (strncmp(buffer, CMD_SCAN , strlen(CMD_SCAN)) == 0) {
			get_args(buffer,  &num_tokens);
			u16 in0 = (u16) atoi(tokens[0]);
			u16 in1 = (u16) atoi(tokens[1]);
			u16 in2 = (u16) atoi(tokens[2]);
			u64 result = scan(in0, in1, in2);
			printf("%04x %04x %04x\n", (u16) ((result >> 32) & 0xFFFF),
									(u16) ((result >> 16) & 0xFFFF),
									(u16) ((result) & 0xFFFF));
		}
		// Write to FIFO
		else if (strncmp(buffer, CMD_FIFO_WRITE, strlen(CMD_FIFO_WRITE)) == 0) {
			get_args(buffer,  &num_tokens);
			u16 data = hex2i(tokens[0]);
			read_fifo_write(data);
			printf("Writing %04x to FIFO\n", data);
		}

		// Read from FIFO
		else if (strncmp(buffer, CMD_FIFO_READ, strlen(CMD_FIFO_READ)) == 0) {
			u16 x = write_fifo_read();
			printf("%04x\n", x);
		}

		// Check if Write FIFO is empty
		else if (strncmp(buffer, CMD_FIFO_EMPTY, strlen(CMD_FIFO_EMPTY)) == 0) {
			int x = write_fifo_is_empty();
			printf("%d\n", x);
		}

		// Check if Read FIFO is full
		else if (strncmp(buffer, CMD_FIFO_FULL, strlen(CMD_FIFO_FULL)) == 0) {
			int x = read_fifo_is_full();
			printf("%d\n", x);
		}

		// Reset the fifos
		else if (strncmp(buffer, CMD_FIFO_RESET, strlen(CMD_FIFO_RESET)) == 0) {
			get_args(buffer,  &num_tokens);
			u16 in0 = (u16) atoi(tokens[0]);
			reset_fifos(in0);
			printf("FIFOs Reset %04x\n",in0);
		}

		// Reset endurer
		else if (strncmp(buffer, CMD_ENDR_RESET, strlen(CMD_ENDR_RESET)) == 0) {
            int x = reset_endurer_state();
			printf("ENDURER Reset; status: %d\n", x);
		}

		// Change current segment in use
		else if (strncmp(buffer, CMD_CHIP_CHSEGMENT, strlen(CMD_CHIP_CHSEGMENT)) == 0) {
			get_args(buffer,  &num_tokens);
            FORCE_NUM_TOKENS(3);
			e_uint rram_chip    = (e_uint) atoi(tokens[0]);
		    e_uint segment_type = (e_uint) atoi(tokens[1]);
		    e_uint chunk        = (e_uint) atoi(tokens[2]);
		    int x = select_segment(rram_chip, segment_type, chunk);
			printf("Selected segment; status: %d\n", x);
		}

		// Assign a 'physical' chip to a segment
		else if (strncmp(buffer, CMD_ENDR_ASSIGNSEGMENT, strlen(CMD_ENDR_ASSIGNSEGMENT)) == 0) {
			get_args(buffer,  &num_tokens);
            FORCE_NUM_TOKENS(4);
			e_uint phys_chip    = (e_uint) atoi(tokens[0]);
		    e_uint rram_chip    = (e_uint) atoi(tokens[1]);
		    e_uint segment_type = (e_uint) atoi(tokens[2]);
		    e_uint chunk        = (e_uint) atoi(tokens[3]);
		    int x = assign_physical_segment(phys_chip, rram_chip, segment_type, chunk);
			printf("Assigned segment; status: %d\n", x);
		}

		// Verify the loaded segments can work for system (no overlap)
		else if (strncmp(buffer, CMD_ENDR_VERIFYSEGMENTS, strlen(CMD_ENDR_VERIFYSEGMENTS)) == 0) {
		    int x = verify_segments();
			printf("Segment error? %d\n", x);
		}

		// Stream the loaded segments for the system out 
		else if (strncmp(buffer, CMD_ENDR_READOUTSEGMENTS, strlen(CMD_ENDR_READOUTSEGMENTS)) == 0) {
		    read_out_physical_segments();
		}

		// Change endurer mode only
		else if (strncmp(buffer, CMD_ENDR_CHMOD, strlen(CMD_ENDR_CHMOD)) == 0) {
            get_args(buffer,  &num_tokens);
            FORCE_NUM_TOKENS(1);
			e_endurer_mode mode = (e_endurer_mode) atoi(tokens[0]);
		    switch(mode) {
                case DISTRIBUTED_ENDURER: 
                    enable_distributed_endurer(); 
         		    printf("DISTRIBUTED ENDURER enabled\n");
	                break;
                case ENDURER: 
                    enable_endurer(); 
        			printf("ENDURER enabled\n");
                    break;
	            default: 
                    disable_endurer(); 
       			    printf("ENDURER disable\n");
	        }
		}

		// Set Endurer state
		else if (strncmp(buffer, CMD_ENDR_SETSTATE, strlen(CMD_ENDR_SETSTATE)) == 0) {
            get_args(buffer,  &num_tokens);
            FORCE_NUM_TOKENS(29);
			
            e_endurer_mode mode = (e_endurer_mode) atoi(tokens[0]);
            e_count swap_count = (e_count) atoll(tokens[1]);
            e_count counts[8];
            e_uint virt_to_phys_map[8], offsets[8];
            for (int index=0; index<8; index++) virt_to_phys_map[index] = (e_uint) atoi(tokens[2+index]);
            for (int index=0; index<8; index++) offsets[index] = (e_uint) atoi(tokens[10+index]);
            for (int index=0; index<8; index++) counts[index] = (e_count) atoll(tokens[18+index]);
            double TAU = atof(tokens[26]);
            double EPS = atof(tokens[27]);
            double C   = atof(tokens[28]); 
 
            int x = set_endurer_state(mode, swap_count, virt_to_phys_map, offsets, counts, TAU, EPS, C);

            printf("Set Endurer state, status: %d\n", x);
		}

		// Get Endurer state
		else if (strncmp(buffer, CMD_ENDR_GETSTATE, strlen(CMD_ENDR_GETSTATE)) == 0) {
		    e_uint virt_to_phys_map[8], phys_to_virt_map[8], offsets[8];
            e_count counts[8];
            e_count swap_count;
            e_endurer_mode mode;
            double TAU, EPS, C;
            get_endurer_state(&mode, &swap_count, virt_to_phys_map, 
                              phys_to_virt_map, offsets, counts, TAU, EPS, C);

            printf("%1d %5lld ", mode, swap_count);
            for (int index=0; index<8; index++) printf("%1d ", virt_to_phys_map[index]);
            //for (int index=0; index<8; index++) printf("%d ", phys_to_virt_map[index]);
            for (int index=0; index<8; index++) printf("%5d ", offsets[index]);
            for (int index=0; index<8; index++) printf("%20lld ", counts[index]);
            printf("%20lld\n", counts[7]);
            printf("%12.6f %12.6f %12.6f\n", TAU, EPS, C);
		}
 
		// Send the trace to a given address
		else if (strncmp(buffer, CMD_ENDR_SENDTRACE, strlen(CMD_ENDR_SENDTRACE)) == 0) {
            get_args(buffer,  &num_tokens);
            FORCE_NUM_TOKENS(1);
			int trace_addr = (int) hextoi(tokens[0]);
            int x = send_trace_pattern(trace_addr);
            printf("%d\n",x);
		}

        // Run the trace on a specific chip
		else if (strncmp(buffer, CMD_ENDR_RUNTRACE, strlen(CMD_ENDR_RUNTRACE)) == 0) {
            get_args(buffer,  &num_tokens);
            FORCE_NUM_TOKENS(3);
		    int virt_chip = (int) atoi(tokens[0]);
            int trace_offset = (int) hextoi(tokens[1]);
            int trace_length = (int) atoi(tokens[2]);
            int x = run_trace(virt_chip, trace_offset, trace_length);
            printf("Ran trace @%x to len %d on virt %d, status %d\n",
                    trace_offset, trace_length, virt_chip, x);
		}

        // Extract entire trace space
		else if (strncmp(buffer, CMD_ENDR_READOUTTRACE, strlen(CMD_ENDR_READOUTTRACE)) == 0) {
            FORCE_NUM_TOKENS(2);
            int trace_offset = (int) hextoi(tokens[0]);
            int trace_length = (int) atoi(tokens[1]);
            int x = read_out_trace(trace_offset, trace_length);
            printf("Read out trace status: %d\n", x);
		}

        // Run trace through endurer system (based on mode) for certain amount of time
		else if (strncmp(buffer, CMD_ENDR_RUNENDURER, strlen(CMD_ENDR_RUNENDURER)) == 0) {
            get_args(buffer,  &num_tokens);
            FORCE_NUM_TOKENS(4);
            int trace_offset = (int) hextoi(tokens[0]);
            int trace_length = (int) atoi(tokens[1]);
            int times_before_remap = (int) atoi(tokens[2]);
            int total_times = (int) atoi(tokens[3]);
	        int x = run_endurer(trace_offset, trace_length, times_before_remap, total_times);
            printf("Ran endurer at @%x for len %d, %d times with %d times between remaps; status %d\n", trace_offset, trace_length, total_times, times_before_remap, x);
		}

        // Run trace through endurer system (based on mode) for certain amount of time
        // But only on one chip
        // Distributed endurer is not activated directly
		else if (strncmp(buffer, CMD_ENDR_RUNENDURERCHIP, strlen(CMD_ENDR_RUNENDURERCHIP)) == 0) {
            get_args(buffer,  &num_tokens);
            FORCE_NUM_TOKENS(5);
			int chip = (int) atoi(tokens[0]);
            int trace_offset = (int) hextoi(tokens[1]);
            int trace_length = (int) atoi(tokens[2]);
            int times_before_remap = (int) atoi(tokens[3]);
            int total_times = (int) atoi(tokens[4]);
            int x = run_endurer_single_chip(chip, trace_offset, trace_length, times_before_remap, total_times);
	        printf("Ran endurer on virt %d at @%x for len %d, %d times with %d times between remaps; status %d\n", chip, trace_offset, trace_length, total_times, times_before_remap, x);
		}

        // Read memory based on virtual addresses
		else if (strncmp(buffer, CMD_ENDR_READMEM, strlen(CMD_ENDR_READMEM)) == 0) {
            read_out_memory();
		}

        // Read memory based on physical addresses
		else if (strncmp(buffer, CMD_ENDR_READMEMRAW, strlen(CMD_ENDR_READMEMRAW)) == 0) {
            read_out_memory_raw();
		}

        // Set memory of pre-selected chip based on memory traces 
		else if (strncmp(buffer, CMD_ENDR_SETMEM, strlen(CMD_ENDR_SETMEM)) == 0) {
            get_args(buffer,  &num_tokens);
            FORCE_NUM_TOKENS(1);
			e_uint chip = (e_uint) atoi(tokens[0]);
            int x = stream_initial_chip_memory(chip);
            printf("Streamed in memory of virtual chip %d, status %d\n", chip, x);
		}

        // Do distributed remap/chip swap based on physical chip ids
        // if in distributed endurer mode
		else if (strncmp(buffer, CMD_ENDR_SWAP, strlen(CMD_ENDR_SWAP)) == 0) {
            get_args(buffer,  &num_tokens);
            FORCE_NUM_TOKENS(2);
			e_uint ch0 = (e_uint) atoi(tokens[0]);
	        e_uint ch1 = (e_uint) atoi(tokens[1]);
            distributed_remap(ch0, ch1);
            printf("Swapped physical chip %d and %d\n",ch0,ch1);
		}

	}

}
