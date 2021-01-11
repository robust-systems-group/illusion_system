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

#include "tensor.h"
#include "omsp_func.h"
#include "test.h"
#include "model.c"

//Setup the data.s.structure for the different multi-chip modes
//Note that model.c contains the RRAM weights for this case

//MODEILLUSION
//Nvmonic Multi-Chip Case - Split into 8
//All weights/biases are stored on a respective chip
//All activations stored on local SRAM, and messages passed

//MODEILLUSIONSM
//Nvmonic Multi-Chip Case - Split into 4
//All weights/biases are stored on a respective chip
//All activations stored on local SRAM, and messages passed

//MODETARGET
//Best case single chip.  Weights to do not fit, so we re-use weights to proxy as if they did
//All activations in SRAM

typedef union {
    struct S {
        int8_t input[IDIM];
        int8_t h_in[HDIM];
        int16_t c_in[HDIM];
        int8_t h_out[HDIM];
        int16_t c_out[HDIM];
        int32_t classes_partial[ODIM];
        int16_t classes[ODIM];
    } s;
    struct U { //Makes it easier to message pass condensed only usigned 16 bit words
        uint16_t input[IDIM/2];
        uint16_t h_in[HDIM/2];
        uint16_t c_in[HDIM];
        uint16_t h_out[HDIM/2];
        uint16_t c_out[HDIM];
        uint16_t classes_partial[ODIM*2];
        uint16_t classes[ODIM];
    } u;
} DTot;

DTot __attribute__((section (".noinit"))) data;

int16_t __attribute__((section (".noinit"))) max_pred;
int16_t __attribute__((section (".noinit"))) pred_prob;
uint16_t __attribute__((section (".noinit"))) last_input;
int  __attribute__((section (".noinit"))) start;
int  __attribute__((section (".noinit"))) end;
int  __attribute__((section (".noinit"))) times;

uint16_t __attribute__((section (".noinit"))) mode;

//Modes are one-hot
#define MTARG 0x8000

#define MILSM0 0x0100
#define MILSM1 0x0200
#define MILSM2 0x0400
#define MILSM3 0x0800

#define MIL0 0x0001
#define MIL1 0x0002
#define MIL2 0x0004
#define MIL3 0x0008
#define MIL4 0x0010
#define MIL5 0x0020
#define MIL6 0x0040
#define MIL7 0x0080

void set_start(uint16_t md){ // LUT function to determine HDIM loops
    if(mode&(MTARG|MILSM0|MIL0)){   start = 0*(HDIM)/8;}
    else if(mode&(MIL1)){           start = 1*(HDIM)/8;}
    else if(mode&(MILSM1|MIL2)){    start = 2*(HDIM)/8;}
    else if(mode&(MIL3)){           start = 3*(HDIM)/8;}
    else if(mode&(MILSM2|MIL4)){    start = 4*(HDIM)/8;}
    else if(mode&(MIL5)){           start = 5*(HDIM)/8;}
    else if(mode&(MILSM3|MIL6)){    start = 6*(HDIM)/8;}
    else if(mode&(MIL7)){           start = 7*(HDIM)/8;}
}

void set_end(uint16_t md){ // LUT function to determine HDIM loops
    if(mode&(MTARG|MILSM3|MIL7)){   end  = 8*(HDIM)/8;}
    else if(mode&(MIL6)){           end  = 7*(HDIM)/8;}
    else if(mode&(MILSM2|MIL5)){    end  = 6*(HDIM)/8;}
    else if(mode&(MIL4)){           end  = 5*(HDIM)/8;}
    else if(mode&(MILSM1|MIL3)){    end  = 4*(HDIM)/8;}
    else if(mode&(MIL2)){           end  = 3*(HDIM)/8;}
    else if(mode&(MILSM0|MIL1)){    end  = 2*(HDIM)/8;}
    else if(mode&(MIL0)){           end  = 1*(HDIM)/8;}
}

void set_times(uint16_t md){ // LUT function to determine HDIM loops
    if(mode&(MTARG)){times = 8;}
    else if(mode&(MILSM0|MILSM1|MILSM2|MILSM3)){times = 2;}
    else if(mode&(MIL0|MIL1|MIL2|MIL3|MIL4|MIL5|MIL6|MIL7)){ times = 1;}
}

void __attribute__((optimize("O0"))) classify(){
    int i,j,k = 0;
    //Layer  1
#if NVCELL
    int16_t * cell_addr = cell_state;
#else
    int16_t * cell_addr = data.s.c_in;
#endif
    for (j=0; j < times; j++) {
        k = start+j*HDIM/8;
        lstm(data.s.input, data.s.h_in, cell_addr + k, m.lstm.lstm_i_H, m.lstm.lstm_h_H, m.lstm.lstm_B, HDIM/8, data.s.h_out + k, data.s.c_out + k);
        if(last_input){dense(data.s.h_out+k, m.fc.fc_H, data.s.classes_partial, HDIM/8, ODIM);}
    }
    if(mode&(MTARG|MILSM3|MIL7)&&last_input){
        add_bias(m.fc.fc_B, data.s.classes_partial, data.s.classes, ODIM); 
        pred_prob = INT16_MIN;
        max_pred = 0;
        for (i = 0; i < 11; i++ ) {
            if (data.s.classes[i] > pred_prob){
                pred_prob = data.s.classes[i];
                max_pred = i;
            }
        }
    }
    return;
}
 
void __attribute__((optimize("O0"))) read_input() {
    int i;
    mode = *(SENSOR_PORT); //Read first value (bogus flip flop dropped now due to timing) 
    set_start(mode);
    set_end(mode);
    set_times(mode);
    if(!(mode&0x8FFF)){CORE_DONE;};
    last_input = *(SENSOR_PORT);
    for (i = 0; i < IDIM/2; i++)   {data.u.input[i] = *(SENSOR_PORT);}     //Get input
    for (i = 0; i < HDIM/2; i++)   {data.u.h_in[i] = *(SENSOR_PORT);}     //Get input

    if(mode&(MTARG|MILSM0|MIL0)){ 
        if(last_input){for (i = 0; i < ODIM*2; i++)   {data.u.classes_partial[i] = 0;}}
    } else if(mode&(MILSM1|MILSM2|MILSM3|MIL1|MIL2|MIL3|MIL4|MIL5|MIL6|MIL7)){ 
        if(last_input){for (i = 0; i < ODIM*2; i++)   {data.u.classes_partial[i] = *(SENSOR_PORT);}}
    }
#if !NVCELL
    uint16_t d = *(SENSOR_PORT); //Overread to empty fifo 
    for (i = start; i < end; i++)   {data.u.c_in[i] = *(SENSOR_PORT);}
#endif
    uint16_t c = *(SENSOR_PORT); //Overread to empty fifo 
    if (c != 0) {
        CORE_DONE;  ///Last input needs to be 0 otherwise MAC doesn't work
    }
    return;
}

//Send the ouptut to the next chip
void __attribute__((optimize("O0"))) send_output() {
    int i;
    if(mode&(MTARG|MILSM3|MIL7)){ 
        *(SENSOR_PORT) = 0;
        if(last_input){ *(SENSOR_PORT) = (uint16_t) max_pred;}
    } else if(mode&(MILSM0|MILSM1|MILSM2|MIL0|MIL1|MIL2|MIL3|MIL4|MIL5|MIL6)){ 
        *(SENSOR_PORT) = (mode<<1);
        if(last_input){for (i = 0; i < ODIM*2; i++)   {*(SENSOR_PORT) = data.u.classes_partial[i];}}
    }
    for (i = (start>>1); i < (end>>1); i++)      {*(SENSOR_PORT) = data.u.h_out[i];}     //Get input
#if NVCELL
        for (i = start; i < end; i++)       {cell_state[i] = data.s.c_out[i];}
#else
        for (i = start; i < end; i++)       {*(SENSOR_PORT) = data.u.c_out[i];}
#endif 
    return;
}

