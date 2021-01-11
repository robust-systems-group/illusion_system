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
        int8_t input[28*28];
        int8_t conv1_out[12*12*6];
        int8_t conv2_out[10*10*6];
        int8_t conv3_out[4*4*8];
        int8_t conv4_out[2*2*24];
        int32_t fc1_out[10];
        int16_t classes[10];
    } s;
    struct U { //Makes it easier to message pass condensed only usigned 16 bit words
        uint16_t input[28*28/2];
        uint16_t conv1_out[12*12*6/2];
        uint16_t conv2_out[10*10*6/2];
        uint16_t conv3_out[4*4*8/2];
        uint16_t conv4_out[2*2*24/2];
        uint16_t fc1_out[10*2];
        uint16_t classes[10];
    } u;
} DTot;


DTot __attribute__((section (".noinit"))) data;
int16_t __attribute__((section (".noinit"))) max_pred;
int16_t __attribute__((section (".noinit"))) pred_prob;
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



void __attribute__((optimize("O0"))) classify(){
    int i;
    if(mode&(MTARG|MILSM0|MIL0)){ 
        for (i = 0; i < 6; i++) {
            conv2D_1filter(data.s.input, layer1_H + (i*5*5), layer1_B[i], 1, 28, 5, 1, data.s.conv1_out + (i*12*12));
        }
        for (i = 0; i < 6; i++) {
            conv2D_1filter(data.s.conv1_out, layer2_H + (i*3*3*6), layer2_B[i], 6, 12, 3, 0, data.s.conv2_out + (i*10*10));
        }
    }
    if(mode&(MTARG|MILSM0|MIL1)){ 
        for (i = 0; i < 8; i++) {
            conv2D_1filter(data.s.conv2_out, layer3_H + (i*3*3*6), layer3_B[i], 6, 10, 3, 1, data.s.conv3_out + (i*4*4));
        }
    }
    if(mode&(MTARG|MILSM1|MIL2)){ 
        for (i = 0; i < 6; i++) {
            conv2D_1filter(data.s.conv3_out, layer4a_H + (i*3*3*8), layer4a_B[i], 8, 4, 3, 0, data.s.conv4_out + ((0+i)*2*2));
        }
    }
    if(mode&(MTARG|MILSM1|MIL3)){ 
        for (i = 0; i < 6; i++) {
            conv2D_1filter(data.s.conv3_out, layer4b_H + (i*3*3*8), layer4b_B[i], 8, 4, 3, 0, data.s.conv4_out + ((6+i)*2*2));
        }
    }
    if(mode&(MTARG|MILSM2|MIL4)){ 
        for (i = 0; i < 6; i++) {
            conv2D_1filter(data.s.conv3_out, layer4c_H + (i*3*3*8), layer4c_B[i], 8, 4, 3, 0, data.s.conv4_out + ((12+i)*2*2));
        }
    }
    if(mode&(MTARG|MILSM2|MIL5)){ 
        for (i = 0; i < 6; i++) {
            conv2D_1filter(data.s.conv3_out, layer4d_H + (i*3*3*8), layer4d_B[i], 8, 4, 3, 0, data.s.conv4_out + ((18+i)*2*2));
        }
    }
    if(mode&(MTARG|MILSM3|MIL6)){ 
        for (i = 0; i < 20; i++)        {data.u.fc1_out[i] = 0;} //Dense adds to inpt
        dense(data.s.conv4_out, layer5a_H, data.s.fc1_out, 48, 10);
    }
    if(mode&(MTARG|MILSM3|MIL7)){ 
        dense(data.s.conv4_out+48, layer5b_H, data.s.fc1_out, 48, 10);
        add_bias(layer5_B, data.s.fc1_out, data.s.classes, 10);
        
        pred_prob = INT16_MIN;
        max_pred = 0;
        for (i = 0; i < 10; i++ ) {
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
    mode = (uint16_t) *(SENSOR_PORT); //Read first value (bogus flip flop dropped now due to timing) 
    if(mode&(MTARG|MILSM0|MIL0)){ 
        for (i = 0; i < 28*28/2; i++)   {data.u.input[i] = *(SENSOR_PORT);}     //Get input
    } else if(mode&(MILSM1|MILSM2|MIL2|MIL3|MIL4|MIL5)){ 
        for (i = 0; i < 4*4*8/2; i++)   {data.u.conv3_out[i] = *(SENSOR_PORT);}//Get input
    } else if(mode&(MILSM3)){ 
        for (i = 0; i < 2*2*24/2; i++)  {data.u.conv4_out[i] = *(SENSOR_PORT);}//Get input
    } else if(mode&(MIL1)){ 
        for (i = 0; i < 10*10*6/2; i++) {data.u.conv2_out[i] = *(SENSOR_PORT);}//Get input
    } else if(mode&(MIL6)){ 
        for (i = 0; i < 48/2; i++)      {data.u.conv4_out[i] = *(SENSOR_PORT);} //Get input
    } else if(mode&(MIL7)){ 
        for (i = 48/2; i < 96/2; i++)   {data.u.conv4_out[i] = *(SENSOR_PORT);} //Get input
        for (i = 0; i < 20; i++)        {data.u.fc1_out[i] = *(SENSOR_PORT);} //Get input
    }
    uint16_t c = (uint16_t) *(SENSOR_PORT); //Overread to empty fifo 
    if (c != 0) {
        CORE_DONE;  ///Last input needs to be 0 otherwise MAC doesn't work
    }
    return;
}

void __attribute__((optimize("O0"))) send_output() {
    int i;
    if(mode&(MTARG|MILSM3|MIL7)){ 
        *(SENSOR_PORT) = 0; //Return to host
        *(SENSOR_PORT) = (uint16_t) max_pred;
    } else if(mode&(MILSM0)){ 
        *(SENSOR_PORT) = MILSM1; 
        for (i = 0; i < 4*4*8/2; i++)   {*(SENSOR_PORT) = data.u.conv3_out[i];}
        *(SENSOR_PORT) = MILSM2; //send in parallel
        for (i = 0; i < 4*4*8/2; i++)   {*(SENSOR_PORT) = data.u.conv3_out[i];}
    } else if(mode&(MILSM1)){ 
        *(SENSOR_PORT) = MILSM3; //Send in parallel
        for (i = 0; i < 48/2; i++)      {*(SENSOR_PORT) = data.u.conv4_out[i];}
    } else if(mode&(MILSM2)){ 
        *(SENSOR_PORT) = MILSM3; //Send in parallel
        for (i = 48/2; i < 96/2; i++)   {*(SENSOR_PORT) = data.u.conv4_out[i];}
    } else if(mode&(MIL0)){ 
        *(SENSOR_PORT) = MIL1; 
        for (i = 0; i < 10*10*6/2; i++) {*(SENSOR_PORT) = data.u.conv2_out[i];}
    } else if(mode&(MIL1)){ 
        *(SENSOR_PORT) = MIL2; //Send in parallel
        for (i = 0; i < 4*4*8/2; i++)   {*(SENSOR_PORT) = data.u.conv3_out[i];}
        *(SENSOR_PORT) = MIL3; //Send in parallel
        for (i = 0; i < 4*4*8/2; i++)   {*(SENSOR_PORT) = data.u.conv3_out[i];}
        *(SENSOR_PORT) = MIL4; //Send in parallel
        for (i = 0; i < 4*4*8/2; i++)   {*(SENSOR_PORT) = data.u.conv3_out[i];}
        *(SENSOR_PORT) = MIL5; //Send in parallel
        for (i = 0; i < 4*4*8/2; i++)   {*(SENSOR_PORT) = data.u.conv3_out[i];}
    } else if(mode&(MIL2)){ 
        *(SENSOR_PORT) = MIL6; 
        for (i = 0; i < 24/2; i++)      {*(SENSOR_PORT) = data.u.conv4_out[i];}
    } else if(mode&(MIL3)){ 
        *(SENSOR_PORT) = MIL6; 
        for (i = 24/2; i < 48/2; i++)   {*(SENSOR_PORT) = data.u.conv4_out[i];}
    } else if(mode&(MIL4)){ 
        *(SENSOR_PORT) = MIL7; 
        for (i = 48/2; i < 72/2; i++)   {*(SENSOR_PORT) = data.u.conv4_out[i];}
    } else if(mode&(MIL5)){ 
        *(SENSOR_PORT) = MIL7; 
        for (i = 72/2; i < 96/2; i++)   {*(SENSOR_PORT) = data.u.conv4_out[i];}
    } else if(mode&(MIL6)){ 
        *(SENSOR_PORT) = MIL7; 
        for (i = 0; i < 20; i++)        {*(SENSOR_PORT) = data.u.fc1_out[i];}
    }
    return;
}


