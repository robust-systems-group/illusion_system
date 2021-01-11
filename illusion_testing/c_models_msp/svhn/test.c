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
//Nvmonic Multi-Chip Case
//All weights/biases are stored on a respective chip
//All activations stored on local SRAM, and messages passed

//MODETARGET
//Best case single chip.  Weights to do not fit, so we re-use weights to proxy as if they did
//SRAM used for buffer

typedef union {
    struct S {
        int8_t input[32*32*3]; //Input data.s.& output write over
        int8_t conv1_out[18*8*8];
        int8_t conv2_out[18*8*8];
        int8_t conv3_out[24*4*4];
        int32_t fc1_part[52];
        int16_t fc1_out[52];
        int16_t fc2_out[60];
        int16_t fc3_out[10];
    } s;
    struct U { //Makes it easier to message pass condensed only usigned 16 bit words
       uint16_t input[32*32*3/2]; //Input data.s.& output write over
       uint16_t conv1_out[18*8*8/2];
       uint16_t conv2_out[18*8*8/2];
       uint16_t conv3_out[24*4*4/2];
       uint16_t fc1_part[52*2];
       uint16_t fc1_out[52];
       uint16_t fc2_out[60];
       uint16_t fc3_out[10];
    } u;
} DTot;

DTot __attribute__((section (".noinit"))) data;
int16_t __attribute__((section (".noinit"))) max_pred;
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

void __attribute__((optimize("O0"))) classify() {
    int i;
    if(mode&(MTARG|MILSM0|MIL0)){ 
#ifdef NV
        for (i = 0; i < 16; i++) {
            conv2D_1filter_pad(data.s.input, m.l12.layer1_H + (i*3*3*3), m.l12.layer1_B[i], 3, 32, 3, 2, data.s.conv1_out + (i*8*8));
        }
        for (i = 16; i < 18; i++) {
            conv2D_1filter_pad_nv(data.s.input, m.l12.layer1_H + (i*3*3*3), m.l12.layer1_B[i], 3, 32, 3, 2, m.l12.nv.nv_conv1_sm + ((i-16)*8*8/2));
        }
        for (i = 0; i < 2*8*8; i++) {
            data.s.conv1_out[16*8*8+i] = m.l12.nv.nv_conv1_out[i];
        }
#else
        for (i = 0; i < 18; i++) {
            conv2D_1filter_pad(data.s.input, m.l12.layer1_H + (i*3*3*3), m.l12.layer1_B[i], 3, 32, 3, 2, data.s.conv1_out + (i*8*8));
        }
#endif
        for (i = 0; i < 18; i++) {
            conv2D_1filter_pad(data.s.conv1_out, m.l12.layer2_H + (i*3*3*18), m.l12.layer2_B[i], 18, 8, 3, 0, data.s.conv2_out + (i*8*8));
        }
    }
    if(mode&(MTARG|MILSM0|MIL1)){ 
        for (i = 0; i < 24; i++) {
            conv2D_1filter_pad(data.s.conv2_out, m.l3.layer3_H + (i*3*3*18), m.l3.layer3_B[i], 18, 8, 3, 1, data.s.conv3_out + (i*4*4));
        }
    }
    if(mode&(MTARG|MILSM1|MIL2)){ 
        for (i = 0; i < 52*2; i++)      {data.u.fc1_part[i] = 0;}
        dense(data.s.conv3_out, m.l4.layer4_H, data.s.fc1_part, 77, 52);
    }
    if(mode&(MTARG|MILSM1|MIL3)){ 
        dense(data.s.conv3_out + 77, m.l4.layer4_H, data.s.fc1_part, 77, 52); 
    }
    if(mode&(MTARG|MILSM2|MIL4)){ 
        dense(data.s.conv3_out + 2*77, m.l4.layer4_H, data.s.fc1_part, 77, 52); 
    }
    if(mode&(MTARG|MILSM2|MIL5)){ 
        dense(data.s.conv3_out + 3*77, m.l4.layer4_H, data.s.fc1_part, 77, 52); 
    }
    if(mode&(MTARG|MILSM3|MIL6)){ 
        dense(data.s.conv3_out + 4*77, m.l4.l4s.layer4_H, data.s.fc1_part, 76, 52); //Divided slightly unequallyt 
        add_bias(m.l4.l4s.layer4_B, data.s.fc1_part, data.s.fc1_out, 52);           //Data read in summed partial products, now bias correctly
    }
    if(mode&(MTARG|MILSM3|MIL7)){ 
        dense_bias(data.s.fc1_out, m.l56.layer5_H, m.l56.layer5_B, data.s.fc2_out, 52, 60);
        dense_bias(data.s.fc2_out, m.l56.layer6_H, m.l56.layer6_B, data.s.fc3_out, 60, 10);
        
        int16_t pred = INT16_MIN;
        max_pred = 0;
        for (i =0; i < 10; i++ ) {
            if (data.s.fc3_out[i] > pred){
                pred = data.s.fc3_out[i];
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
        for (i = 0; i < 32*32*3/2; i++) {data.u.input[i] = *(SENSOR_PORT);}     //Get input
    } else if(mode&(MILSM1)){ 
        for (i = 0; i < 77; i++)        {data.u.conv3_out[i] = *(SENSOR_PORT);} 
    } else if(mode&(MILSM2)){ 
        for (i = 77; i < 77*2; i++)     {data.u.conv3_out[i] = *(SENSOR_PORT);} 
        for (i = 0; i < 52*2; i++)      {data.u.fc1_part[i] = *(SENSOR_PORT);}
    } else if(mode&(MILSM3|MIL6)){ 
        for (i = 77*2; i < 384/2; i++)  {data.u.conv3_out[i] = *(SENSOR_PORT);} 
        for (i = 0; i < 52*2; i++)      {data.u.fc1_part[i] = *(SENSOR_PORT);}
    } else if(mode&(MIL1)){ 
        for (i = 0; i < 8*8*18/2; i++)  {data.u.conv2_out[i] = *(SENSOR_PORT);}              //Initialize Buffers
    } else if(mode&(MIL2)){ 
        for (i = 0; i < 78/2; i++)      {data.u.conv3_out[i] = *(SENSOR_PORT);} 
    } else if(mode&(MIL3)){ 
        for (i = 76/2; i < 154/2; i++)  {data.u.conv3_out[i] = *(SENSOR_PORT);} 
        for (i = 0; i < 52*2; i++)      {data.u.fc1_part[i] = *(SENSOR_PORT);}
    } else if(mode&(MIL4)){ 
        for (i = 77; i < 232/2; i++)    {data.u.conv3_out[i] = *(SENSOR_PORT);} 
        for (i = 0; i < 52*2; i++)      {data.u.fc1_part[i] = *(SENSOR_PORT);}
    } else if(mode&(MIL5)){ 
        for (i = 230/2; i < 77*2; i++)  {data.u.conv3_out[i] = *(SENSOR_PORT);} 
        for (i = 0; i < 52*2; i++)      {data.u.fc1_part[i] = *(SENSOR_PORT);}
    } else if(mode&(MIL7)){ 
        for (i = 0; i < 52; i++)        {data.u.fc1_out[i] = *(SENSOR_PORT);}
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
        *(SENSOR_PORT) = MILSM1; //Send in parallel
        for (i = 0; i < 77; i++)   {*(SENSOR_PORT) = data.u.conv3_out[i];}
        *(SENSOR_PORT) = MILSM2; //Send in parallel
        for (i =77; i < 77*2; i++)   {*(SENSOR_PORT) = data.u.conv3_out[i];}
        *(SENSOR_PORT) = MILSM3; //Send in parallel 
        for (i = 77*2; i < 384/2; i++)  {*(SENSOR_PORT) = data.u.conv3_out[i];}
    } else if(mode&(MILSM1)){ 
        *(SENSOR_PORT) = MILSM3; //Send in parallel
        for (i = 0; i < 52*2; i++)      {*(SENSOR_PORT) = data.u.fc1_part[i];}
    } else if(mode&(MILSM2)){ 
        *(SENSOR_PORT) = MILSM3; //Send in parallel
        for (i = 0; i < 52*2; i++)   {*(SENSOR_PORT) = data.u.fc1_part[i];}
    } else if(mode&(MIL0)){ 
        *(SENSOR_PORT) = MIL1; //Send in parallel
        for (i = 0; i < 8*8*18/2; i++) {*(SENSOR_PORT) = data.u.conv2_out[i];}
    } else if(mode&(MIL1)){ 
        *(SENSOR_PORT) = MIL2; //Send in parallel
        for (i = 0; i < 78/2; i++)      {*(SENSOR_PORT) = data.u.conv3_out[i];} 
        *(SENSOR_PORT) = MIL3; //Send in parallel
        for (i = 76/2; i < 154/2; i++)  {*(SENSOR_PORT) = data.u.conv3_out[i];} 
        *(SENSOR_PORT) = MIL4; //Send in parallel
        for (i = 77; i < 232/2; i++)    {*(SENSOR_PORT) = data.u.conv3_out[i];}
        *(SENSOR_PORT) = MIL5; //Send in parallel
        for (i = 230/2; i < 77*2; i++)  {*(SENSOR_PORT) = data.u.conv3_out[i];} 
        *(SENSOR_PORT) = MIL6; //Send in parallel
        for (i = 77*2; i < 384/2; i++)  {*(SENSOR_PORT) = data.u.conv3_out[i];} 
    } else if(mode&(MIL2)){ 
        *(SENSOR_PORT) = MIL3; //Send in parallel
        for (i = 0; i < 52*2; i++)      {*(SENSOR_PORT) = data.u.fc1_part[i];}
    } else if(mode&(MIL3)){ 
        *(SENSOR_PORT) = MIL4; //Send in parallel
        for (i = 0; i < 52*2; i++)      {*(SENSOR_PORT) = data.u.fc1_part[i];}
    } else if(mode&(MIL4)){ 
        *(SENSOR_PORT) = MIL5; //Send in parallel
        for (i = 0; i < 52*2; i++)      {*(SENSOR_PORT) = data.u.fc1_part[i];}
    } else if(mode&(MIL5)){ 
        *(SENSOR_PORT) = MIL6; //Send in parallel
        for (i = 0; i < 52*2; i++)      {*(SENSOR_PORT) = data.u.fc1_part[i];}
    } else if(mode&(MIL6)){ 
        *(SENSOR_PORT) = MIL7; //Send in parallel
        for (i = 0; i < 52; i++)        {*(SENSOR_PORT) = data.u.fc1_out[i];}
    }
    return;
}

