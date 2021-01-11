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

#include "tensor.h"
#include "omsp_func.h"
#include "test.h"
#include "model.c"

//Setup the data.s.structure for the different multi-chip modes
//Note that model.c contains the RRAM weights for this case

//MODEsh at simulation time  21.909 ms
//ILLUSION
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
        int8_t n1_conv1_out[6*6*6];
        int8_t n1_conv2_out[6*6*12];
        int8_t q_input[3*3*12];
        int8_t q_fc1_out[64];
        int8_t q_fc2_out[2];
        int8_t n2_conv1_out[4*4*12];
        int8_t n2_conv2_out[2*2*24];
        int8_t n2_fc1_out[128];
        int8_t n2_fc2_out[24];
        int8_t n3_fc1_out[32];
        int8_t classes[10];
    } s;
    struct U { //Makes it easier to message pass condensed only usigned 16 bit words
        uint16_t input[28*28/2];
        uint16_t n1_conv1_out[6*6*6/2];
        uint16_t n1_conv2_out[6*6*12/2];
        uint16_t q_input[3*3*12/2];
        uint16_t q_fc1_out[64/2];
        uint16_t q_fc2_out[2/2];
        uint16_t n2_conv1_out[4*4*12/2];
        uint16_t n2_conv2_out[2*2*24/2];
        uint16_t n2_fc1_out[128/2];
        uint16_t n2_fc2_out[24/2];
        uint16_t n3_fc1_out[32/2];
        uint16_t classes[10/2];
    } u;
} DTot;

DTot __attribute__((section (".noinit"))) data;

int16_t __attribute__((section (".noinit"))) max_pred;
int16_t __attribute__((section (".noinit"))) pred_prob;
uint16_t __attribute__((section (".noinit"))) high_low;
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
            conv2D_1filter(data.s.input, m.n1q.N1_layer1_H + (i*5*5),  m.n1q.N1_layer1_B[i], 1, 28, 5, 2, data.s.n1_conv1_out + (i*6*6));
        }
        for (i = 0; i < 12; i++) {
            conv2D_1filter(data.s.n1_conv1_out,  m.n1q.N1_layer2_H + (i*1*1*6),  m.n1q.N1_layer2_B[i], 6, 6, 1, 0, data.s.n1_conv2_out + (i*6*6));
        }
        for (i = 0; i < 12; i++) {
            max_pool(data.s.n1_conv2_out +(i*6*6), 6, 1, data.s.q_input + (i*3*3));
        }
        dense_bias(data.s.q_input, m.n1q.Q_layer1a_H, m.n1q.Q_layer1a_B, data.s.q_fc1_out, 108, 28);
    }
    if(mode&(MTARG|MILSM0|MIL1)){ 
        dense_bias(data.s.q_input, m.q.Q_layer1b_H, m.q.Q_layer1b_B, data.s.q_fc1_out+28, 108, 36);
        dense_bias(data.s.q_fc1_out, m.q.Q_layer2_H, m.q.Q_layer2_B, data.s.q_fc2_out, 64, 2);
    }
    if(mode&(MIL1)){    
        //Only time that model is correct to determine path, otherwise external override used (for target, etc).
        if ( data.s.q_fc2_out[0] > data.s.q_fc2_out[1] ) {
            high_low = 1;    
        } else { 
            high_low = 0;
        }
    }
    if (high_low==0){
        if(mode&(MTARG|MILSM1|MIL2)){
            dense_bias(data.s.q_input, m.n3.N3_layer1_H, m.n3.N3_layer1_B, data.s.n3_fc1_out, 3*3*12, 32);
            dense_bias(data.s.n3_fc1_out, m.n3.N3_layer2_H, m.n3.N3_layer2_B, data.s.classes, 32, 10);
            pred_prob = INT16_MIN;
            max_pred = 0;
            for (i = 0; i < 10; i++ ) {
                if (data.s.classes[i] > pred_prob){
                    pred_prob = data.s.classes[i];
                    max_pred = i;
                }
            }
        }
    } else {
        if(mode&(MTARG|MILSM1|MIL3)){ 
            for (i = 0; i < 12; i++) {
                conv2D_1filter(data.s.n1_conv2_out, m.n2a.N2_layer1_H + (i*3*3*12), m.n2a.N2_layer1_B[i], 12, 6, 3, 0, data.s.n2_conv1_out + (i*4*4));
            }
            for (i = 0; i < 24; i++) {
                conv2D_1filter(data.s.n2_conv1_out, m.n2a.N2_layer2_H + (i*3*3*12), m.n2a.N2_layer2_B[i], 12, 4, 3, 0, data.s.n2_conv2_out + (i*2*2));
            }
        }
        if(mode&(MTARG|MILSM2|MIL4)){ 
            dense_bias(data.s.n2_conv2_out, m.n2b.N2_layer3_H, m.n2b.N2_layer3_B, data.s.n2_fc1_out, 96, 42);
        }
        if(mode&(MTARG|MILSM2|MIL5)){ 
            dense_bias(data.s.n2_conv2_out, m.n2b.N2_layer3_H, m.n2b.N2_layer3_B, data.s.n2_fc1_out+42, 96, 42);
        }
        if(mode&(MTARG|MILSM3|MIL6)){ 
            dense_bias(data.s.n2_conv2_out, m.n2b.N2_layer3_H, m.n2b.N2_layer3_B, data.s.n2_fc1_out+42*2, 96, 42);
        }
        if(mode&(MTARG|MILSM3|MIL7)){ 
            dense_bias(data.s.n2_conv2_out, m.n2c.N2_layer3_H, m.n2c.N2_layer3_B, data.s.n2_fc1_out+42*3, 96, 2);
            dense_bias(data.s.n2_fc1_out, m.n2c.N2_layer4_H, m.n2c.N2_layer4_B, data.s.n2_fc2_out, 128, 24);
            dense_bias(data.s.n2_fc2_out, m.n2c.N2_layer5_H, m.n2c.N2_layer5_B, data.s.classes, 24, 10);
            pred_prob = INT16_MIN;
            max_pred = 0;
            for (i = 0; i < 10; i++ ) {
                if (data.s.classes[i] > pred_prob){
                    pred_prob = data.s.classes[i];
                    max_pred = i;
                }
            }
        }
    }
    return;
}

void __attribute__((optimize("O0"))) read_input() {
    int i;
    mode = (uint16_t) *(SENSOR_PORT); //Read first value (bogus flip flop dropped now due to timing) 
    if(!(mode&0x8FFF)){CORE_DONE;};
    if(mode&(MTARG|MILSM0)){ 
        high_low =  *(SENSOR_PORT);
        for (i = 0; i < 28*28/2; i++)   {data.u.input[i] = *(SENSOR_PORT);} 
    } else if(mode&(MILSM1)){ 
        high_low =  *(SENSOR_PORT);
        if (high_low == 0){
            for (i = 0; i < 12*3*3/2; i++)  {data.u.q_input[i] = *(SENSOR_PORT);}
        } else {
            for (i = 0; i < 12*6*6/2; i++)  {data.u.n1_conv2_out[i] = *(SENSOR_PORT);}
        }
    } else if(mode&(MILSM2)){ 
        high_low =  *(SENSOR_PORT);
        for (i = 0; i < 96/2; i++)  {data.u.n2_conv2_out[i] = *(SENSOR_PORT);}
    } else if(mode&(MILSM3)){ 
        high_low =  *(SENSOR_PORT);
        for (i = 0; i < 96/2; i++)  {data.u.n2_conv2_out[i] = *(SENSOR_PORT);}
        for (i = 0; i < 84/2; i++)  {data.u.n2_fc1_out[i] = *(SENSOR_PORT);}
    } else if(mode&(MIL0)){ 
        high_low = 1; //Default to low path unless data says otherwise
        #if NVACTIV
            uint16_t get_nvval = *(SENSOR_PORT);   
            if (get_nvval == 0xffff){ //This can't be an input for the MNIST
                *(SENSOR_PORT) = MIL3; //Wake up only to send output to High path
                *(SENSOR_PORT) = 1;
                for (i = 0; i < 12*6*6/2; i++)  {*(SENSOR_PORT) = m.n1q.nv.n1_conv2_out_nvout[i];}//Get input
                #ifdef TIME
                    END_TIME;  // Clear P3[0]
                #endif
                CORE_DONE; 
            }
            else {
                for (i = 0; i < 28*28/2; i++)   {data.u.input[i] = *(SENSOR_PORT);}     //Get input
            }
        #else
            for (i = 0; i < 28*28/2; i++)   {data.u.input[i] = *(SENSOR_PORT);}     //Get input
        #endif
    } else if(mode&(MIL1)){ 
        high_low = 1; //Default to low path unless data says otherwise
        #if NVACTIV
            for (i = 0; i < 12*3*3/2; i++)   {data.u.q_input[i] = *(SENSOR_PORT);}
            for (i = 0; i < 28/2; i++)   {data.u.q_fc1_out[i] = *(SENSOR_PORT);}
        #else
            for (i = 0; i < 12*6*6/2; i++)   {data.u.n1_conv2_out[i] = *(SENSOR_PORT);}
            for (i = 0; i < 28/2; i++)   {data.u.q_fc1_out[i] = *(SENSOR_PORT);}
            for (i = 0; i < 12; i++) {
                max_pool(data.s.n1_conv2_out +(i*6*6), 6, 1, data.s.q_input + (i*3*3));
            }
        #endif
    } else if(mode&(MIL2)){ 
        high_low =  *(SENSOR_PORT);
        for (i = 0; i < 12*3*3/2; i++)   {data.u.q_input[i] = *(SENSOR_PORT);}
    } else if(mode&(MIL3)){ 
        high_low =  *(SENSOR_PORT);
        for (i = 0; i < 12*6*6/2; i++)   {data.u.n1_conv2_out[i] = *(SENSOR_PORT);}
    } else if(mode&(MIL4|MIL5|MIL6)){ 
        high_low =  *(SENSOR_PORT);
        for (i = 0; i < 96/2; i++)  {data.u.n2_conv2_out[i] = *(SENSOR_PORT);}
    } else if(mode&(MIL7)){ 
        high_low =  *(SENSOR_PORT);
        for (i = 0; i < 96/2; i++)  {data.u.n2_conv2_out[i] = *(SENSOR_PORT);}
        for (i = 0; i < (128-2)/2; i++)  {data.u.n2_fc1_out[i] = *(SENSOR_PORT);}
    }
    uint16_t c = (uint16_t) *(SENSOR_PORT); //Overread to empty fifo 
    if (c != 0) {
        CORE_DONE;  ///Last input needs to be 0 otherwise MAC doesn't work
    }
    return;
}

void __attribute__((optimize("O0"))) send_output() {
    int i;
    if(mode&(MTARG|MILSM3|MIL2|MIL7)){ 
        *(SENSOR_PORT) = 0;
        *(SENSOR_PORT) = (uint16_t) max_pred;
    } else if(mode&(MILSM0)){ 
        if (high_low == 0){
            *(SENSOR_PORT) = MILSM1;
            *(SENSOR_PORT) = high_low;
            for (i = 0; i < 12*3*3/2; i++)  {*(SENSOR_PORT) = data.u.q_input[i];}//Get input
        } else {
            *(SENSOR_PORT) = MILSM2;
            *(SENSOR_PORT) = high_low;
            for (i = 0; i < 12*6*6/2; i++)  {*(SENSOR_PORT) = data.u.n1_conv2_out[i];}//Get input
        }
    } else if(mode&(MILSM1)){ 
        if (high_low == 0){
            *(SENSOR_PORT) = 0;
            *(SENSOR_PORT) = (uint16_t) max_pred;
        } else {
            *(SENSOR_PORT) = MILSM2;
            *(SENSOR_PORT) = high_low;
            for (i = 0; i < 2*2*24/2; i++)   {*(SENSOR_PORT) = data.u.n2_conv2_out[i];}
            *(SENSOR_PORT) = MILSM3; //Send in parallel
            *(SENSOR_PORT) = high_low;
            for (i = 0; i < 2*2*24/2; i++)   {*(SENSOR_PORT) = data.u.n2_conv2_out[i];}
        }
    } else if(mode&(MILSM2)){ 
        *(SENSOR_PORT) = MILSM3; //Should be 3 next
        for (i = 0; i < 84/2; i++)   {*(SENSOR_PORT) = data.u.n2_fc1_out[i];}
    } else if(mode&(MIL0)){ 
        *(SENSOR_PORT) = MIL1;
        #if NVACTIV
            for (i = 0; i < 12*3*3/2; i++)   {*(SENSOR_PORT) = data.u.q_input[i];}
            for (i = 0; i < 12*6*6; i++)  {m.n1q.nv.n1_conv2_out_nv[i] = data.s.n1_conv2_out[i];}//Get input
        #else
            for (i = 0; i < 12*6*6/2; i++)  {*(SENSOR_PORT) = data.u.n1_conv2_out[i];}//Get input
        #endif
        for (i = 0; i < 28/2; i++)   {*(SENSOR_PORT) = data.u.q_fc1_out[i];}
    } else if(mode&(MIL1)){ 
        #if NVACTIV
            if (high_low == 0){
                *(SENSOR_PORT) = MIL2;
                *(SENSOR_PORT) = high_low;
                for (i = 0; i < 12*3*3/2; i++)  {*(SENSOR_PORT) = data.u.q_input[i];}//Get input
            } else {
                *(SENSOR_PORT) = MIL0;
                *(SENSOR_PORT) = 0xFFFF;
            }
        #else
            if (high_low == 0){
                *(SENSOR_PORT) = MIL2;
                *(SENSOR_PORT) = high_low;
                for (i = 0; i < 12*3*3/2; i++)  {*(SENSOR_PORT) = data.u.q_input[i];}//Get input
            } else {
                *(SENSOR_PORT) = MIL3;
                *(SENSOR_PORT) = high_low;
                for (i = 0; i < 12*6*6/2; i++)  {*(SENSOR_PORT) = data.u.n1_conv2_out[i];}//Get input
            }
        #endif
    } else if(mode&(MIL3)){ 
        *(SENSOR_PORT) = MIL4;
        *(SENSOR_PORT) = high_low;
        for (i = 0; i < 2*2*24/2; i++)   {*(SENSOR_PORT) = data.u.n2_conv2_out[i];}
        *(SENSOR_PORT) = MIL5; //Send in parallel
        *(SENSOR_PORT) = high_low;
        for (i = 0; i < 2*2*24/2; i++)   {*(SENSOR_PORT) = data.u.n2_conv2_out[i];}
        *(SENSOR_PORT) = MIL6; //Send in parallel
        *(SENSOR_PORT) = high_low;
        for (i = 0; i < 2*2*24/2; i++)   {*(SENSOR_PORT) = data.u.n2_conv2_out[i];}
        *(SENSOR_PORT) = MIL7; //Send in parallel
        *(SENSOR_PORT) = high_low;
        for (i = 0; i < 2*2*24/2; i++)   {*(SENSOR_PORT) = data.u.n2_conv2_out[i];}
    } else if(mode&(MIL4)){ 
        *(SENSOR_PORT) = MIL7;
        for (i = 0; i < 42/2; i++)      {*(SENSOR_PORT) = data.u.n2_fc1_out[i];}
    } else if(mode&(MIL5)){ 
        *(SENSOR_PORT) = MIL7;
        for (i = 42/2; i < 84/2; i++)   {*(SENSOR_PORT) = data.u.n2_fc1_out[i];}
    } else if(mode&(MIL6)){ 
        *(SENSOR_PORT) = MIL7;
        for (i = 84/2; i < 126/2; i++)   {*(SENSOR_PORT) = data.u.n2_fc1_out[i];}
    }
    return;
}


