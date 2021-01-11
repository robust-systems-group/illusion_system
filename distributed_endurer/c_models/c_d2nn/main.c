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

#include <stdint.h>
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.c"
#include "data.c"
#include "activation_mask.c"

//Print Mode 0: No print
//Print Mode 1: Target print (Chip 0 only)
//Print Mode 2: Small Illusion print (Chips 0-3)
//Print Mode 3: Large Illusion print (Chips 0-7)
//Print Mode -1: NVM writes

void print_array(int8_t *array, int a, int b, int c){
    int i,j,k,l;
    union Join {struct {int8_t l; int8_t h;} b; uint16_t o;} join;
    l = 0;
    for (i = 0; i < a/2; i++) {
    for (j = 0; j < b; j++) {
    for (k = 0; k < c; k++) {
        join.b.l = array[l];
        l++;
        join.b.h = array[l];
        l++;
        printf("%04x\n", join.o);
    }
    //printf("\n");
    }
    //printf("\n");
    }
}

void print_array_nvm(int8_t *array, int chip, int start_address, int a, int b, int c) {
    int i,j,k,l;
    union Join {struct {int8_t l; int8_t h;} b; uint16_t o;} join;
    l = 0;
    for (i = 0; i < a; i++) {
    for (j = 0; j < b; j++) {
    for (k = 0; k < c/2; k++) {
        join.b.l = array[l];
        l++;
        join.b.h = array[l];
        l++;
        printf("%d,%04x,%04x\n", chip, start_address + l/2, join.o);
    }
    }
    }
}

void print_array16(int16_t *array, int a, int b, int c){
    int i,j,k,l;
    l = 0;
    for (i = 0; i < a; i++) {
    for (j = 0; j < b; j++) {
    for (k = 0; k < c; k++) {
        printf("%04x\n", (uint16_t) array[l]);
        l++;
    }
    //printf("\n");
    }
    //printf("\n");
    }

}

void print_array32(int32_t *array, int a, int b, int c){
    int i,j,k,l;
    l = 0;
    union Join {struct {uint16_t l; uint16_t h;} b; int32_t o;} join;
    for (i = 0; i < a; i++) {
    for (j = 0; j < b; j++) {
    for (k = 0; k < c; k++) {
        join.o = array[l];
        printf("%04x\n", (uint16_t) join.b.l);
        printf("%04x\n", (uint16_t) join.b.h);
        l++;
    }
    //printf("\n");
    }
    //printf("\n");
    }

}
int highs=0;

int8_t n1_conv2_out_raw[12*6*6];

int classify(int z, int print_mode) {
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
            int32_t n2_fc1_part[128];
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
            uint16_t n2_fc1_part[128*2];
            uint16_t n2_fc1_out[128/2];
            uint16_t n2_fc2_out[24/2];
            uint16_t n3_fc1_out[32/2];
            uint16_t classes[10/2];
        } u;
    } DTot;
    
    
    DTot data = {0};
    
    int high_low;
    int max_pred;
    int pred_prob;
    int class_out;
    int i,j,k,l;
    
    if (print_mode > 0) { 
        printf("Chip 0 Input\n");
        printf("%04x\n",0); //Chip 0 reiceves input in all partitions
        print_array(input_data + z*28*28, 28*28/2, 1, 1);
    }
    
    for (i = 0; i < 6; i++) {
        conv2D_1filter(input_data + z*28*28, N1_layer1_H + (i*5*5), N1_layer1_B[i], 1, 28, 5, 2, data.s.n1_conv1_out + (i*6*6));
    }
    for (i = 0; i < 12; i++) {
        conv2D_1filter_stuck(data.s.n1_conv1_out, N1_layer2_H + (i*1*1*6), N1_layer2_B[i], 6, 6, 1, 0, data.s.n1_conv2_out + (i*6*6),
                            n1_conv2_forced_ones + (i*6*6), n1_conv2_forced_zeros + (i*6*6), n1_conv2_out_raw + (i*6*6));
    }
    for (i = 0; i < 12; i++) {
        max_pool(data.s.n1_conv2_out +(i*6*6), 6, 1, data.s.q_input + (i*3*3));
    }
    
    int8_t q_input[12*3*3];
    for (i = 0; i <12*3*3; i++) {
        q_input[i] = (int8_t) data.s.q_input[i];
    }

    dense_bias(data.s.q_input, Q_layer1_H, Q_layer1_B, data.s.q_fc1_out, 108, 28);

    // D2NN stream writes     
    if (print_mode == -1) {
        print_array_nvm(n1_conv2_out_raw, 0, 2047-6*6*6, 12, 6, 6);
    }

    // process stuff saved in NVM only for high path
    for (i=0; i<12*6*6; i++) {
        uint8_t temp = n1_conv2_out_raw[i]; // grab pre relu value, process faults
        temp |= (uint8_t) n1_conv2_forced_zeros[i]; // stuck at 1
        temp &= (uint8_t) n1_conv2_forced_ones[i];  // stuck at 0
        if ((int8_t)temp<0) temp = 0; // apply relu
        temp |= (uint8_t) n1_conv2_forced_zeros[i]; // stuck at 1 applied again post relu
        n1_conv2_out_raw[i] = (int8_t) temp;
    }       
 
    if (print_mode == 3) {
        printf("Chip 1 Input\n");
        printf("%04x\n",1); //Chip 1 reiceves input in all partitions
        print_array(q_input, 12, 3, 3);
        print_array(data.s.q_fc1_out, 28, 1, 1);
        printf("Chip 1 Input Full\n");
        print_array(data.s.n1_conv2_out, 12, 6, 6);
        print_array(data.s.q_fc1_out, 28, 1, 1);
    }

    dense_bias(data.s.q_input, Q_layer1_H+28*108, Q_layer1_B+28, data.s.q_fc1_out+28, 108, 36);
    dense_bias(data.s.q_fc1_out, Q_layer2_H, Q_layer2_B, data.s.q_fc2_out, 64, 2);

    if ( data.s.q_fc2_out[1] < data.s.q_fc2_out[0] ) {
        high_low = 1;
        highs+=1;
        if (print_mode == 2) {
            printf("Chip 1 Input High Full\n");
            printf("%04x\n",1); //Chip 1 reiceves input in all partitions
            print_array(data.s.n1_conv2_out, 12, 6, 6);
        } else if (print_mode == 3) {
            printf("Chip 3 Input High Full\n");
            printf("%04x\n",3); //Chip 1 reiceves input in all partitions
            print_array(data.s.n1_conv2_out, 12, 6, 6);
        }
    } else { 
        high_low = 0;
        if (print_mode == 2) {
            printf("Chip 1 Input Low Maxpool\n");
            printf("%04x\n",1); //Chip 1 reiceves input in all partitions
            print_array(q_input, 12, 3, 3);
        } else if (print_mode == 3) {
            printf("Chip 2 Input Low Maxpool\n");
            printf("%04x\n",2); //Chip 1 reiceves input in all partitions
            print_array(q_input, 12, 3, 3 );
        }
    }
    
    if (!high_low) { 
        dense_bias(data.s.q_input, N3_layer1_H, N3_layer1_B, data.s.n3_fc1_out, 3*3*12, 32);
        dense_bias(data.s.n3_fc1_out, N3_layer2_H, N3_layer2_B, data.s.classes, 32, 10);
        pred_prob = INT16_MIN;
        max_pred = 0;
        for (i = 0; i < 10; i++ ) {
            if (data.s.classes[i] > pred_prob){
                pred_prob = data.s.classes[i];
                max_pred = i;
            }
        }
    }
    
    if (high_low) { 
        for (i = 0; i < 12; i++) {
            conv2D_1filter(n1_conv2_out_raw, N2_layer1_H + (i*3*3*12), N2_layer1_B[i], 12, 6, 3, 0, data.s.n2_conv1_out + (i*4*4));
        }
        for (i = 0; i < 24; i++) {
            conv2D_1filter(data.s.n2_conv1_out, N2_layer2_H + (i*3*3*12), N2_layer2_B[i], 12, 4, 3, 0, data.s.n2_conv2_out + (i*2*2));
        }
        if (print_mode == 3) {
            printf("Chip 4 Input\n");
            printf("%04x\n",4); //Chip 1 reiceves input in all partitions
            print_array(data.s.n2_conv2_out, 96, 1, 1);
            //print_array32(data.s.n2_fc1_part, 128, 1, 1);
            printf("Chip 5 Input\n");
            printf("%04x\n",5); //Chip 1 reiceves input in all partitions
            print_array(data.s.n2_conv2_out, 96, 1, 1);
            printf("Chip 6 Input\n");
            printf("%04x\n",6); //Chip 1 reiceves input in all partitions
            print_array(data.s.n2_conv2_out, 96, 1, 1);
            printf("Chip 7 Input\n");
            printf("%04x\n",7); //Chip 1 reiceves input in all partitions
            print_array(data.s.n2_conv2_out, 96, 1, 1);
        }
    }
    
    if (high_low) { 
        dense_bias(data.s.n2_conv2_out, N2_layer3_H, N2_layer3_B, data.s.n2_fc1_out, 96, 42);
    }
    
    if (high_low) { 
        dense_bias(data.s.n2_conv2_out, N2_layer3_H+42*96, N2_layer3_B+42, data.s.n2_fc1_out+42, 96, 42);
    }
    
    if (high_low) { 
        dense_bias(data.s.n2_conv2_out, N2_layer3_H+42*96*2, N2_layer3_B+42*2, data.s.n2_fc1_out+42*2, 96, 42);
    }
    
    if (high_low) { 
        if (print_mode == 3) {
            printf("Chip 4 Output\n");
            printf("%04x\n",7); //Chip 1 reiceves input in all partitions
            print_array(data.s.n2_fc1_out, 42, 1, 1);
            //print_array32(data.s.n2_fc1_part, 128, 1, 1);
            printf("Chip 5 Output\n");
            printf("%04x\n",7); //Chip 1 reiceves input in all partitions
            print_array(data.s.n2_fc1_out+42, 42, 1, 1);
            printf("Chip 6 Output\n");
            printf("%04x\n",7); //Chip 1 reiceves input in all partitions
            print_array(data.s.n2_fc1_out+84, 42, 1, 1);
        }
        dense_bias(data.s.n2_conv2_out, N2_layer3_H+42*96*3, N2_layer3_B+42*3, data.s.n2_fc1_out+42*3, 96, 2);
    }
    
    if (high_low) { 
        dense_bias(data.s.n2_fc1_out, N2_layer4_H, N2_layer4_B, data.s.n2_fc2_out, 128, 24);
        dense_bias(data.s.n2_fc2_out, N2_layer5_H, N2_layer5_B, data.s.classes, 24, 10);
        pred_prob = INT16_MIN;
        max_pred = 0;
        for (i = 0; i < 10; i++ ) {
            if (data.s.classes[i] > pred_prob){
                pred_prob = data.s.classes[i];
                max_pred = i;
            }
        }
    }
    
    if (print_mode >= 1) {
        printf("Final Output\n");
        printf("%04x\n", 8); 
        printf("%04x\n", max_pred ); 
    }
    return max_pred;
}

int main(int argc, char *argv[]) {
    int correct = 0;
    int total = 0;
    int i;
    int class;
    int gt;
    int j = atoi(argv[1]);
    int k = atoi(argv[2]);
    int p = atoi(argv[3]);
    for (i = j ; i < (k + 1); i++) {
        gt = ground_truth[i];
        class = classify(i,p);
        if (class == gt) correct++;
        total++;

        //if (p==0){printf("%d got %d\n", gt, class);}
    }
    if (p==0){ printf("%f", (double) correct / (total*1.0) * 100);}
    if (p==0){ printf(",%f\n", (double) highs/ (total*1.0) * 100);}
    
}

