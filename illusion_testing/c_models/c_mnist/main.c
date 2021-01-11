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

//Print Mode 0: No print
//Print Mode 1: Target print (Chip 0 only)
//Print Mode 2: Small Illusion print (Chips 0-3)
//Print Mode 3: Large Illusion print (Chips 0-7)

void print_array(int8_t *array, int a, int b, int c){
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
        printf("%04x\n", join.o);
    }
    //printf("\n");
    }
    //printf("\n");
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

int classify(int z, int print_mode) {
    int8_t conv1_out[12*12*6] = {0};
    int8_t conv2_out[10*10*6] = {0};
    int8_t conv3_out[4*4*8] = {0};
    int8_t conv4_out[2*2*24] = {0};
    int16_t fc0_out[96] = {0};
    int16_t classes[10] = {0};
    int i,j,k,l;
    
    if (print_mode > 0) { 
        printf("Chip 0 Input\n");
        printf("%04x\n",0); //Chip 0 reiceves input in all partitions
        print_array(input_data + z*28*28, 1, 28, 28);
    }
    // Layer 1
    for (i = 0; i < 6; i++) {
        conv2D_1filter(input_data +z*28*28, layer1_H + (i*5*5), layer1_B[i], 1, 28, 5, 1, conv1_out+(i*12*12));
    }

    //print_array(conv1_out, 6, 12, 12);
    // Layer 2
    for (i = 0; i < 6; i++) {
        conv2D_1filter(conv1_out, layer2_H + (i*3*3*6), layer2_B[i], 6, 12, 3, 0, conv2_out+(i*10*10));
    }
   
    if (print_mode == 3) {
        printf("Chip 1 Input\n");
        printf("%04x\n",1); //Chip 1 reiceves input in all partitions
        print_array(conv2_out, 6, 10, 10);
    }

    // Layer 3
    for (i = 0; i < 8; i++) {
        conv2D_1filter(conv2_out, layer3_H + (i*3*3*6), layer3_B[i], 6, 10, 3, 1, conv3_out+(i*4*4));
    }
    
    if (print_mode == 2) {
        printf("Chip 1 Input\n");
        printf("%04x\n",1); //Chip 1 reiceves input in all partitions
        print_array(conv3_out, 8, 4, 4);
        printf("Chip 2 Input\n");
        printf("%04x\n",2); //Chip 1 reiceves input in all partitions
        print_array(conv3_out, 8, 4, 4);
    } else if (print_mode == 3) {
        printf("Chip 2 Input\n");
        printf("%04x\n",2); //Chip 1 reiceves input in all partitions
        print_array(conv3_out, 8, 4, 4);
        printf("Chip 3 Input\n");
        printf("%04x\n",3); //Chip 1 reiceves input in all partitions
        print_array(conv3_out, 8, 4, 4);
        printf("Chip 4 Input\n");
        printf("%04x\n",4); //Chip 1 reiceves input in all partitions
        print_array(conv3_out, 8, 4, 4);
        printf("Chip 5 Input\n");
        printf("%04x\n",5); //Chip 1 reiceves input in all partitions
        print_array(conv3_out, 8, 4, 4);
    }
    
    // Layer 4
    for (i = 0; i < 6; i++) {
        conv2D_1filter(conv3_out, layer4_H + (i*3*3*8), layer4_B[i], 8, 4, 3, 0, conv4_out+(i*2*2));
    }
    for (i = 6; i < 12; i++) {
        conv2D_1filter(conv3_out, layer4_H + (i*3*3*8), layer4_B[i], 8, 4, 3, 0, conv4_out+(i*2*2));
    }
    for (i = 12; i < 18; i++) {
        conv2D_1filter(conv3_out, layer4_H + (i*3*3*8), layer4_B[i], 8, 4, 3, 0, conv4_out+(i*2*2));
    }
    for (i = 18; i < 24; i++) {
        conv2D_1filter(conv3_out, layer4_H + (i*3*3*8), layer4_B[i], 8, 4, 3, 0, conv4_out+(i*2*2));
    }
    
    //Flatten
    l = 0;
    for (i = 0; i < 2; i++) {
    for (j = 0; j < 2; j++) {
    for (k = 0; k < 24; k++) {
        fc0_out[l] = (int16_t) conv4_out[l];
        l++;
    }
    }
    } 
    
    // Layer 4
    //Fix H's into two parts
    int8_t layer5_HA[960/2];  
    int8_t layer5_HB[960/2]; 
    
    l = 0;
    for (i = 0; i < 10; i++) {
        for (j = 0; j < 48; j++) {
            layer5_HA[l] = layer5_H[i*96+j];
            layer5_HB[l] = layer5_H[i*96+j+48];
            l++;
        }
    }
    
    int32_t classes2[10] = {0};
    dense(fc0_out, layer5_HA, classes2, 48, 10);
    
    if (print_mode == 2) {
        printf("Chip 3 Input\n");
        printf("%04x\n",3); //Chip 1 reiceves input in all partitions
        print_array(conv4_out, 1, 1, 96);
    } else if (print_mode == 3) {
        printf("Chip 6 Input\n");
        printf("%04x\n",6); //Chip 1 reiceves input in all partitions
        print_array(conv4_out, 1, 1, 48);
        printf("Chip 7 Input\n");
        printf("%04x\n",7); //Chip 1 reiceves input in all partitions
        print_array(conv4_out+48, 1, 1, 48);
        print_array32(classes2, 1, 1, 10); //Recieves partial sum
    }
   
    dense(fc0_out+48, layer5_HB, classes2, 48, 10);
    add_bias(layer5_B, classes2, classes, 10);
    //Depending upon how last layer is split one of these can be used: target is first line
    //dense2(fc0_out, layer5_H, layer5_B, classes, 96, 10);
    //Split outputs
    //dense2(fc0_out, layer5_H, layer5_B, classes, 96, 5);
    //dense2(fc0_out, layer5_H+96*5, layer5_B+5, classes+5, 96, 5);
    
    //print_array16(classes, 1, 1, 10);
    //print_array16(classes, 1, 1, 10);
    int class_out = 0;
    int16_t prob = INT16_MIN;
    for (i = 0; i < 10; i++){
        if (prob < classes[i]) {
            class_out = i;
            prob = classes[i];
        }
    }
    if (print_mode >= 1) {
        printf("Final Output\n");
        printf("%04x\n",8); //Chip 1 reiceves input in all partitions
        printf("%04x\n",class_out); //Chip 1 reiceves input in all partitions
    }
    return class_out;
}

int main(int argc, char *argv[]) {
    int correct = 0;
    int total = 0;
    int i;
    int class;
    int gt;
    int j = atoi(argv[1]);
    int p = atoi(argv[2]);
    for (i = 0; i< j ; i++) {
        gt = ground_truth[i];
        class = classify(i,p);
        if (class == gt) correct++;
        total++;

        if (p==0){printf("%d got %d\n", gt, class);}
    }
    if (p==0){ printf("Accuracy: %f\n", (double) correct / (total*1.0) * 100);}
    
}

