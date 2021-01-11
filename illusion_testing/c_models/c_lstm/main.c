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
#include "model_chunked_LSTM.c"
#include "data.c"

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
    int i,j,k,l;
    int16_t h_hist[(SEQLEN+1)*HDIM] = {0};
    int32_t c_hist[(SEQLEN+1)*HDIM] = {0};
    int32_t classes[ODIM] = {0};
    int32_t classes_p[ODIM] = {0};
    int class_out = 0;
    int32_t prob = INT32_MIN;
    for( i = 0; i < SEQLEN; i+=1) {
        int last = (i == SEQLEN-1);//lstm(input + z*SEQLEN*IDIM + i*IDIM, h_hist + i*HDIM, c_hist + i*HDIM, lstm_i_H, lstm_h_H, lstm_B, HDIM, h_hist + (i+1)*HDIM , c_hist + (i+1)*HDIM);
        if (((i == 0) || (i == SEQLEN-1)) && (print_mode == 1)){
            printf("Chip 0 Input\n");
            printf("%04x\n", 0);
            print_array16(input + z*SEQLEN*IDIM + i*IDIM, 1, 1, IDIM);
            print_array16(h_hist + i*HDIM, 1, 1, HDIM);
            print_array32(c_hist + i*HDIM, 1, 1, HDIM);
        }
        if (((i == 0) || (i == SEQLEN-1)) && (print_mode == 2)){
            for (j = 0; j < 4; j++) {
                printf("Chip %d Input\n", j);
                printf("%04x\n", j);
                print_array16(input + z*SEQLEN*IDIM + i*IDIM, 1, 1, IDIM);
                print_array16(h_hist + i*HDIM, 1, 1, HDIM);
                print_array32(c_hist + i*HDIM + j*HDIM/4, 1, 1, HDIM/4);
            }
        }
        if (((i == 0) || (i == SEQLEN-1)) && (print_mode == 3)){
            for (j = 0; j < 8; j++) {
                printf("Chip %d Input\n", j);
                printf("%04x\n", j);
                print_array16(input + z*SEQLEN*IDIM + i*IDIM, 1, 1, IDIM);
                print_array16(h_hist + i*HDIM, 1, 1, HDIM);
                print_array32(c_hist + i*HDIM + j*HDIM/4, 1, 1, HDIM/8);
            }
        }
        lstm(input + z*SEQLEN*IDIM + i*IDIM, h_hist + i*HDIM, c_hist + i*HDIM + 0*7, layerlstm_i_H_0, layerlstm_h_H_0, layerlstm_B_0, 7, h_hist + (i+1)*HDIM + 0*7, c_hist + (i+1)*HDIM + 0*7);
        if(last){
            dense(h_hist + SEQLEN*HDIM + 0*7, layerfc_H_0, classes_p, 7, ODIM);
            if (print_mode >= 3){
                printf("Chip %d Input\n",1/(4-print_mode));
                print_array32(classes_p, 1, 1, ODIM);
            }
        }
        
        lstm(input + z*SEQLEN*IDIM + i*IDIM, h_hist + i*HDIM, c_hist + i*HDIM + 1*7, layerlstm_i_H_1, layerlstm_h_H_1, layerlstm_B_1, 7, h_hist + (i+1)*HDIM + 1*7, c_hist + (i+1)*HDIM + 1*7);
        if(last){
            dense(h_hist + SEQLEN*HDIM + 1*7, layerfc_H_1, classes_p, 7, ODIM);
            if (print_mode >= 2){
                printf("Chip %d Input\n",2/(4-print_mode));
                print_array32(classes_p, 1, 1, ODIM);
            }
        }
        
        lstm(input + z*SEQLEN*IDIM + i*IDIM, h_hist + i*HDIM, c_hist + i*HDIM + 2*7, layerlstm_i_H_2, layerlstm_h_H_2, layerlstm_B_2, 7, h_hist + (i+1)*HDIM + 2*7, c_hist + (i+1)*HDIM + 2*7);
        if(last){
            dense(h_hist + SEQLEN*HDIM + 2*7, layerfc_H_2, classes_p, 7, ODIM);
            if (print_mode >= 3){
                printf("Chip %d Input\n",3/(4-print_mode));
                print_array32(classes_p, 1, 1, ODIM);
            }
        }
        
        lstm(input + z*SEQLEN*IDIM + i*IDIM, h_hist + i*HDIM, c_hist + i*HDIM + 3*7, layerlstm_i_H_3, layerlstm_h_H_3, layerlstm_B_3, 7, h_hist + (i+1)*HDIM + 3*7, c_hist + (i+1)*HDIM + 3*7);
        if(last){
            dense(h_hist + SEQLEN*HDIM + 3*7, layerfc_H_3, classes_p, 7, ODIM);
            if (print_mode >= 2){
                printf("Chip %d Input\n",4/(4-print_mode));
                print_array32(classes_p, 1, 1, ODIM);
            }
        }
        
        lstm(input + z*SEQLEN*IDIM + i*IDIM, h_hist + i*HDIM, c_hist + i*HDIM + 4*7, layerlstm_i_H_4, layerlstm_h_H_4, layerlstm_B_4, 7, h_hist + (i+1)*HDIM + 4*7, c_hist + (i+1)*HDIM + 4*7);
        if(last){
            dense(h_hist + SEQLEN*HDIM + 4*7, layerfc_H_4, classes_p, 7, ODIM);
            if (print_mode >= 3){
                printf("Chip %d Input\n",5/(4-print_mode));
                print_array32(classes_p, 1, 1, ODIM);
            }
        }
        
        lstm(input + z*SEQLEN*IDIM + i*IDIM, h_hist + i*HDIM, c_hist + i*HDIM + 5*7, layerlstm_i_H_5, layerlstm_h_H_5, layerlstm_B_5, 7, h_hist + (i+1)*HDIM + 5*7, c_hist + (i+1)*HDIM + 5*7);
        if(last){
            dense(h_hist + SEQLEN*HDIM + 5*7, layerfc_H_5, classes_p, 7, ODIM);
            if (print_mode >= 2){
                printf("Chip %d Input\n",6/(4-print_mode));
                print_array32(classes_p, 1, 1, ODIM);
            }
        }
        
        lstm(input + z*SEQLEN*IDIM + i*IDIM, h_hist + i*HDIM, c_hist + i*HDIM + 6*7, layerlstm_i_H_6, layerlstm_h_H_6, layerlstm_B_6, 7, h_hist + (i+1)*HDIM + 6*7, c_hist + (i+1)*HDIM + 6*7);
        if(last){
            dense(h_hist + SEQLEN*HDIM + 6*7, layerfc_H_6, classes_p, 7, ODIM);
            if (print_mode >= 3){
                printf("Chip %d Input\n",7/(4-print_mode));
                print_array32(classes_p, 1, 1, ODIM);
            }
        }
        
        lstm(input + z*SEQLEN*IDIM + i*IDIM, h_hist + i*HDIM, c_hist + i*HDIM + 7*7, layerlstm_i_H_7, layerlstm_h_H_7, layerlstm_B_7, 7, h_hist + (i+1)*HDIM + 7*7, c_hist + (i+1)*HDIM + 7*7);
        if(last){
            dense(h_hist + SEQLEN*HDIM + 7*7, layerfc_H_7, classes_p, 7, ODIM);
            add_bias(layer_fc_B, classes_p, classes, ODIM); 
            for (j = 0; j < 11; j++){
                if (prob < classes[j]) {
                    class_out = j;
                    prob = classes[j];
                }
            }
            if (print_mode >= 1){
                printf("Output\n");
                printf("%04x\n",class_out);
            }
        }
        
        if (((i == 0) || (i == SEQLEN-1)) && (print_mode == 1)){
            printf("Chip 0 Output\n");
            printf("%04x\n", 1);
            print_array16(h_hist + (i+1)*HDIM, 1, 1, HDIM);
            print_array32(c_hist + (i+1)*HDIM, 1, 1, HDIM);
        }
        if (((i == 0) || (i == SEQLEN-1)) && (print_mode == 2)){
            for (j = 0; j < 4; j++) {
                printf("Chip %d Output\n", j);
                printf("%04x\n", j+1);
                print_array16(h_hist + (i+1)*HDIM + j*HDIM/4, 1, 1, HDIM/4);
                print_array32(c_hist + (i+1)*HDIM + j*HDIM/4, 1, 1, HDIM/4);
            }
        }
        if (((i == 0) || (i == SEQLEN-1)) && (print_mode == 3)){
            for (j = 0; j < 8; j++) {
                printf("Chip %d Output\n", j);
                printf("%04x\n", j+1);
                print_array16(h_hist + (i+1)*HDIM + j*HDIM/8, 1, 1, HDIM/8);
                print_array32(c_hist + (i+1)*HDIM + j*HDIM/8, 1, 1, HDIM/8);
            }
        }
        return 0;
    }    
    //dense2(h_hist + SEQLEN*HDIM, layer_fc_H, layer_fc_B, classes, HDIM, ODIM);
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
    for (i = 0; i < j; i++) {
        gt = ground_truth[i];
        class = classify(i,p);
        if (class == gt) correct++;
        total++;
        printf("%d got %d\n", gt, class);
    }
    printf("Accuracy: %f\n", (double) correct / (total*1.0) * 100);
}

