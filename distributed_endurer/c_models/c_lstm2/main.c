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

#include "model_chunked_LSTM.c"
#include "data.c"
#include "activation_mask.c"

void print_array(int8_t *array, int a){
    int i,l;
    l = 0;
    union Join {struct {int8_t l; int8_t h;} b; uint16_t o;} join;
    for (i = 0; i < a/2; i++) {
        join.b.l = array[l]; 
        l++;
        join.b.h = array[l];
        l++;
        printf("%04x\n", join.o);
    }
}

void print_array_nvm(int8_t *array, int chip, int start_address, int a) {
    int i,j,k,l;
    union Join {struct {int8_t l; int8_t h;} b; uint16_t o;} join;
    l = 0;
    for (i = 0; i < a/2; i++) {
        join.b.l = array[l];
        l++;
        join.b.h = array[l];
        l++;
        printf("%d,%04x,%04x\n", chip, start_address + l/2, join.o);
    }
}

void print_array16(int16_t *array, int a){
    int i,l;
    l = 0;
    for (i = 0; i < a; i++) {
        printf("%04x\n", (uint16_t) array[l]);
        l++;
    }

}

void print_array32(int32_t *array, int a){
    int i,l;
    l = 0;
    union Join {struct {uint16_t l; uint16_t h;} b; int32_t o;} join;
    for (i = 0; i < a; i++) {
        join.o = array[l];
        printf("%04x\n", (uint16_t) join.b.l);
        printf("%04x\n", (uint16_t) join.b.h);
        l++;
    }

}

int classify(int z, int print_mode) {
    int i,j,k,l;
    int8_t h_hist[(SEQLEN+1)*HDIM] = {0};
    int16_t c_hist[(SEQLEN+1)*HDIM] = {0};
    int16_t classes[ODIM] = {0};
    int32_t classes_p[ODIM] = {0};
    int class_out = 0;
    int16_t prob = INT16_MIN;
    
    //for( i = 0; i < SEQLEN; i+=1) {
    //    lstm(input + z*SEQLEN*IDIM + i*IDIM, h_hist + i*HDIM, c_hist + i*HDIM, lstm_i_H, lstm_h_H, lstm_B, HDIM, h_hist + (i+1)*HDIM , c_hist + (i+1)*HDIM);
    //}   
    for( i = 0; i < SEQLEN; i+=1) {
        int last = (i == SEQLEN-1);
        if (((i == 0) || (i == SEQLEN-1)) && (print_mode == 1)){
            printf("Chip 0 Input\n");
            printf("%04x\n", 0);
            print_array(input + z*SEQLEN*IDIM + i*IDIM,IDIM);
            print_array(h_hist + i*HDIM, HDIM);
            print_array16(c_hist + i*HDIM, HDIM);
        }
        if (((i == 0) || (i == SEQLEN-1)) && (print_mode == 2)){
            for (j = 0; j < 4; j++) {
                printf("Chip %d Input\n", j);
                printf("%04x\n", j);
                print_array(input + z*SEQLEN*IDIM + i*IDIM, IDIM);
                print_array(h_hist + i*HDIM, HDIM);
                print_array16(c_hist + i*HDIM + j*HDIM/4, HDIM/4);
            }
        }
        if (((i == 0) || (i == SEQLEN-1)) && (print_mode == 3)){
            for (j = 0; j < 8; j++) {
                printf("Chip %d Input\n", j);
                printf("%04x\n", j);
                print_array(input + z*SEQLEN*IDIM + i*IDIM, IDIM);
                print_array(h_hist + i*HDIM, HDIM);
                print_array16(c_hist + i*HDIM + j*HDIM/8, HDIM/8);
            }
        }

        if (print_mode < 0) {
            k = - 1 - print_mode;
            //print_array_nvm(h_hist + i*HDIM, k, 2047-HDIM*3/2, HDIM);
            print_array_nvm((int8_t*) (c_hist + i*HDIM + k*HDIM/8), k, 2047-HDIM/4, HDIM/4);
        }



        lstm(input + z*SEQLEN*IDIM + i*IDIM, h_hist + i*HDIM, c_hist + i*HDIM + 0*HDIM/8, lstm_i_H_0, lstm_h_H_0, lstm_B_0, HDIM/8, h_hist + (i+1)*HDIM + 0*HDIM/8, c_hist + (i+1)*HDIM + 0*HDIM/8);
       
        int e_; 
        for (e_=0; e_<HDIM/8;e_++) {
            int16_t temp = c_hist[(i+1)*HDIM+0*HDIM/8 + e_];
            temp |= chip0_forced_zeros[e_]; // stuck at 1
            temp &= chip0_forced_ones[e_];  // stuck at 0
            c_hist[(i+1)*HDIM+0*HDIM/8 + e_] = temp;

        }
 
        if(last){
            dense(h_hist + SEQLEN*HDIM + 0*HDIM/8, fc_H_0, classes_p, HDIM/8, ODIM);
            if (print_mode >= 3){
                printf("Chip %d Input\n",1/(4-print_mode));
                print_array32(classes_p, ODIM);
            }
        }
        lstm(input + z*SEQLEN*IDIM + i*IDIM, h_hist + i*HDIM, c_hist + i*HDIM + 1*HDIM/8, lstm_i_H_1, lstm_h_H_1, lstm_B_1, HDIM/8, h_hist + (i+1)*HDIM + 1*HDIM/8, c_hist + (i+1)*HDIM + 1*HDIM/8);
        
        for (e_=0; e_<HDIM/8;e_++) {
            int16_t temp = c_hist[(i+1)*HDIM+1*HDIM/8 + e_];
            temp |= chip1_forced_zeros[e_]; // stuck at 1
            temp &= chip1_forced_ones[e_];  // stuck at 0
            c_hist[(i+1)*HDIM+1*HDIM/8 + e_] = temp;

        } 
        

        
        if(last){
            dense(h_hist + SEQLEN*HDIM + 1*HDIM/8, fc_H_1, classes_p, HDIM/8, ODIM);
            if (print_mode >= 2){
                printf("Chip %d Input\n",2/(4-print_mode));
                print_array32(classes_p, ODIM);
            }
        }
        lstm(input + z*SEQLEN*IDIM + i*IDIM, h_hist + i*HDIM, c_hist + i*HDIM + 2*HDIM/8, lstm_i_H_2, lstm_h_H_2, lstm_B_2, HDIM/8, h_hist + (i+1)*HDIM + 2*HDIM/8, c_hist + (i+1)*HDIM + 2*HDIM/8);
        
        for (e_=0; e_<HDIM/8;e_++) {
            int16_t temp = c_hist[(i+1)*HDIM+2*HDIM/8 + e_];
            temp |= chip2_forced_zeros[e_]; // stuck at 1
            temp &= chip2_forced_ones[e_];  // stuck at 0
            c_hist[(i+1)*HDIM+2*HDIM/8 + e_] = temp;

        } 
        
        if(last){
            dense(h_hist + SEQLEN*HDIM + 2*HDIM/8, fc_H_2, classes_p, HDIM/8, ODIM);
            if (print_mode >= 3){
                printf("Chip %d Input\n",3/(4-print_mode));
                printf("Blah\n",3/(4-print_mode));
                print_array32(classes_p, ODIM);
            }
        }
        lstm(input + z*SEQLEN*IDIM + i*IDIM, h_hist + i*HDIM, c_hist + i*HDIM + 3*HDIM/8, lstm_i_H_3, lstm_h_H_3, lstm_B_3, HDIM/8, h_hist + (i+1)*HDIM + 3*HDIM/8, c_hist + (i+1)*HDIM + 3*HDIM/8);
        
        for (e_=0; e_<HDIM/8;e_++) {
            int16_t temp = c_hist[(i+1)*HDIM+3*HDIM/8 + e_];
            temp |= chip3_forced_zeros[e_]; // stuck at 1
            temp &= chip3_forced_ones[e_];  // stuck at 0
            c_hist[(i+1)*HDIM+3*HDIM/8 + e_] = temp;

        } 
        
        if(last){
            dense(h_hist + SEQLEN*HDIM + 3*HDIM/8, fc_H_3, classes_p, HDIM/8, ODIM);
            if (print_mode >= 2){
                printf("Chip %d Input\n",4/(4-print_mode));
                print_array32(classes_p, ODIM);
            }
        }
        //printf("CHIP5\n");
        lstm(input + z*SEQLEN*IDIM + i*IDIM, h_hist + i*HDIM, c_hist + i*HDIM + 4*HDIM/8, lstm_i_H_4, lstm_h_H_4, lstm_B_4, HDIM/8, h_hist + (i+1)*HDIM + 4*HDIM/8, c_hist + (i+1)*HDIM + 4*HDIM/8);
        
        for (e_=0; e_<HDIM/8;e_++) {
            int16_t temp = c_hist[(i+1)*HDIM+4*HDIM/8 + e_];
            temp |= chip4_forced_zeros[e_]; // stuck at 1
            temp &= chip4_forced_ones[e_];  // stuck at 0
            c_hist[(i+1)*HDIM+4*HDIM/8 + e_] = temp;

        } 
        
        if(last){
            dense(h_hist + SEQLEN*HDIM + 4*HDIM/8, fc_H_4, classes_p, HDIM/8, ODIM);
            if (print_mode >= 3){
                printf("Chip %d Input\n",5/(4-print_mode));
                print_array32(classes_p, ODIM);
            }
        }
        lstm(input + z*SEQLEN*IDIM + i*IDIM, h_hist + i*HDIM, c_hist + i*HDIM + 5*HDIM/8, lstm_i_H_5, lstm_h_H_5, lstm_B_5, HDIM/8, h_hist + (i+1)*HDIM + 5*HDIM/8, c_hist + (i+1)*HDIM + 5*HDIM/8);
        
        for (e_=0; e_<HDIM/8;e_++) {
            int16_t temp = c_hist[(i+1)*HDIM+5*HDIM/8 + e_];
            temp |= chip5_forced_zeros[e_]; // stuck at 1
            temp &= chip5_forced_ones[e_];  // stuck at 0
            c_hist[(i+1)*HDIM+5*HDIM/8 + e_] = temp;

        } 


        
        if(last){
            dense(h_hist + SEQLEN*HDIM + 5*HDIM/8, fc_H_5, classes_p, HDIM/8, ODIM);
            if (print_mode >= 2){
                printf("Chip %d Input\n",6/(4-print_mode));
                print_array32(classes_p, ODIM);
            }
        }
        lstm(input + z*SEQLEN*IDIM + i*IDIM, h_hist + i*HDIM, c_hist + i*HDIM + 6*HDIM/8, lstm_i_H_6, lstm_h_H_6, lstm_B_6, HDIM/8, h_hist + (i+1)*HDIM + 6*HDIM/8, c_hist + (i+1)*HDIM + 6*HDIM/8);
        
        for (e_=0; e_<HDIM/8;e_++) {
            int16_t temp = c_hist[(i+1)*HDIM+6*HDIM/8 + e_];
            temp |= chip6_forced_zeros[e_]; // stuck at 1
            temp &= chip6_forced_ones[e_];  // stuck at 0
            c_hist[(i+1)*HDIM+6*HDIM/8 + e_] = temp;

        } 

        if(last){
            dense(h_hist + SEQLEN*HDIM + 6*HDIM/8, fc_H_6, classes_p, HDIM/8, ODIM);
            if (print_mode >= 3){
                printf("Chip %d Input\n",7/(4-print_mode));
                print_array32(classes_p, ODIM);
            }
        }
        lstm(input + z*SEQLEN*IDIM + i*IDIM, h_hist + i*HDIM, c_hist + i*HDIM + 7*HDIM/8, lstm_i_H_7, lstm_h_H_7, lstm_B_7, HDIM/8, h_hist + (i+1)*HDIM + 7*HDIM/8, c_hist + (i+1)*HDIM + 7*HDIM/8);
        
        for (e_=0; e_<HDIM/8;e_++) {
            int16_t temp = c_hist[(i+1)*HDIM+7*HDIM/8 + e_];
            temp |= chip7_forced_zeros[e_]; // stuck at 1
            temp &= chip7_forced_ones[e_];  // stuck at 0
            c_hist[(i+1)*HDIM+7*HDIM/8 + e_] = temp;

        } 
       
        if(last){
            dense(h_hist + SEQLEN*HDIM + 7*HDIM/8, fc_H_7, classes_p, HDIM/8, ODIM);
            add_bias(fc_B, classes_p, classes, ODIM); 
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
            print_array(h_hist + (i+1)*HDIM, HDIM);
            print_array16(c_hist + (i+1)*HDIM, HDIM);
        }
        if (((i == 0) || (i == SEQLEN-1)) && (print_mode == 2)){
            for (j = 0; j < 4; j++) {
                printf("Chip %d Output\n", j);
                printf("%04x\n", j+1);
                print_array(h_hist + (i+1)*HDIM + j*HDIM/4, HDIM/4);
                print_array16(c_hist + (i+1)*HDIM + j*HDIM/4, HDIM/4);
            }
        }
        if (((i == 0) || (i == SEQLEN-1)) && (print_mode == 3)){
            for (j = 0; j < 8; j++) {
                printf("Chip %d Output\n", j);
                printf("%04x\n", j+1);
                print_array(h_hist + (i+1)*HDIM + j*HDIM/8, HDIM/8);
                print_array16(c_hist + (i+1)*HDIM + j*HDIM/8, HDIM/8);
            }
        }
    }    

    
    for (j = 0; j < 11; j++){
        if (prob < classes[j]) {
            class_out = j;
            prob = classes[j];
        }
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
    for (i = 0; i < j; i++) {
        gt = ground_truth[i];
        class = classify(i,p);
        if (class == gt) correct++;
        total++;
        if (p>0)
            printf("%d got %d\n", gt, class);
    }
    if (p>=0) printf("%f\n", (double) correct / (total*1.0) * 100);
}

