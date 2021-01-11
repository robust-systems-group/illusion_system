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
#include <stdio.h>

#define QOUT 11
#define QIN 4
#define SHIFT (QOUT - QIN)
#define IMAX (int8_t) 1<<QIN

#define SIGMAX (4 << QOUT)
#define SIGHALF (1 << (QOUT - 1))
#define TANHMAX (1 << QOUT)

inline int8_t qSig(int32_t a) {
    if (a > SIGMAX) {
        return IMAX;
    } else if (a < -SIGMAX) {
        return 0;
    } else {
        int32_t b = a >> 3;
        b = b + SIGHALF + (1<<6);
        return b >> SHIFT; 
    }
}
inline int8_t qTanh(int32_t a) {
    if (a > TANHMAX) {
        return IMAX;
    } else if (a < -TANHMAX) {
        return -IMAX;
    } else {
        
        if(a>0){int32_t b = a + (1<<6);}
        else{int32_t b = a - (1<<6);}
        return (int8_t) (a >> SHIFT);
    }
}

void lstm(int8_t *I, int8_t *H, int16_t *C,  const int8_t *W_IH, const int8_t *W_HH, const int8_t *B, int Z, int8_t *H_O, int16_t *C_O) {
    int v,z,t;
    int32_t gates[4*Z];
    int8_t igate[Z];
    int8_t fgate[Z];
    int8_t cgate[Z];
    int8_t ogate[Z];
    
    for (z = 0; z < 4*Z; z++){gates[z] = (B[z] << QIN);}
    dense(I, W_IH, gates, IDIM, Z*4); 
    dense(H, W_HH, gates, HDIM, Z*4); 
    //for (z = 0; z < Z; z++) {
    //    printf("%04x ",gates[z]);
    //}
    
    for (z = 0; z < Z; z++) {
        igate[z] = qSig(gates[z]);
        fgate[z] = qSig(gates[z+Z]);
        cgate[z] = qTanh(gates[z+2*Z]);
        ogate[z] = qSig(gates[z+3*Z]);
        C_O[z] = fgate[z]*(C[z] >> QIN) + igate[z]*cgate[z];
        H_O[z] = (int8_t) (ogate[z]*qTanh(C_O[z]<<3) >> QIN);
    }
    //printf("\n");
    //for (z = 0; z < Z; z++) {
    //    printf("%04x ",ogate[z]);
    //}
    //printf("\n");
    //for (z = 0; z < Z; z++) {
    //    printf("%04x ",C_O[z]);
    //}
    //printf("\n");
    //for (z = 0; z < Z; z++) {
    //    printf("%08x ",C_O[z]);
    //}
    //printf("\n");
    //for (z = 0; z < Z; z++) {
    //    printf("%04x ",H_O[z]);
    //}
    //printf("\n");
}


void dense(int8_t *A, const int8_t *H, int32_t *C, int V, int Z) {
    int v, z;
    int idx = 0;
    for (z = 0; z < Z; z++){
        for (v = 0; v < V; v++){
            C[z] +=  A[v]*H[idx];
            idx ++;
        }
    }
}

void add_bias(const int8_t *B, int32_t *C, int16_t *CO, int Z) {
    int z;
    for (z = 0; z < Z; z++) {
        CO[z] = ((C[z] + (B[z] << QIN)) >> SHIFT);
    }
}

void dense2(int8_t *A, const int8_t *H,  const int8_t *B, int16_t *C, int V, int Z) {
    int v, z;
    int32_t temp;
    int idx = 0;
    for (z = 0; z < Z; z++){
        temp = B[z] << QIN;
        for (v = 0; v < V; v++){
            temp +=  A[v]*H[idx];
            idx ++;
        }
        C[z] = (int16_t) (temp >> SHIFT);
    }
}
