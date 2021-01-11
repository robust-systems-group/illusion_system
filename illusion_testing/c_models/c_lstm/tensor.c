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

#define QOUT 24
#define QIN 12
#define SHIFT (QOUT - QIN)
#define BIAS ( ((int32_t) 1) << (SHIFT - 1))
#define OMAX ((((int32_t) 1) << (QOUT)) - 1)
#define IMAX ((((int16_t) 1) << (QIN)) - 1)

inline int32_t qRelu(int32_t a){
    //Rulu Quantized to q bits
    //m_in is number of dec bits for fp in
    //m_out is number of dec bits qRelu quantizes to
    if (a < 0){
        return 0;
    }
    else {
        int32_t rup = a + BIAS; // round up
        if (rup > OMAX) return IMAX; //return 2**frac -1
        else return (rup >> SHIFT); //bit shift back to m_out frac bits
    }
}

#define SIGMAX (4 << QOUT)
inline int16_t qSig(int32_t a) {
    if (a > SIGMAX) {
        return IMAX;
    } else if (a < -SIGMAX) {
        return (int16_t) 0;
    } else {

        return (((a >> 3) + (1 << (QOUT-1))) >> SHIFT);
    }
}
#define TANHMAX (1 << QOUT)
inline int16_t qTanh(int32_t a) {
    if (a > TANHMAX) {
        return IMAX;
    } else if (a < -TANHMAX) {
        return -IMAX;
    } else {
        return (int16_t) (a >> SHIFT);
    }
}

void lstm(int16_t *I, int16_t *H, int32_t *C,  const int16_t *W_IH, const int16_t *W_HH, const int16_t *B, int Z, int16_t *H_O, int32_t *C_O) {
    int v,z,t;
    int32_t gates[4*Z];
    int16_t igate[Z];
    int16_t fgate[Z];
    int16_t cgate[Z];
    int16_t ogate[Z];
    
    for (z = 0; z < 4*Z; z++) {
        gates[z] = (B[z] << SHIFT);
    }
    dense(I, W_IH, gates, IDIM, Z*4); 
    dense(H, W_HH, gates, HDIM, Z*4); 
    //for (z = 0; z < 4*Z; z++) {
    //    printf("%08x\n", gates[z]);
    //}
    
    for (z = 0; z < Z; z++) {
        igate[z] = qSig(gates[z]);
        fgate[z] = qSig(gates[z+Z]);
        cgate[z] = qTanh(gates[z+2*Z]);
        ogate[z] = qSig(gates[z+3*Z]);
        C_O[z] = fgate[z]*(C[z] >> SHIFT) + igate[z]*cgate[z];
        H_O[z] = (ogate[z]*qTanh(C_O[z]) >> SHIFT);
    }
    //for (z = 0; z < Z; z++) {
    //    printf("%08x\n", igate[z]);
    //}
    //for (z = 0; z < Z; z++) {
    //    printf("%08x\n", cgate[z]);
    //}
    //for (z = 0; z < Z; z++) {
    //    printf("%08x\n", qTanh(C_O[z]));
    //}
    //for (z = 0; z < Z; z++) {
    //    printf("%08x\n", H_O[z]);
    //}
}


void dense(int16_t *A, const int16_t *H, int32_t *C, int V, int Z) {
    int v, z;
    int idx = 0;
    for (z = 0; z < Z; z++){
        for (v = 0; v < V; v++){
            C[z] +=  A[v]*H[idx];
            idx ++;
        }
    }
}

void add_bias(const int16_t *B, int32_t *C, int32_t *CO, int Z) {
    int z;
    for (z = 0; z < Z; z++) {
        CO[z] = ((C[z] + (B[z] << SHIFT)) >> SHIFT);
    }
}

void dense2(int16_t *A, const int16_t *H,  const int16_t *B, int32_t *C, int V, int Z) {
    int v, z;
    int32_t temp;
    int idx = 0;
    for (z = 0; z < Z; z++){
        temp = B[z] << SHIFT;
        for (v = 0; v < V; v++){
            temp +=  A[v]*H[idx];
            idx ++;
        }
        C[z] = temp >> SHIFT;
    }
}
