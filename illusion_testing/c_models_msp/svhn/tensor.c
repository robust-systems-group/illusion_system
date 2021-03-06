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
#include "test.h"
#include "omsp_func.h"

#define RESHI (volatile int16_t *) 0x013C
#define RESLO (volatile int16_t *) 0x013A
#define MACS  (volatile int16_t *) 0x0136
#define MPYS  (volatile int16_t *) 0x0132
#define OP2   (volatile int16_t *) 0x0138

#define QOUT 11
#define QIN 7
#define SHIFT (QOUT - QIN)
#define BIAS ( ((int32_t) 1) << (SHIFT - 1))
#define OMAX ((((int32_t) 1) << (QOUT)) - 1)
#define IMAX ((((int16_t) 1) << (QIN)) - 1)

inline int max(int a, int b){
    if (a > b) return a;
    else return b; 
}

inline int min(int a, int b){
    if (a < b) return a;
    else return b; 
}

inline int8_t qRelu(int32_t a){
    //Rulu Quantized to q bits
    //m_in is number of dec bits for fp in
    //m_out is number of dec bits qRelu quantizes to
    if (a < 0){
        return 0;
    }
    else {
        int32_t rup = a + BIAS; // round up
        if (rup > OMAX) return IMAX; //return 2**frac -1
        else return (int8_t) (rup >> SHIFT); //bit shift back to m_out frac bits
    }
}

void dense(int8_t *A, const int8_t *H, int32_t *C, int V, int Z) {
    int v, z;
    register int32_t temp;
    int idx = 0;
    for (z = 0; z < Z; z++){
        *(RESHI) = 0;
        *(RESLO) = 0;
        for (v = 0; v < V; v++){
            *(MACS) = A[v];
            *(OP2) = H[idx];
            idx ++;
        }
        union Out {struct {uint16_t l; uint16_t h;} b; int32_t o;} ou;
        ou.b.l = *(RESLO);
        ou.b.h = *(RESHI);
        C[z] += ou.o;
    }
    return;
}

void add_bias(const int8_t *B, int32_t *C, int16_t *CO, int Z) {
    int z;
    for (z = 0; z < Z; z++) {
        CO[z] = (int16_t) ((C[z] + (B[z] << SHIFT)) >> SHIFT);
    }
    return;
}

void dense_bias(int16_t *A, const int8_t *H,  const int8_t *B, int16_t *C, int V, int Z) {
    int v, z;
    int idx = 0;
    for (z = 0; z < Z; z++){
        *(RESHI) = 0;
        *(RESLO) = 0;
        *(MACS) = (int16_t) (B[z] << SHIFT);  //Pre-load with bias
        *(OP2) = 1; 
        for (v = 0; v < V; v++){
            *(MACS) = A[v];
            *(OP2) = H[idx];
            idx ++;
        }
        union Out {struct {uint16_t l; uint16_t h;} b; int32_t o;} ou;
        ou.b.l = *(RESLO);
        ou.b.h = *(RESHI);
        C[z] = (int16_t) (ou.o >> (SHIFT)); //quantizing to 16-bit implicit
    }
    return;
}

void conv2D_1filter(int8_t *A, const int8_t *H, int8_t B, int IC, int X, int FX, int MPP, int8_t *O ) {
    // H: Weights (Filter)
    // A: Input
    // O: Output
    // MPP: Max pool (power of 2):  only 2^MPP max pooling supported as no divider
    // Filter is IC x FX x FY
    // Input is IC x X x Y
    // Output is going tobe X x Y
    int8_t temp = 0;
    
    int i, j;
    int i_p, j_p;
    int fx, fy, c;
    int idx1, idx2, idxo;
    int first_pool;
    int off = X*X;
    int foff = X*FX;
    int M = X - FX + 1;
    int M_p = M >> MPP;
    int MPP_MASK =  (1 << MPP) - 1;
    for (i = 0; i < M; i++){  
        for (j = 0; j < M; j++){  //Iterate through the input image
            idx1 = i*X +j;
            idx2 = 0;
            i_p = i >> MPP;
            j_p = j >> MPP;
            first_pool = !((i & MPP_MASK) || (j & MPP_MASK));
            idxo = i_p*M_p + j_p; 
            //Start maxpool 
            *(RESHI) = 0;
            *(RESLO) = 0;
            *(MACS) = (int16_t) (B<<SHIFT);  //Pre-load with bias
            *(OP2) = 1; 
            for (c = 0; c < IC; c++){
                for (fx = 0; fx < FX; fx++){  //Iterate through non-zero input region
                    for (fy = 0; fy < FX; fy++){
                        *(MACS) = A[idx1];
                        *(OP2) = H[idx2];
                        idx1++;
                        idx2++;
                    }
                    idx1 += X - FX;
                }
                idx1 += off - foff;
            }
            //Batch norm, requantize output
            union Out {struct {uint16_t l; uint16_t h;} b; int32_t o;} ou;
            ou.b.l = *(RESLO);
            ou.b.h = *(RESHI);
            temp = qRelu(ou.o);
            //Max pool 
            if (first_pool) {
                O[idxo] = temp;
            } else if (temp > O[idxo]) {
                O[idxo] = temp;
            }
        }
    }
    return;
}


void conv2D_1filter_pad(int8_t *A, const int8_t *H, int8_t B, int IC, int X, int FX, int MPP, int8_t *O ) {
    // H: Weights (Filter)
    // A: Input
    // O: Output
    // MPP: Max pool (power of 2):  only 2^MPP max pooling supported as no divider
    // Biases for each filter are added in after (not in this function)
    // Filter is IC x FX x FY
    // Input is IC x X x Y
    // Output is going tobe X x Y
    
    int8_t temp = 0;
    
    int i, j, i_p, j_p;
    int fx, fy, c;
    
    int SX, SY, EX, EY;
    
    int idx1, idx2, idxo;
    int IDX1, IDX2;
    int first_pool;

    int off = X*X;
    int foff = FX*FX;

    int HF = FX >> 1; //Lower half of the filter ignored from padding)
    int M_p = X >> MPP;
    int MPP_MASK =  (1 << MPP) - 1;
    for (i = 0; i < X; i++){  
        SX = max(0, HF - i);
        EX = min(FX, X + HF - i);
        for (j = 0; j < X; j++){  //Iterate through the input image
            SY = max(0, HF - j);
            EY = min(FX, X + HF - j);
            
            IDX1 = max(i - HF, 0)*X + max(j - HF, 0);
            IDX2 = FX*SX + SY;
            
            i_p = i >> MPP;
            j_p = j >> MPP;
            first_pool = !((i & MPP_MASK) || (j & MPP_MASK));
            idxo = i_p*M_p + j_p; 
            
            *(RESHI) = 0;
            *(RESLO) = 0;
            *(MACS) = (int16_t)(B << SHIFT);  //Pre-load with bias
            *(OP2) = 1; 
            for (c = 0; c < IC; c++){
                idx1 = IDX1;
                idx2 = IDX2;
                for (fx = SX; fx < EX; fx++){  //Iterate through non-zero input region
                    for (fy = SY; fy < EY; fy++){
                        *(MACS) = A[idx1];
                        *(OP2) = H[idx2];
                        idx1++;
                        idx2++;
                    }
                    idx1 += X + SY - EY ;
                    idx2 += FX + SY - EY ;
                }
                IDX1 += off;
                IDX2 += foff;
            }
            //Batch norm, requantize output
            union Out {struct {uint16_t l; uint16_t h;} b; int32_t o;} ou;
            ou.b.l = *(RESLO);
            ou.b.h = *(RESHI);
            temp = qRelu(ou.o);
            //Max pool
            if (first_pool) {
                    O[idxo] = temp;
            } else if (temp > O[idxo]) {
                    O[idxo] = temp;
            }
        }
    }
    return;
}

#ifdef NV

void conv2D_1filter_pad_nv(int8_t *A, const int8_t *H, int8_t B, int IC, int X, int FX, int MPP, int16_t *O ) {
    // H: Weights (Filter)
    // A: Input
    // O: Output
    // MPP: Max pool (power of 2):  only 2^MPP max pooling supported as no divider
    // Biases for each filter are added in after (not in this function)
    // Filter is IC x FX x FY
    // Input is IC x X x Y
    // Output is going tobe X x Y
    
    int8_t temp = 0;
    
    int i, j, i_p, j_p;
    int fx, fy, c;
    
    int SX, SY, EX, EY;
    
    int idx1, idx2, idxo;
    int IDX1, IDX2;
    int first_pool;

    int off = X*X;
    int foff = FX*FX;

    int HF = FX >> 1; //Lower half of the filter ignored from padding)
    int M_p = X >> MPP;
    int MPP_MASK =  (1 << MPP) - 1;
    for (i = 0; i < X; i++){  
        SX = max(0, HF - i);
        EX = min(FX, X + HF - i);
        for (j = 0; j < X; j++){  //Iterate through the input image
            SY = max(0, HF - j);
            EY = min(FX, X + HF - j);
            
            IDX1 = max(i - HF, 0)*X + max(j - HF, 0);
            IDX2 = FX*SX + SY;
            
            i_p = i >> MPP;
            j_p = j >> MPP;
            first_pool = !((i & MPP_MASK) || (j & MPP_MASK));
            idxo = i_p*M_p + j_p; 
            
            *(RESHI) = 0;
            *(RESLO) = 0;
            *(MACS) = (int16_t)(B << SHIFT);  //Pre-load with bias
            *(OP2) = 1; 
            for (c = 0; c < IC; c++){
                idx1 = IDX1;
                idx2 = IDX2;
                for (fx = SX; fx < EX; fx++){  //Iterate through non-zero input region
                    for (fy = SY; fy < EY; fy++){
                        *(MACS) = A[idx1];
                        *(OP2) = H[idx2];
                        idx1++;
                        idx2++;
                    }
                    idx1 += X + SY - EY ;
                    idx2 += FX + SY - EY ;
                }
                IDX1 += off;
                IDX2 += foff;
            }
            //Batch norm, requantize output
            union Out {struct {uint16_t l; uint16_t h;} b; int32_t o;} ou;
            ou.b.l = *(RESLO);
            ou.b.h = *(RESHI);
            temp = qRelu(ou.o);
            union Rout {struct {int8_t l; int8_t h;} o; int16_t r;} rou;
            rou.r = O[idxo>>1];
            if (idxo != ((idxo>>1)<<1)) {
                if (first_pool) {
                    rou.o.h = temp;
                    O[idxo>>1] = rou.r;
                } else if (temp > rou.o.h) {
                    rou.o.h = temp;
                    O[idxo>>1] = rou.r;
                }
            } else {
                if (first_pool) {
                    rou.o.l = temp;
                    O[idxo>>1] = rou.r;
                } else if (temp > rou.o.l) {
                    rou.o.l = temp;
                    O[idxo>>1] = rou.r;
                }
            }
            //Max pool
        }
    }
    return;
}

#endif
