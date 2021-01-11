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
#define QIN 7
#define SHIFT (QOUT - QIN)
#define BIAS ( ((int32_t) 1) << (SHIFT - 1))
#define OMAX ((((int32_t) 1) << (QOUT)) - 1)
#define IMAX ((((int16_t) 1) << (QIN)) - 1)


inline int32_t max(int32_t a, int32_t b){
    if (a > b) return a;
    else return b; 
}

inline int min(int a, int b){
    if (a < b) return a;
    else return b; 
}

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


__attribute__((optimize("unroll-loops")))
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

__attribute__((optimize("unroll-loops")))
void add_bias(const int8_t *B, int32_t *C, int16_t *CO, int Z) {
    int z;
    for (z = 0; z <Z; z++) {
        CO[z] = (int16_t) ((C[z] + (B[z] << SHIFT)) >> SHIFT);
    }
}

__attribute__((optimize("unroll-loops")))
void dense2(int16_t *A, const int8_t *H,  const int8_t *B, int16_t *C, int V, int Z) {
    int v, z;
    int32_t temp;
    int idx = 0;
    for (z = 0; z < Z; z++){
        temp = B[z] << SHIFT;
        for (v = 0; v < V; v++){
            //printf("%04d ", A[v]);
            //printf("%04d ", H[idx]);
            temp +=  A[v]*H[idx];
            //printf("%04d ", temp);
            idx ++;
        }
        //printf("\n"); 
        C[z] = (int16_t) (temp>>SHIFT) ;
    }
}

//__attribute__((optimize("unroll-loops")))
int conv2D_1filter(int8_t *A, const int8_t *H, int8_t B, int IC, int X, int FX, int MPP, int8_t *O ) {
    // H: Weights (Filter)
    // A: Input
    // O: Output
    // MPP: Max pool (power of 2):  only 2^MPP max pooling supported as no divider
    // Biases for each filter are added in after (not in this function)
    // Filter is IC x FX x FY
    // Input is IC x X x Y
    // Output is going tobe X x Y
    
    register int32_t temp = 0;
    
    int i, j;
    int fx, fy, c;
    
    
    int idx1, idx2, idxo;

    int off = X*X;
    int foff = FX*FX;

    int M = X - FX + 1; //Lower half of the filter ignored from padding)
    
    for (i = 0; i < M; i++){  
        for (j = 0; j < M; j++){  //Iterate through the input image
            idx1 = i*X +j;
            idx2 = 0;
            temp = B << SHIFT;
            for (c = 0; c < IC; c++){
                for (fx = 0; fx < FX; fx++){  //Iterate through non-zero input region
                    for (fy = 0; fy < FX; fy++){
                        //printf("%d ", idx1);
                        //putbyte(A[idx1]);
                        //putbyte(H[idx2]);
                        temp +=  A[idx1] * H[idx2];
                        idx1++;
                        idx2++;
                    }
                    idx1 += X  - FX ;
                }
                idx1 += off - X*FX; 
                //printf("\n");
            }
            //printf("\n %d \n", temp);
            //printf("%6d ", temp);
            temp = qRelu(temp);
            
            //Max pool 
            idxo = (i >> MPP)*(M >> MPP) + (j >> MPP); 
            //printf("%d ",idxo);
            if (( (int8_t) temp) > O[idxo]){
                O[idxo] = (int8_t) temp;
            }
            //printf("%04d ", temp);
            idxo++;
        }
        //printf("\n");
    }
    return 0;
}

//__attribute__((optimize("unroll-loops")))
int conv2D_1filter_pad(int8_t *A, const int8_t *H, int8_t B, int IC, int X, int FX, int MPP, int8_t *O ) {
    // H: Weights (Filter)
    // A: Input
    // O: Output
    // MPP: Max pool (power of 2):  only 2^MPP max pooling supported as no divider
    // Biases for each filter are added in after (not in this function)
    // Filter is IC x FX x FY
    // Input is IC x X x Y
    // Output is going tobe X x Y
    
    register int32_t temp = 0;
    
    int i, j;
    int fx, fy, c;
    
    int SX, SY, EX, EY;
    
    int idx1, idx2, idxo;
    int IDX1, IDX2;

    int off = X*X;
    int foff = FX*FX;

    int HF = FX >> 1; //Lower half of the filter ignored from padding)
    
    for (i = 0; i < X; i++){  
        SX = max(0, HF - i);
        EX = min(FX, X + HF - i);
        for (j = 0; j < X; j++){  //Iterate through the input image
            SY = max(0, HF - j);
            EY = min(FX, X + HF - j);
            
            IDX1 = max(i - HF, 0)*X + max(j - HF, 0);
            IDX2 = FX*SX + SY;
            
            temp = B << SHIFT;  //Pre-load with bias
            for (c = 0; c < IC; c++){
                idx1 = IDX1;
                idx2 = IDX2;
                for (fx = SX; fx < EX; fx++){  //Iterate through non-zero input region
                    for (fy = SY; fy < EY; fy++){
                        temp = temp + A[idx1]*H[idx2];
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
            temp = qRelu(temp);
            //Max pool 
            idxo = (i >> MPP)*(X >> MPP) + (j >> MPP); 
            if (( (int8_t) temp) > O[idxo]) {O[idxo] = (int8_t) temp;}
            idxo++;
        }
    }
    return 0;
}
