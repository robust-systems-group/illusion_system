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

#define RESHI (volatile int16_t *) 0x013C
#define RESLO (volatile int16_t *) 0x013A
#define MACS  (volatile int16_t *) 0x0136
#define MPYS  (volatile int16_t *) 0x0132
#define OP2   (volatile int16_t *) 0x0138

#define QOUT 11
#define QIN 4
#define SHIFT (QOUT - QIN)
#define IMAX (int8_t) 1<<QIN

#define SIGMAX (4 << QOUT)
#define SIGHALF (1 << (QOUT - 1))
#define TANHMAX (1 << QOUT)
#define ROUND (1 << (SHIFT-1))

inline int8_t qSig(int32_t a) {
    if (a > SIGMAX) {
        return IMAX;
    } else if (a < -SIGMAX) {
        return 0;
    } else {
        return ((a >> 3) + SIGHALF + ROUND) >> SHIFT; //Round up as between 0/1
    }
}
inline int8_t qTanh(int32_t a) {
    if (a > TANHMAX) {
        return IMAX;
    } else if (a < -TANHMAX) {
        return -IMAX;
    } else if (a > 0){
        return a>>SHIFT;
        //return ((a + ROUND) >> SHIFT); //Not needed for some reason
    } else {
        return a>>SHIFT;
        //return ((a - ROUND) >> SHIFT);
    }
}

void lstm(int8_t *I, int8_t *H, int16_t *C,  const int8_t *W_IH, const int8_t *W_HH, const int8_t *B, int Z, int8_t *H_O, int16_t *C_O) {
    int v,z,t;
    int32_t gates[4*Z];
    int8_t igate[Z];
    int8_t fgate[Z];
    int8_t cgate[Z];
    int8_t ogate[Z];
    int16_t temp;
    for (z = 0; z < 4*Z; z++) {
        gates[z] = ((int32_t)  B[z] ) << QIN;
    }
    dense(I, W_IH, gates, IDIM, Z*4); 
    dense(H, W_HH, gates, HDIM, Z*4); 
    
    for (z = 0; z < Z; z++) {
        igate[z] = qSig(gates[z]);
        fgate[z] = qSig(gates[z+Z]);
        cgate[z] = qTanh(gates[z+2*Z]);
        ogate[z] = qSig(gates[z+3*Z]);
    }
    
    
    for (z = 0; z < Z; z++) {
        //Compute C_O
        *(RESHI) = 0;
        *(RESLO) = 0;
        *(MACS) =  (int16_t) fgate[z];
        *(OP2) =  (int16_t) (C[z] >> QIN);
        *(MACS) = (int16_t) igate[z];
        *(OP2) = (int16_t) cgate[z];
        
        union Out {struct {uint16_t l; uint16_t h;} b; int32_t o;} ou;
        ou.b.l = *(RESLO);
        ou.b.h = *(RESHI);
        C_O[z] = (int16_t) ou.o;
        
        //COmpute H_O
        *(RESHI) = 0;
        *(RESLO) = 0;
        int32_t temp = ((int32_t)C_O[z]) << (SHIFT-QIN);
        *(MACS)  =  qTanh(((int32_t)C_O[z]) << (SHIFT-QIN));
        *(OP2) = ogate[z];
        ou.b.l = *(RESLO);
        ou.b.h = *(RESHI);
        H_O[z] = (int8_t) (ou.o >> QIN);

    }
    return;
}

void dense(int8_t *A, const int8_t *H, int32_t *C, int V, int Z) {
    int v, z;
    register int32_t temp;
    int idx = 0;
    for (z = 0; z < Z; z++){
        *(RESHI) = 0;
        *(RESLO) = 0;
        for (v = 0; v < V; v++){
            *(MACS) = (int16_t)  A[v];
            *(OP2) = (int16_t)  H[idx];
            idx ++;
        }
        union Out {struct {uint16_t l; uint16_t h;} b; int32_t o;} ou;
        ou.b.l = *(RESLO);
        ou.b.h = *(RESHI);
        C[z] += ou.o;
    //    putbyte(C[z]);
    }
    return;
}

void add_bias(const int8_t *B, int32_t *C, int16_t *CO, int Z) {
    int z;
    for (z = 0; z < Z; z++) {
        CO[z] = ((C[z] + (B[z] << SHIFT)) >> SHIFT);
    }
    return;
}
