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

/*
 * This file defines a test driver for the ENDUReR mechanism.
 *
 * Note that ENDUReR addresses are 16-bit unsigned ints (e_address).
 */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "endurer.h"

#define N1 4
#define N2 4
#define N3 5

#define TEST_RUNS 3
#define REMAP_PERIOD 3
#define GLOBAL_REMAP_PERIOD 3


// all arrays static to initialize to 0 
static e_data A[N1][N2];
static e_data B[N2][N3];
//static e_data C[N1][N3]; // C = A*B
static e_data D[N3][N2]; 
static e_data E[N1][N2]; // E = C*D
static e_data F[N1][N2]; // expected

e_address A_addr0[N1][N2];
e_address B_addr0[N2][N3];
e_address C_addr0[N1][N3];
e_address C_addr1[N1][N3];
e_address D_addr1[N3][N2];
e_address E_addr1[N1][N2];

int mat_equals(e_data data_1[N1][N2], e_data data_2[N1][N2]) {
    for (e_uint i = 0; i < N1; ++i) {
        for (e_uint j = 0; j < N2; ++j) {
            if (data_1[i][j] != data_2[i][j]) {
                return 1;
            }
        }
    }
    return 0;
}

int math1() {
    //printf("Started math 1: %d\n", chip->id);
    select_chip(0);
    for (e_uint i=0; i<N1; i++) {
        for (e_uint k=0; k<N3; k++) {
            endurer_write_word(C_addr0[i][k], 0);
            for (e_uint j=0; j<N2; j++) {
                e_data x = endurer_read_word(A_addr0[i][j]);
                e_data y = endurer_read_word(B_addr0[j][k]);
                e_data z = endurer_read_word(C_addr0[i][k]);
                z = z + x*y;
                endurer_write_word(C_addr0[i][k], z);
            }
        }
    }

    // Transfer output
    e_data word;
    for (int i=0; i<N1; i++) for (int j=0; j<N2; j++) {
        select_chip(0);
        word = endurer_read_word(C_addr0[i][j]);
        select_chip(1);
        endurer_write_word(C_addr1[i][j], word);
    }

    return 0;
}



int math2() {
    //printf("Started math2: %d\n", chip->id);
    select_chip(1);
    for (e_uint i=0; i<N1; i++) {
        for (e_uint k=0; k<N2; k++) {
            endurer_write_word(E_addr1[i][k], 0);
            for (e_uint j=0; j<N3; j++) {
                e_data x = endurer_read_word(C_addr1[i][j]);
                e_data y = endurer_read_word(D_addr1[j][k]);
                e_data z = endurer_read_word(E_addr1[i][k]);
                z = z + x*y;
                endurer_write_word(E_addr1[i][k], z);
            }
        }
    }
 
    return 0;
}

int remap_start = 0;

void calculate_gemm() {
    for (int i=0; i<TEST_RUNS; i++) {
            for (int k=0; k<REMAP_PERIOD; k++) {
		        math1();
		        math2();
            }
// while remap() and global_remap() are stubbed out internally
//   when those macros are 1,
// it's unnecessary overhead to call these anyway, especially
//   since we'd be risking an alignment problem
#if (NO_ENDURER==0)
            //for (int l=0; l<N_WORKERS; l++) remap(chips+l);
#endif
    }

}

int main() {
    printf("Hello from ENDUReR.\n");
    enable_endurer();

    A[0][0] = 1; A[0][1] = 2; A[1][0] = 0; A[1][1] = 3;
    B[0][0] = 1; B[1][1] = 1; B[2][2] = 1; B[3][3] = 1;
    // expect: C[0][0] = 1; C[0][1] = 2; C[1][0] = 0; C[1][1] = 3; 
    D[0][0] = 1; D[0][1] = 1; D[0][2] = 1; D[0][3] = 1;
    F[0][0] = 1; F[0][1] = 1; F[0][2] = 1; F[0][3] = 1;

    select_chip(0);
    // initialize addresses & values
    for (e_uint i=0; i<N1; i++) {
        for (e_uint j=0; j<N2; j++) {
            A_addr0[i][j] = 0*M + N2*i + j; // chip0
            E_addr1[i][j] = 1*M + N1*N3 + N3*N2 + N2*i + j; // chip 1
            endurer_write_word(A_addr0[i][j], A[i][j]);
        }
    }
    for (e_uint i=0; i<N2; i++) {
        for (e_uint j=0; j<N3; j++) {
            B_addr0[i][j] = 0*M + N1*N2 + N3*i + j; // chip0
            endurer_write_word(B_addr0[i][j], B[i][j]);
        }
    }
    for (e_uint i=0; i<N1; i++) {
        for (e_uint j=0; j<N3; j++) {
            C_addr0[i][j] = 0*M + N1*N2 + N2*N3 + N3*i + j; // chip0
            C_addr1[i][j] = 1*M + N3*i + j; // chip1
        }
    }
    select_chip(1);
    for (e_uint i=0; i<N3; i++) {
        for (e_uint j=0; j<N2; j++) {
            D_addr1[i][j] = 1*M + N1*N3 + N2*i + j; // chip1
            endurer_write_word(D_addr1[i][j], D[i][j]);
        }
    }

    calculate_gemm();

    select_chip(1);
    // Read output
    for (e_uint i=0; i<N1; i++) {
        for (e_uint j=0; j<N2; j++) {
            E[i][j] = endurer_read_word(E_addr1[i][j]);
        }
    }

    assert(mat_equals(E, F)==0);

    printf("Done computing E=(A*B)*D.\n");
    printf("Result matrices match.\n");

    return 0;
}
