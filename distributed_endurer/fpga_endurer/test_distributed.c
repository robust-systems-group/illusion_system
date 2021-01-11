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
 * Note that ENDUReR addresses are M-bit unsigned ints (e_address).
 */
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "endurer.h"

#define TEST_NUM_ITERS 15
#define RESULT_VEC_LEN 20
#define min(x,y) ((x<y)? x: y)
#define MEMORY min(min(M,100),RESULT_VEC_LEN)

/* Clear rram of virt chip */
void clear_rram(int virt) {
    select_chip(virt);
    for (e_address i=0; i<M; i++) endurer_write_word(i, 0);
    flush_and_write_back(virt);
}

e_data A[RESULT_VEC_LEN];
e_data B[RESULT_VEC_LEN];
e_data C[RESULT_VEC_LEN];
e_data D[RESULT_VEC_LEN];
e_data E[RESULT_VEC_LEN];
e_data F[RESULT_VEC_LEN];

/* Compares A and B element-wise. Returns 0 if A and B match; 1 otherwise. */
int vec_not_equals(e_data *A, e_data *B) {
    for (e_uint i = 0; i < RESULT_VEC_LEN; ++i) {
        if (A[i] != B[i]) {
            return 1;
        }
    }
    return 0;
}

int test_three_chips() {
    printf("--------Begin %s()--------\n", __func__);
    // initialize the test results vectors
    memset(A, 0, sizeof(e_data)*RESULT_VEC_LEN);
    memset(B, 0, sizeof(e_data)*RESULT_VEC_LEN);
    memset(C, 0, sizeof(e_data)*RESULT_VEC_LEN);
    memset(D, 0, sizeof(e_data)*RESULT_VEC_LEN);
    memset(E, 0, sizeof(e_data)*RESULT_VEC_LEN);
    memset(F, 0, sizeof(e_data)*RESULT_VEC_LEN);

    reset_endurer();

    for (e_uint i = 0; i < TEST_NUM_ITERS; ++i) {
        e_address address = (e_address)i;
        e_data data = (e_data) 1000 - i;

        // reference
        A[i] = data;
        B[i] = 2*data;
        C[i] = 3*data;

        // simple consistency check
        select_chip(0);
        endurer_write_word(address, data);
        select_chip(1);
        endurer_write_word(address, 2*data);
        select_chip(2);
        endurer_write_word(address, 3*data);
        select_chip(0);
        D[i] = endurer_read_word(address);
        select_chip(1);
        E[i] = endurer_read_word(address);
        select_chip(2);
        F[i] = endurer_read_word(address);
    }

    flush_and_write_back(0);
    flush_and_write_back(1);
    flush_and_write_back(2);

    assert(get_count(0)==(TEST_NUM_ITERS));
    assert(get_count(1)==(TEST_NUM_ITERS));
    assert(get_count(2)==(TEST_NUM_ITERS));

    return vec_not_equals(A, D) || vec_not_equals(B, E) || vec_not_equals(C, F);
}

/*
 * Tests distributed remap
 */
int test_single_distributed_remap() {
    printf("--------Begin %s()--------\n", __func__);
    reset_endurer();

    // initialize the test results vectors
    memset(A, 0, sizeof(e_data)*RESULT_VEC_LEN);
    memset(B, 0, sizeof(e_data)*RESULT_VEC_LEN);
    memset(C, 0, sizeof(e_data)*RESULT_VEC_LEN);
    memset(D, 0, sizeof(e_data)*RESULT_VEC_LEN);
    memset(E, 0, sizeof(e_data)*RESULT_VEC_LEN);
    memset(F, 0, sizeof(e_data)*RESULT_VEC_LEN);

    select_chip(0);
    for (e_uint i = 0; i < MEMORY; ++i) {
        endurer_write_word(i, i);
        A[i] = i;   // reference
    }

    select_chip(0);
    remap();
    select_chip(1);
    remap();
    select_chip(2);
    remap();

    select_chip(1);
    for (e_uint i = 0; i < 4; ++i) {
        endurer_write_word(i, 3*i);
        B[i] = 3*i;   // reference
    }

    select_chip(0);
    remap();
    select_chip(1);
    remap();
    select_chip(2);
    remap();
    assert(get_physical_chip_id(0)==0);
    assert(get_physical_chip_id(1)==1);
    assert(get_physical_chip_id(2)==2);
    select_chip(0);
    distributed_remap(0, 1);
    assert(get_physical_chip_id(0)==1);
    assert(get_physical_chip_id(1)==0);
    assert(get_physical_chip_id(2)==2);

    select_chip(0);
    remap();
    select_chip(1);
    remap();
    select_chip(2);
    remap();
    
    // Physical chip 0 contains virtual chip 1's memory,
    // and vice versa,
    // Write to virt 0, phys 1
    select_chip(0);
    for (e_uint i = 0; i < 4; ++i) {
        endurer_write_word(i, 2*i);
        A[i] = 2*i;   // reference
    }

    select_chip(0);
    remap();
    select_chip(1);
    remap();
    select_chip(2);
    remap();

    for (e_uint i = 0; i < MEMORY; ++i) {
        select_chip(0);
        D[i] = endurer_read_word(i);
        select_chip(1);
        E[i] = endurer_read_word(i);
        select_chip(2);
        F[i] = endurer_read_word(i);
    }

    // these counts are for physical chip address
    assert(get_count(0)==5*M + MEMORY); 
    assert(get_count(1)==5*M + 4 + 4);
    assert(get_count(2)==4*M);

    return vec_not_equals(A, D) || vec_not_equals(B, E) || vec_not_equals(C, F);
} 

int main(int argc, char *argv[]) {
    printf("Hello from ENDUReR.\n");
    enable_distributed_endurer();

    //assert(M==20);
    //assert(M<=MEMORY);
    //assert(N_WORKERS==3);

    initialize(12345);
    assert(!test_three_chips());
    clear_rram(0);
    clear_rram(1);
    clear_rram(2);
    teardown(0);
    teardown(1);
    teardown(2);
 
    initialize(12346);
    assert(!test_single_distributed_remap());
    teardown(0);
    teardown(1);
    teardown(2);

    printf("All results vectors match.\n");
    printf("Done.\n");

    return 0;
}
