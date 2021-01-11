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

#define min(x,y) ((x<y)? x: y)
#define RESULT_VEC_LEN 20
#define MEMORY min(min(M,100),RESULT_VEC_LEN)
#define TEST_NUM_ITERS 9


e_data A[RESULT_VEC_LEN];
e_data B[RESULT_VEC_LEN];
e_data C[RESULT_VEC_LEN];
e_data D[RESULT_VEC_LEN];

/* Clear rram of virt chip */
void clear_rram(int virt) {
    select_chip(virt);
    for (e_address i=0; i<M; i++) endurer_write_word(i, 0);
    flush_and_write_back(virt);
}

/* Compares A and B element-wise. Returns 0 if A and B match; 1 otherwise. */
int vec_not_equals(e_data *A, e_data *B) {
    for (e_uint i = 0; i < RESULT_VEC_LEN; ++i) {
        if (A[i] != B[i]) {
            return 1;
        }
    }
    return 0;
}

int test_two_chips_0() {
    printf("--------Begin %s()--------\n", __func__);
    // initialize the test results vectors
    memset(A, 0, sizeof(e_data)*RESULT_VEC_LEN);
    memset(B, 0, sizeof(e_data)*RESULT_VEC_LEN);
    memset(C, 0, sizeof(e_data)*RESULT_VEC_LEN);
    memset(D, 0, sizeof(e_data)*RESULT_VEC_LEN);
    reset_endurer();

    for (e_uint i = 0; i < TEST_NUM_ITERS; ++i) {
        e_address address = (e_address)i;
        e_data data = (e_data) 1000 - i;

        // reference
        A[i] = data;
        B[i] = 2*data;

        // simple consistency check
        select_chip(0);
        endurer_write_word(address, data);
        select_chip(1);
        endurer_write_word(address, 2*data);
        select_chip(0);
        C[i] = endurer_read_word(address);
        select_chip(1);
        D[i] = endurer_read_word(address);
    }

    flush_and_write_back(0);
    flush_and_write_back(1);

    assert(get_count(0)==TEST_NUM_ITERS);
    assert(get_count(1)==TEST_NUM_ITERS);

    return vec_not_equals(A, C) || vec_not_equals(B, D);
}

int test_two_chips_1() {
    printf("--------Begin %s()--------\n", __func__);
    // initialize the test results vectors
    memset(A, 0, sizeof(e_data)*RESULT_VEC_LEN);
    memset(B, 0, sizeof(e_data)*RESULT_VEC_LEN);
    memset(C, 0, sizeof(e_data)*RESULT_VEC_LEN);
    memset(D, 0, sizeof(e_data)*RESULT_VEC_LEN);
    reset_endurer();

    // Use the first four addresses as a "scratch area,"
    // and ensure that the rest of RRAM remains consistent.
    select_chip(0);
    for (e_uint i = 0; i < MEMORY; ++i) {
        endurer_write_word(i, i);
        A[i] = i;   // reference
    }

    for (e_uint i = 0; i < 4; ++i) {
        endurer_write_word(i, 99);
        A[i] = 99;
    }

    for (e_uint i = 0; i < MEMORY; ++i) {
        select_chip(0);
        C[i] = endurer_read_word(i);
        select_chip(1);
        D[i] = endurer_read_word(i);
        // single threaded test, cannot do endurer_read_word on other chip!
    }

    flush_and_write_back(0);
    flush_and_write_back(1);

    assert(get_count(0)==(MEMORY+4)); //since buffer had M items, and everything was written back
    assert(get_count(1)==0);  //since buffer was empty and had nothing to write back

    return vec_not_equals(A, C) || vec_not_equals(B, D);
}

/*
 * Tests remap().
 */
int test_two_chips_2() {
    printf("--------Begin %s()--------\n", __func__);
    // initialize the test results vectors
    memset(A, 0, sizeof(e_data)*RESULT_VEC_LEN);
    memset(B, 0, sizeof(e_data)*RESULT_VEC_LEN);
    memset(C, 0, sizeof(e_data)*RESULT_VEC_LEN);
    memset(D, 0, sizeof(e_data)*RESULT_VEC_LEN);
    reset_endurer();


    // Use the first four addresses as a "scratch area,"
    // and ensure that the rest of RRAM remains consistent.

    for (e_uint i = 0; i < MEMORY; ++i) {
        select_chip(0);
        endurer_write_word(i, i);
        A[i] = i;   // reference
        select_chip(1);
        endurer_write_word(i, M-i);
        B[i] = M-i;   // reference
    }

    select_chip(0);
    remap();
    select_chip(1);
    remap();

    for (e_uint i = 0; i < MEMORY; ++i) {
        select_chip(0);
        C[i] = endurer_read_word(i);
        select_chip(1);
        D[i] = endurer_read_word(i);
     }

    assert(get_count(0)==(M + MEMORY));
    assert(get_count(1)==(M + MEMORY)); // once fro initial write, once for remap   

    return vec_not_equals(A, C) || vec_not_equals(B,D);
}

/*
 * Tests remap & wear.
 */
int test_two_chips_3() {
    printf("--------Begin %s()--------\n", __func__);
    reset_endurer();

    // initialize the test results vectors
    memset(A, 0, sizeof(e_data)*RESULT_VEC_LEN);
    memset(B, 0, sizeof(e_data)*RESULT_VEC_LEN);
    memset(C, 0, sizeof(e_data)*RESULT_VEC_LEN);
    memset(D, 0, sizeof(e_data)*RESULT_VEC_LEN);


    // Use the first four addresses as a "scratch area,"
    // and ensure that the rest of RRAM remains consistent.

    select_chip(0);
    for (e_uint i = 0; i < MEMORY; ++i) {
        endurer_write_word(i, i);
        A[i] = i;   // reference
    }

    select_chip(0);
    remap();
    select_chip(1);
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
    select_chip(0);
    remap();
    select_chip(1);
    remap();
    select_chip(0);
    remap();
    select_chip(1);
    remap();

    for (e_uint i = 0; i < MEMORY; ++i) {
        select_chip(0);
        C[i] = endurer_read_word(i);
        select_chip(1);
        D[i] = endurer_read_word(i);
    }

    assert(get_count(0)==4*M + MEMORY); 
    assert(get_count(1)==4*M + 4);

    return vec_not_equals(A, C) || vec_not_equals(B, D);
}

/*
 * Tests write locations after many distributed & local remaps
 */
int test_two_chips_4() {
    printf("--------Begin %s()--------\n", __func__);
    printf(" This may take ~1min.\n");
    reset_endurer();

    // initialize the test results vectors
    memset(A, 0, sizeof(e_data)*RESULT_VEC_LEN);
    memset(B, 0, sizeof(e_data)*RESULT_VEC_LEN);
    memset(C, 0, sizeof(e_data)*RESULT_VEC_LEN);
    memset(D, 0, sizeof(e_data)*RESULT_VEC_LEN);

    // Use the first four addresses as a "scratch area,"
    // and ensure that the rest of RRAM remains consistent.

    select_chip(0);
    for (e_uint i = 0; i < MEMORY; ++i) {
        endurer_write_word(i, i);
        A[i] = i;   // reference
    }

    select_chip(0);
    remap();
    select_chip(1);
    remap();

    for (e_uint i = 0; i < 4; ++i) {
        endurer_write_word(i, 3*i);
        B[i] = 3*i;   // reference
    }

    for (int i=0; i<16; i++) {
        for (int j=0; j<5e3; j++) {
            select_chip(0);
            remap();
            select_chip(1);
            remap();
        }
        distributed_remap(0, 1);
    }

    select_chip(0);
    for (e_uint i = 0; i < 4; ++i) {
        endurer_write_word(i, 2*i);
        A[i] = 2*i;   // reference
    }

    select_chip(0);
    remap();
    select_chip(1);
    remap();

    for (e_uint i = 0; i < MEMORY; ++i) {
        select_chip(0);
        C[i] = endurer_read_word(i);
        select_chip(1);
        D[i] = endurer_read_word(i);
    }

    return vec_not_equals(A, C) || vec_not_equals(B, D);
}



int main(int argc, char *argv[]) {
    printf("Hello from ENDUReR.\n");
    enable_distributed_endurer();

    initialize(12345);
    assert(!test_two_chips_0());
    clear_rram(0);
    clear_rram(1);
    teardown(0);
    teardown(1);
 
    initialize(12346);
    assert(!test_two_chips_1());
    clear_rram(0);
    clear_rram(1);
    teardown(0);
    teardown(1);
    
    initialize(12347);
    assert(!test_two_chips_2());
    clear_rram(0);
    clear_rram(1);
    teardown(0);
    teardown(1);
    
    initialize(12348);
    assert(!test_two_chips_3());
    clear_rram(0);
    clear_rram(1);
    teardown(0);
    teardown(1);
    
    initialize(12375);
    assert(!test_two_chips_4());
    clear_rram(0);
    clear_rram(1);
    teardown(0);
    teardown(1);
    
    printf("All results vectors match.\n");
    printf("Done.\n");

    return 0;
}
