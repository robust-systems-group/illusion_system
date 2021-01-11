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

/* Compares A and B element-wise. Returns 0 if A and B match; 1 otherwise. */
int vec_not_equals(e_data *A, e_data *B) {
    for (e_uint i = 0; i < RESULT_VEC_LEN; ++i) {
        if (A[i] != B[i]) {
            return 1;
        }
    }
    return 0;
}

int test_single_chip_0() {
    printf("--------Begin %s()--------\n", __func__);
    // initialize the test results vectors
    memset(A, 0, sizeof(e_data)*RESULT_VEC_LEN);
    memset(B, 0, sizeof(e_data)*RESULT_VEC_LEN);

    for (e_uint i = 0; i < TEST_NUM_ITERS; ++i) {
        e_address address = (e_address)i;
        e_data data = (e_data) 1000 - i;

        // reference
        A[i] = data;

        // simple consistency check
        endurer_write_word(address, data);
        B[i] = endurer_read_word(address);
    }

    return vec_not_equals(A, B);
}

int test_single_chip_1() {
    printf("--------Begin %s()--------\n", __func__);
    // initialize the test results vectors
    memset(A, 0, sizeof(e_data)*RESULT_VEC_LEN);
    memset(B, 0, sizeof(e_data)*RESULT_VEC_LEN);

    // Use the first four addresses as a "scratch area,"
    // and ensure that the rest of RRAM remains consistent.

    for (e_uint i = 0; i < MEMORY; ++i) {
        endurer_write_word(i, i);
        A[i] = i;   // reference
    }

    for (e_uint i = 0; i < 4; ++i) {
        endurer_write_word(i, 99);
        A[i] = 99;
    }

    for (e_uint i = 0; i < MEMORY; ++i) {
        B[i] = endurer_read_word(i);
    }

    return vec_not_equals(A, B) || !(get_relative_shift(0)==0);
}

/*
 * Tests remap().
 */
int test_single_chip_2() {
    printf("--------Begin %s()--------\n", __func__);
    // initialize the test results vectors
    memset(A, 0, sizeof(e_data)*RESULT_VEC_LEN);
    memset(B, 0, sizeof(e_data)*RESULT_VEC_LEN);

    // Use the first four addresses as a "scratch area,"
    // and ensure that the rest of RRAM remains consistent.

    for (e_uint i = 0; i < MEMORY; ++i) {
        endurer_write_word(i, i);
        A[i] = i;   // reference
    }

    remap();

    for (e_uint i = 0; i < MEMORY; ++i) {
        B[i] = endurer_read_word(i);
    }

    return vec_not_equals(A, B) || !(get_relative_shift(0)!=0);
}

/*
 * Tests multiple remap()s.
 */
int test_single_chip_3() {
    printf("--------Begin %s()--------\n", __func__);
    // initialize the test results vectors
    memset(A, 0, sizeof(e_data)*RESULT_VEC_LEN);
    memset(B, 0, sizeof(e_data)*RESULT_VEC_LEN);

    // Use the first four addresses as a "scratch area,"
    // and ensure that the rest of RRAM remains consistent.

    for (e_uint i = 0; i < MEMORY; ++i) {
        endurer_write_word(i, i);
        A[i] = i;   // reference
    }

    remap();
    remap();
    remap();
    remap();
    remap();

    for (e_uint i = 0; i < MEMORY; ++i) {
        B[i] = endurer_read_word(i);
    }

    return vec_not_equals(A, B) || !(get_relative_shift(0)!=0);
}

int main(int argc, char *argv[]) {
    printf("Hello from ENDUReR.\n");
    enable_endurer();
    select_chip(0);

    initialize(12345);
    assert(!test_single_chip_0());
    teardown(0);

    initialize(12346);
    assert(!test_single_chip_1());
    teardown(0);

    initialize(12347);
    assert(!test_single_chip_2());
    teardown(0);

    initialize(12348);
    assert(!test_single_chip_3());
    teardown(0);

    printf("All results vectors match.\n");
    printf("Done.\n");

    return 0;
}
