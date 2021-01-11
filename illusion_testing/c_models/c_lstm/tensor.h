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


#ifndef TENSOR_H
#define TENSOR_H

#include <stdint.h>

#define max(a,b) ((a > b) ? (a) : (b))
#define min(a,b) ((a < b) ? (a) : (b))

#define SEQLEN 40
#define HDIM 56
#define IDIM 10
#define ODIM 11

//void lstm(int16_t *I, int16_t *H, int16_t *C,  const int16_t *W_IH, const int16_t *W_HH, const int16_t *B, int Z, int16_t *H_O, int16_t *C_O);
void lstm(int16_t *I, int16_t *H, int32_t *C,  const int16_t *W_IH, const int16_t *W_HH, const int16_t *B, int Z, int16_t *H_O, int32_t *C_O);

void dense(int16_t *A, const int16_t *H, int32_t *C, int V, int Z);
void add_bias(const int16_t *B, int32_t *C, int32_t *CO, int Z);
void dense2(int16_t *A, const int16_t *H,  const int16_t *B, int32_t *C, int V, int Z);

#endif
