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
 * This file defines a write driver for Endurer simulations
 */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "mock_board.h"
#include "endurer.h"

char filename[200] = "/pool0/zainabk/illusion_system/c_models/c_d2nn/d2nn_write_pattern.txt";

void save_memory(const char* fname) {
    FILE* final = fopen(fname,"w");

    for (int i=0; i<NUM_CHIPS; i++) {
        select_chip(i);
        for (int j1=0; j1<M/16; j1++) {
            for (int j2=0; j2<15; j2++) {
                fprintf(final, "%4x,",endurer_read_word(j1*16+j2));
            } fprintf(final, "%4x\n",endurer_read_word(j1*16+15));
        } 
    }
    fclose(final);
}

void perform_pattern_hex(const char fname[300]) {
    FILE* f  = fopen(fname, "r");
    assert(f);
    int a;
    int d;
    int c;
    while (fscanf(f, "%d,%x,%x\n",&c,&a,&d) != EOF) {
        select_chip(c);
        endurer_write_word(a, d); 
    }
    fclose(f);
}
 
void load_init(const char fname[300], int c) {
    FILE* f  = fopen(fname, "r");
    assert(f);
    int addr, data, error, i;
    select_chip(c);
    while (fscanf(f, "\n") != EOF) {
        error = fscanf(f, "@%x",&addr);
        if (error==EOF) {
            fclose(f);
            return;
        }

        for (i = 0; i<16; i++) {
            error = fscanf(f, "%x", &data);
            if (error==EOF) {
                fclose(f);
                return;
            }
            endurer_write_word(addr + i, data);
        }
    }
    fclose(f);
}
 
int main() {
    char init_filename[300];
    printf("Loading D2nn initial write (weights)\n");
    for (int i =0; i<8; i++) {
        sprintf(init_filename, "../../illusion_testing/programs/d2nn/MODEILLUSION/HIGH/NONV/NOTIME/CHIP%d/dmem.mem",i);
        printf("Load file %s\n", init_filename);
        load_init(init_filename, i); 
    }

    e_uint heavy, light;
    for (int j=0; j<NUM_CHIPS; j++) flush_and_write_back(j);

    int REMAP_PERIOD_IN_HOURS = 4;
    for (int day=0; day<10; day++) {
        int D = day; 
        //printf("Starting day %d\n", D); 
        for (int hour=0; hour<24; hour+=REMAP_PERIOD_IN_HOURS) { 
            // d2nn does it 3 times
            perform_pattern_hex(filename);
            perform_pattern_hex(filename);
            perform_pattern_hex(filename);
            #if (NO_DISTRIBUTED_ENDURER==0) 
                if (time_to_shift_chips(&heavy, &light)) {
                    distributed_remap(heavy, light);
                    printf("%d,%d\n",D,hour); 
                }
            #endif
            #if (NO_ENDURER==0) 
                if ((hour%REMAP_PERIOD_IN_HOURS==0) && !(day==0&&hour==0)) {
                    for (int j=0; j<NUM_CHIPS; j++) remap(j);
               }
            #endif
        }
    }
    save_memory("memory_d2nn.csv");

    return 0;
}
