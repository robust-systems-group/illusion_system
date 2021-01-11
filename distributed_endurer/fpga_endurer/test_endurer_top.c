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
#include "endurer_top.h"
#include "endurer.h"

    u16 dram_buf[DRAM_BUFFER_SIZE];
    
    e_uint virt_to_phys_map[8], phys_to_virt_map[8], offsets[8];
    e_count counters_[8];
    e_count swap_count;
    e_endurer_mode mode;

    double TAU_,EPS_,C_;
    
    int error;

void print_state() {
    printf("Mode: %d, Swap: %lld\n", mode, swap_count);
    printf("Virt to phys chip: ");
    for (int i=0; i<8; i++) printf("%d,",virt_to_phys_map[i]);
    printf("\nPhys to virt chip: ");
    for (int i=0; i<8; i++) printf("%d,",phys_to_virt_map[i]);
    printf("\nOffsets (indexed by virt): ");
    for (int i=0; i<8; i++) printf("%d,",offsets[i]);
    printf("\nCounters (indexed by phys): ");
    for (int i=0; i<8; i++) printf("%lld,",counters_[i]);
    printf("\n");
}
 
int main() {
    reset_endurer_state();   
    enable_endurer();
 
    for (int i=0; i<NUM_CHIPS; i++) {
        select_chip(i);
        for (int j=0; j<10; j++) {
            remap();
        } 
    }

    get_endurer_state(&mode,&swap_count,virt_to_phys_map,phys_to_virt_map,offsets,counters_,&TAU_,&EPS_,&C_);
    print_state();

    assert(TAU_==0.6);
    assert(EPS_==0.6);
    assert(C_==8e6);
    assert(swap_count==0);
    for (int i=0; i<NUM_CHIPS; i++) {
        assert(counters_[i]==10*M);
        assert(virt_to_phys_map[i]==i);
        assert(virt_to_phys_map[i]==phys_to_virt_map[i]);
        assert(offsets[i]!=0);
    }
    assert(mode==1); 

    printf("Increase last count by 1, try to do swap in only endurer mode (no change should happen)\n");
    counters_[7] += 1;
    reset_endurer_state();
    error = set_endurer_state(mode,swap_count,virt_to_phys_map,offsets,counters_,TAU_,EPS_,C_);
    distributed_remap(2,3); 
    get_endurer_state(&mode,&swap_count,virt_to_phys_map,phys_to_virt_map,offsets,counters_,&TAU_,&EPS_,&C_);

    print_state();

    assert(TAU_==0.6);
    assert(EPS_==0.6);
    assert(C_==8e6);
    assert(swap_count==0);
    assert(error==0);
    assert(counters_[7]==10*M+1);
    for (int i=0; i<NUM_CHIPS; i++) {
        if (i!=7) assert(counters_[i]==10*M);
        assert(virt_to_phys_map[i]==i);
        assert(virt_to_phys_map[i]==phys_to_virt_map[i]);
    }
 
    //enable_distributed_endurer();
    mode = (e_uint) DISTRIBUTED_ENDURER;
    printf("Increase first count by 1, try to do swap of 2 & 3\n"); 
 
    counters_[0] += 1;
    EPS_ = .8;
    reset_endurer_state();
    error = set_endurer_state(mode,swap_count,virt_to_phys_map,offsets,counters_,TAU_,EPS_,C_);
    distributed_remap(2,3);
    get_endurer_state(&mode,&swap_count,virt_to_phys_map,phys_to_virt_map,offsets,counters_,&TAU_,&EPS_,&C_);

    print_state();

    assert(TAU_==0.6);
    assert(EPS_==0.8+0.6);
    assert(C_==8e6);
    assert(swap_count==1);
    assert(error==0);
    assert(counters_[7]==10*M+1);
    assert(counters_[0]==10*M+1);
    assert(counters_[2]==11*M);
    assert(counters_[3]==11*M);
    assert(virt_to_phys_map[2]==3);
    assert(phys_to_virt_map[3]==2);
    assert(virt_to_phys_map[3]==2);
    assert(phys_to_virt_map[2]==3);
    for (int i=0; i<NUM_CHIPS; i++) {
        if (i!=0 && i!=2 && i!=3 && i!=7) assert(counters_[i]==10*M);
        if (i!=2 && i!=3) {
            assert(virt_to_phys_map[i]==i);
            assert(virt_to_phys_map[i]==phys_to_virt_map[i]);
        }
    }
    assert(mode==2);
 
    distributed_remap(3,7); 
    get_endurer_state(&mode,&swap_count,virt_to_phys_map,phys_to_virt_map,offsets,counters_,&TAU_,&EPS_,&C_);

    assert(TAU_==0.6);
    assert(EPS_==0.8+0.6+0.36);
    assert(C_==8e6);
    assert(swap_count==2);   
 
    printf("Swapped 3 & 7\n");
    print_state();
    
    return 0;
}
