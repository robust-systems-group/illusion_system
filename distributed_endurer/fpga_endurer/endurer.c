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
 * Use static buffers as arenas of memory that we can use to act as the RRAM+SRAM memory area.
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "endurer.h"
#include "endurer_top.h"
// TODO @Robert comment out next line for FPGA
#include "mock_board.h"

/*
 * Defines the state of the word in the SRAM cache.
 */
typedef enum {
	INVALID,
    SYNC,
    DIRTY//,
    //INVALID   // currently unused (just check if present in the map instead)
} word_state;

///* Unused */
//static void *segment_base = NULL;
//static e_uint segment_size = 0;

static e_uint relative_shift[NUM_CHIPS] = {0};

/* Maps address (e_address) to an index within SEG_SRAM. */
//static std::unordered_map<e_address, std::pair<e_uint, word_state>> sram_map;

typedef struct {
    e_address address;    // address in RRAM for this SRAM cache index
    word_state state;   // state (will be 0, i.e. INVALID, if not present)
} sram_map_entry;

typedef struct {
    u16 physical_rram_id;
    u16 segment_type;
    u16 segment_chunk;
} e_segment_map;

// relative to virtual chip id!
static sram_map_entry sram_map[NUM_CHIPS][S];
static e_uint sram_map_size[NUM_CHIPS] = {0};

static e_uint virt_chip_id = 0; // always start at virtual chip 0
static e_uint phys_chip_id = 0; // always start at physical chip 0 
static e_uint phys_to_virt_chip_id[NUM_CHIPS] = {0,1,2,3,4,5,6,7}; // always start with matching phys & virt chip ids
static e_uint virt_to_phys_chip_id[NUM_CHIPS] = {0,1,2,3,4,5,6,7}; // always start with matching phys & virt chip ids

// relative to physical chip id!
static e_segment_map segment_allocation[NUM_CHIPS] = {{0,CTRL_INSTR,0},{0,CTRL_INSTR,1},{0,CTRL_INSTR,2},{0,CTRL_DATA,0},{1,CTRL_INSTR,0},{1,CTRL_INSTR,1},{1,CTRL_INSTR,2},{1,CTRL_DATA,0}};
static e_count counters[NUM_CHIPS];
static e_uint highlighted_segment_type = 0;
static e_uint highlighted_segment_chunk = 0;


static e_endurer_mode endurer_mode = NO_RESILIENCY;

// Distributor params
static double TAU = .6;
static double EPS = .6;
static e_count DIST_REMAP_COUNTER = 1;
static double C = 8e6; 

void reset_endurer() {
    reset_endurer_state();
}

void update_epsilon() {
    EPS += pow(TAU,(double)DIST_REMAP_COUNTER++);
}

int select_segment(e_uint rram_chip, e_uint segment_type, e_uint chunk) {
    if (segment_type==CTRL_INSTR) {
        // 3 chunks in imem, plus 1 backup array possible
        if (chunk>3) return 1;
    } else if (segment_type==CTRL_DATA) {
        // 1 chunk in dmem, plus 1 backup array possible
        if (chunk>1) return 2;
    } else return 4; //invalid segment type!
    if (rram_chip>=NUM_CHIPS) return 3;
 
    select_rram_chip(rram_chip);
    
    highlighted_segment_type = segment_type;
    highlighted_segment_chunk = chunk;    
    
    return 0;
}
 
int assign_physical_segment(e_uint phys_chip, e_uint rram_chip, e_uint segment_type, e_uint chunk) {
    if (segment_type==CTRL_INSTR) {
        // only 3 chunks in imem
        if (chunk>2) return 1;
    } else if (segment_type==CTRL_DATA) {
        // only 1 chunk in dmem
        if (chunk!=0) return 2;
    } else return 4; //invalid segment type!
    if (rram_chip>=NUM_CHIPS) return 3;
        
    segment_allocation[phys_chip].physical_rram_id = rram_chip;
    segment_allocation[phys_chip].segment_type     = segment_type;
    segment_allocation[phys_chip].segment_chunk    = chunk;
    return 0;
}
 
void read_out_physical_segments() {
    for (int i=0; i<NUM_CHIPS; i++) {
        printf("%1d,%1d,%1d\n", segment_allocation[i].physical_rram_id,
                segment_allocation[i].segment_type,
                segment_allocation[i].segment_chunk);
    }
}
 
int verify_segments() {
    for (int i=0; i<NUM_CHIPS; i++) {
        for (int j=0; j<NUM_CHIPS; j++) {
            if (i==j) continue;
    
            // Cannot allocate 2 chips on same space!!        
            if ((segment_allocation[i].physical_rram_id == segment_allocation[j].physical_rram_id) &&
                (segment_allocation[i].segment_type     == segment_allocation[j].segment_type) &&
                (segment_allocation[i].segment_chunk    == segment_allocation[j].segment_chunk)) return j;
        }
    }
    return 0;
}

int reset_endurer_state() {
    virt_chip_id = 0;
    phys_chip_id = 0;
    DIST_REMAP_COUNTER = 1;
    TAU = 0.6;
    EPS = 0.6;
    C = 8e6;
    for (int i=0; i<NUM_CHIPS;i++) {
        phys_to_virt_chip_id[i] = i;
        virt_to_phys_chip_id[i] = i;
        counters[i] = 0;
        relative_shift[i] = 0;
    }
    return 0;
}

int set_endurer_state(e_uint mode, e_count swap_count, e_uint virt_to_phys_map[NUM_CHIPS], e_uint offsets[NUM_CHIPS],
                    e_count counts[NUM_CHIPS], double TAU_, double EPS_, double C_) {
    if (!(mode==NO_RESILIENCY || mode==ENDURER || mode==DISTRIBUTED_ENDURER)) {
        return 8;
    }
    endurer_mode = (e_endurer_mode) mode;
    
    TAU = TAU_;
    EPS = EPS_;
    C   =   C_;
    DIST_REMAP_COUNTER = swap_count + 1;

    if (mode==NO_RESILIENCY) {
        for (int i=0; i<NUM_CHIPS;i++) {
            if (virt_to_phys_map[i]!=i) return 1;
            virt_to_phys_chip_id[i] = i;
            phys_to_virt_chip_id[i] = i;
            counters[i] = counts[i];
            if (offsets[i]!=0) return 2;
            relative_shift[i] = 0;
        }
    } else {
        for (int i=0; i<NUM_CHIPS;i++) {
            if (virt_to_phys_map[i]<0 || virt_to_phys_map[i]>=NUM_CHIPS) return 3;
            virt_to_phys_chip_id[i] = virt_to_phys_map[i];
            counters[i] = counts[i];
            if (offsets[i]>M || offsets[i]<0) return 4;
            relative_shift[i] = offsets[i];
        }
        for (int i=0; i<NUM_CHIPS;i++) {
            for (int j=0; j<NUM_CHIPS; j++) {
                if (virt_to_phys_chip_id[j] == i) {
                    phys_to_virt_chip_id[i] = j;
                }
            }
        }
    }
 
    return 0;
}

void get_endurer_state(e_uint *mode, e_count *swap_count, e_uint* virt_to_phys_map, 
                    e_uint* phys_to_virt_map, e_uint* offsets, e_count* counts,
                    double* TAU_, double* EPS_, double* C_) {
    *mode = (e_uint) endurer_mode;
    *swap_count = DIST_REMAP_COUNTER -1;

    *TAU_ = TAU;
    *EPS_ = EPS;
    *C_   =   C;

    for (int i=0; i<NUM_CHIPS;i++) {
        virt_to_phys_map[i] = virt_to_phys_chip_id[i];
        phys_to_virt_map[i] = phys_to_virt_chip_id[i];
        counts[i] = counters[i];
        offsets[i] = relative_shift[i];
    }
}


void enable_distributed_endurer() { endurer_mode = DISTRIBUTED_ENDURER; }
void enable_endurer() { endurer_mode = ENDURER; }
void disable_endurer() { endurer_mode = NO_RESILIENCY; }

int get_swap_count() {
    return DIST_REMAP_COUNTER-1;
}

e_endurer_mode get_endurer_mode() {
    return endurer_mode;
}

e_address get_relative_shift(e_uint virt) {
    return relative_shift[virt];
}

e_address get_virtual_chip_id(e_uint phys) {
    return virt_to_phys_chip_id[phys];
}

e_address get_physical_chip_id(e_uint virt) {
    return phys_to_virt_chip_id[virt];
}

e_address get_current_physical_chip() { return phys_chip_id; }
e_address get_current_virtual_chip()  { return virt_chip_id; }

e_count get_count(e_uint phys) {
    return counters[phys];
}

static inline void sram_map_clear(e_uint i) {
    memset(sram_map[i], 0, S * sizeof(sram_map_entry));
    sram_map_size[i] = 0;
}

static inline void sram_map_clear_all() {
    for (int i = 0; i<NUM_CHIPS; i++) sram_map_clear(i);
}

static inline void add_rram_write_count() { 
    counters[phys_chip_id]++;
}

// TODO where is the write verify loop check??
// TODO check whether makes more sense to do add_rram_write once per try or what
void write_word_retry(u16 address, u16 data) {
	//int i;
	u16 r;
//		printf("Writing %x to %x\n", data, address);
		r = read_word(address + M*highlighted_segment_chunk, 
                      highlighted_segment_type);
		r = write_word(address + M*highlighted_segment_chunk, 
                      data,
                      highlighted_segment_type);
		
        add_rram_write_count();
        
		if (r){
            return;
		}

}

// TODO check why done twice
u16 read_word_helper(u16 address) {
	u16 r;
//		printf("Writing %x to %x\n", data, address);
		r = read_word(address + M*highlighted_segment_chunk, 
                      highlighted_segment_type);
		r = read_word(address + M*highlighted_segment_chunk, 
                      highlighted_segment_type);
	
	return r;
}

/*
 * On the 64-bit Linux dev environment, this will (need to) be a larger address.
 * But, on the board, we want the segment to be addressable in 16 bits, so we'll
 * try to map it to a low address.
 */
//static e_data SEG_RRAM[M];
static e_data SEG_SRAM[NUM_CHIPS][S];

/*
 * NOTE: on the board itself, exercise care that segment_size doesn't exceed the
 * amount of available memory.
 */
// TODO use rand_r() or a better PRNG
void initialize(const e_uint seed) {
    srand(seed);
}

// TODO make the chip select mux be programmed here??
/* Get physical chip id to choose which physical RRAM to enable */
e_uint select_chip(e_uint virt) {
    virt_chip_id = virt;
    phys_chip_id = virt_to_phys_chip_id[virt];
    if (segment_allocation[phys_chip_id].physical_rram_id != get_rram_chip()) {
        // mux select to other chip!! TODO
        select_rram_chip(segment_allocation[phys_chip_id].physical_rram_id); 
    }

    highlighted_segment_type = segment_allocation[phys_chip_id].segment_type;
    highlighted_segment_chunk = segment_allocation[phys_chip_id].segment_chunk;    

    return phys_chip_id;
}

/*
 * Internal functions to map virtual (user-space) addresses to
 * physical (RRAM segment) addresses. We need to do a virtual-
 * to-physical translation since physical addresses will change
 * when shifting occurs.
 */
//TODO note + and - switch for FWD-shift loop vs BWD-shift loop
//(just follow the direction of the shift. data goes back, address goes back)
static inline e_address physical_to_virtual(e_address phys) {
    int shift = relative_shift[virt_chip_id];
    return (phys + shift) % M;
}

/*
 * Note that, to deal with the fact that (virt - relative_shift) may underflow,
 * we add an extra M before subtracting (and it works because (M + x) % M = x).
 */
static inline e_address virtual_to_physical(e_address virt) {
    int shift = relative_shift[virt_chip_id];
    return (virt + M - shift) % M;
}

void write_back(e_uint virt) {
	e_address address, index;
    select_chip(virt);
    for (index = 0; index < S; ++index) { // iterate through the whole sram_map table
        address = sram_map[virt_chip_id][index].address;
        word_state state = sram_map[virt_chip_id][index].state;

        if (state == DIRTY) {
        	write_word_retry(address, SEG_SRAM[virt_chip_id][index]);
        }
    }
}

/* Flushes the SRAM cache, writing back only those words with DIRTY word_state in the cache. */
void flush_and_write_back(e_uint virt) {
//    printf("Flushing and writing back dirty words...\n");

    write_back(virt);
    sram_map_clear(virt);
}

// only used for chip swaps
// pre-allocate space for it just cuz 
static e_data chip0[M];
static e_data chip1[M];

/* Moves data/id from most written to chip to least written to chip
 * Then updates the chip id memory map accordingly
 * Clears all SRAM buffers in process
 */
void distributed_remap(e_uint physical_worst, e_uint physical_best) {
    if (endurer_mode!=DISTRIBUTED_ENDURER) return;

 
    //printf("DID DISTRIBUTED REMAP between %d and %d\n", physical_worst, physical_best);

    int virt_worst = phys_to_virt_chip_id[physical_worst];
    int virt_best = phys_to_virt_chip_id[physical_best]; 

    flush_and_write_back(virt_worst);
    flush_and_write_back(virt_best);
    
    select_chip(virt_worst);
    for (int i =0; i<M; i++) {
        chip0[i] = read_word_helper(i);
    }
    select_chip(virt_best);
    for (int i =0; i<M; i++) {
        chip1[i] = read_word_helper(i);
        write_word_retry(i, chip0[i]);
    }
    select_chip(virt_worst);
    for (int i =0; i<M; i++) {
        write_word_retry(i, chip1[i]);
    }

    // Updated virtual chip id maps (virt->phys)

    phys_to_virt_chip_id[physical_worst] = virt_best;
    phys_to_virt_chip_id[physical_best ] = virt_worst;
    virt_to_phys_chip_id[virt_best ] = physical_worst;
    virt_to_phys_chip_id[virt_worst] = physical_best;

    update_epsilon();
}


/*
 * Performs a random-shift remapping.
 * Updates relative_shift so that virtual-physical translation can work.
 */
void remap() {
	if (endurer_mode==NO_RESILIENCY) return;

    e_uint shift = rand() % (M-1);
	relative_shift[virt_chip_id] = (relative_shift[virt_chip_id] + shift) % M; // for virtual-physical mapping
	e_uint num_shift_loops = 1 << __builtin_ctz(shift);
	e_uint shifts_per_loop = M / num_shift_loops; // requires power-of-two-sized segment

	/* Not strictly necessary, but simplifies things. */
	flush_and_write_back(virt_chip_id);

    for (e_uint i = 0; i < num_shift_loops; ++i) {
		e_data tmp = read_word_helper(i); // cache the first element before clobbering it
        e_uint dist = 0;
        for (e_uint j = 0; j < shifts_per_loop - 1; ++j) {
            e_address curr_addr = (i + dist) % M;
            e_address next_addr = (curr_addr + shift) % M;

            e_data tmp2 = read_word_helper( next_addr);
            write_word_retry(curr_addr, tmp2);

            dist += shift;
        }

        e_address last_addr = (i + dist) % M;
        write_word_retry(last_addr, tmp);
    }

}

e_data endurer_read_word(const e_address virt) {
    e_address address = virtual_to_physical(virt);
    //printf("read_word at virtual [%x] (physical [%x])\n", virt, address);
    if (endurer_mode==NO_RESILIENCY) {
        return read_word_helper(virt);
    }
    else {
        for (e_address idx=0; idx<sram_map_size[virt_chip_id]; idx++) {
            if (sram_map[virt_chip_id][idx].address == address &&
                sram_map[virt_chip_id][idx].state != INVALID) {
                // cache hit
                return SEG_SRAM[virt_chip_id][idx];
            }
        }
        // cache miss. go to RRAM, return
        e_data value = read_word_helper(address);
        return value;
    } 
}

/*
 * Make sure you explicitly manage the SRAM cache.
 */
e_uint endurer_write_word(const e_address virt, const e_data data) {
    e_address address = virtual_to_physical(virt);
    //printf("write_word %x at virtual [%x] (physical [%x])\n", data, virt, address);

    // If it's in the cache already, update it.
    // If it's not, then put it in
    //      1.) check if full, flush-and-wb if so
    //      2.) if not full, just put in cache and indicate maybe-dirty
    // If it's in the cache already, update it.
    // If it's not, then put it in
    //      1.) check if full, flush-and-wb if so
    //      2.) if not full, just put in cache and indicate maybe-dirty
    if (endurer_mode==NO_RESILIENCY) {
        write_word_retry(virt, data);
    }
    else {
        for (e_address idx=0; idx<sram_map_size[virt_chip_id]; idx++) {
            if (sram_map[virt_chip_id][idx].address == address &&
                sram_map[virt_chip_id][idx].state != INVALID) {
                SEG_SRAM[virt_chip_id][idx] = data;
                // cache hit
                return data;
            }
        }
        // cache miss
        
        // cache full
        if (sram_map_size[virt_chip_id] == S) {
            flush_and_write_back(virt_chip_id);
        }
        SEG_SRAM[virt_chip_id][sram_map_size[virt_chip_id]] = data;
        sram_map[virt_chip_id][sram_map_size[virt_chip_id]].address = address; sram_map[virt_chip_id][sram_map_size[virt_chip_id]].state = DIRTY; ++sram_map_size[virt_chip_id];
//        printf("Word now in SRAM\n");
    }

    return data;


}

e_uint time_to_shift_chips(e_uint *heavy, e_uint *light) {
    if (endurer_mode!=DISTRIBUTED_ENDURER) return 0;
    long long tot = 0;
    *heavy = NUM_CHIPS;
    *light = 0;
    for (int i=0; i<NUM_CHIPS; i++) {
        tot += counters[i];
    }
    for (int i=0; i<NUM_CHIPS; i++) {
        if (counters[i] > ((double)tot/NUM_CHIPS*(1 + EPS) + C)) {
            if (*heavy==NUM_CHIPS) *heavy = i;
            else if (counters[i] > counters[*heavy]) *heavy = i;
        } if (counters[i] < counters[*light]) *light = i;
    }
    return (*heavy!=NUM_CHIPS); 
} 

e_uint teardown_all() {
    sram_map_clear_all();

    // zero buffers for good measure
    memset(SEG_SRAM, 0, NUM_CHIPS * S * sizeof(e_data));
    return 0;
}

e_uint teardown(e_uint i) {
    sram_map_clear(i);

    // zero buffers for good measure
    memset(SEG_SRAM + i * S * sizeof(e_data), 0, S * sizeof(e_data));
    return 0;
}
