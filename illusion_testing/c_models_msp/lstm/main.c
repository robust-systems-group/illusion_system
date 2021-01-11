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
 *  CNN for MNIST
 *  Author:     Tony Wu
 *
 *************************************************************************
 */

//#include <stdlib.h>
//#include <stdio.h>

#include "omsp_func.h"

#include "test.h"

__asm__(".include \"boot.asm\"");
void __attribute__((optimize("O0")))  main( void ) {
    /*****/
    /* main program, corresponds to procedures        */
    /* Main and Proc_0 in the Ada version             */
    /* Initializations */
    STOP_WATCHDOG;
    //SET_PULSE;
    /***************/
    /* Start timer */
    /***************/
#ifdef TIME
    START_TIME;  // Set P3[0]
#endif
    /******* DO TEST **********/
    read_input();
    classify();
    send_output();

    /**************/
    /* Stop timer */
    /**************/
#ifdef TIME
    END_TIME;  // Clear P3[0]
#endif
    CORE_DONE; //TODO this turns of chip. 
    //DONE_TEST; 
    return; 
}

__asm__(".include \"irq_table.asm\"");

