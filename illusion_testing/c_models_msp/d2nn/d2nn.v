/*===========================================================================*/
/* Copyright (C) 2001 Authors                                                */
/*                                                                           */
/* This source file may be used and distributed without restriction provided */
/* that this copyright statement is not removed from the file and that any   */
/* derivative work contains the original copyright notice and the associated */
/* disclaimer.                                                               */
/*                                                                           */
/* This source file is free software; you can redistribute it and/or modify  */
/* it under the terms of the GNU Lesser General Public License as published  */
/* by the Free Software Foundation; either version 2.1 of the License, or    */
/* (at your option) any later version.                                       */
/*                                                                           */
/* This source is distributed in the hope that it will be useful, but WITHOUT*/
/* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or     */
/* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public       */
/* License for more details.                                                 */
/*                                                                           */
/* You should have received a copy of the GNU Lesser General Public License  */
/* along with this source; if not, write to the Free Software Foundation,    */
/* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA        */
/*                                                                           */
/*===========================================================================*/
/*                                 SANDBOX                                   */
/*---------------------------------------------------------------------------*/
/*                                                                           */
/* Author(s):                                                                */
/*             - Olivier Girard,    olgirard@gmail.com                       */
/*                                                                           */
/*---------------------------------------------------------------------------*/
/* $Rev: 19 $                                                                */
/* $LastChangedBy: olivier.girard $                                          */
/* $LastChangedDate: 2009-08-04 23:47:15 +0200 (Tue, 04 Aug 2009) $          */
/*===========================================================================*/
`define NO_TIMEOUT

`ifdef MODEILLUSION
    `ifdef CHIP00
        `ifdef NV
            `define IN "chip_00_in_NV.txt"
            `define OUT "chip_00_out_NV.txt"
        `endif
    `elsif CHIP0
        `ifdef NV
            `define OUT "chip_0_out_NV.txt"
            `define IN "chip_0_in_NV.txt"
        `else
            `define OUT "chip_0_out.txt"
            `define IN "chip_0_in.txt"
        `endif
    `elsif CHIP1 
        `ifdef NV
            `define IN "chip_1_in_NV.txt"
            `define OUT "chip_1_out_NV.txt"
        `else
            `define IN "chip_1_in.txt"
            `define OUT "chip_1_out.txt"
        `endif
    `elsif CHIP2 
    `define IN "chip_2_in.txt"
    `define OUT "chip_2_out.txt"
    `elsif CHIP3 
    `define IN "chip_3_in.txt"
    `define OUT "chip_3_out.txt"
    `elsif CHIP4 
    `define IN "chip_4_in.txt"
    `define OUT "chip_4_out.txt"
    `elsif CHIP5 
    `define IN "chip_5_in.txt"
    `define OUT "chip_5_out.txt"
    `elsif CHIP6
    `define IN "chip_6_in.txt"
    `define OUT "chip_6_out.txt"
    `elsif CHIP7 
    `define IN "chip_7_in.txt"
    `define OUT "chip_7_out.txt"
    `endif
`else
    `ifdef CHIP0 
    `define IN "chip_0_in.txt"
    `define OUT "chip_0_out.txt"
    `elsif CHIP1 
    `define IN "chip_1_in.txt"
    `define OUT "chip_1_out.txt"
    `elsif CHIP2 
    `define IN "chip_2_in.txt"
    `define OUT "chip_2_out.txt"
    `elsif CHIP3 
    `define IN "chip_3_in.txt"
    `define OUT "chip_3_out.txt"

    `endif
`endif

`ifdef HIGH
    `ifdef MODEILLUSION
    `define HEADER(file) {"/scratch0/radway/test_simulator/src-c/final_d2nn_combo/mode_illusion_io_high/",file}
    `elsif MODEILLUSIONSM
    `define HEADER(file) {"/scratch0/radway/test_simulator/src-c/final_d2nn_combo/mode_illusion_sm_io_high/",file}
    `elsif MODETARGET
    `define HEADER(file) {"/scratch0/radway/test_simulator/src-c/final_d2nn_combo/mode_target_io_high/",file}
    `endif
`elsif LOW
    `ifdef MODEILLUSION
    `define HEADER(file) {"/scratch0/radway/test_simulator/src-c/final_d2nn_combo/mode_illusion_io_low/",file}
    `elsif MODEILLUSIONSM
    `define HEADER(file) {"/scratch0/radway/test_simulator/src-c/final_d2nn_combo/mode_illusion_sm_io_low/",file}
    `elsif MODETARGET
    `define HEADER(file) {"/scratch0/radway/test_simulator/src-c/final_d2nn_combo/mode_target_io_low/",file}
    `endif
`endif

time clk_start_time, clk_end_time;
real clk_period,     clk_frequency;

time start_time, end_time;

integer dmem_trace;
integer pmem_trace;

wire sleep = dut.pg_mem;
reg[15:0] gt_next;  
initial
   begin
      $display(" ===============================================");
      $display("|                 START SIMULATION              |");
      $display(" ===============================================");
      repeat(5) @(posedge clk);
      stimulus_done = 0;

      //---------------------------------------
      // Check CPU configuration
      //---------------------------------------

      if ((`PMEM_SIZE !== 12288) || (`DMEM_SIZE !== 4096))
        begin
           $display(" ===============================================");
           $display("|               SIMULATION ERROR                |");
           $display("|                                               |");
           $display("|  Core must be configured for:                 |");
           $display("|               - 12kB program memory           |");
           $display("|               - 4kB data memory              |");
           $display(" ===============================================");
//           $finish;        
        end

      //---------------------------------------
      // Measure clock period
      //---------------------------------------
      repeat(100) @(posedge clk);
      $timeformat(-3, 3, " ms", 10);
      @(posedge clk);
      clk_start_time = $time;
      @(posedge clk);
      clk_end_time = $time;
      @(posedge clk);
      clk_period    = clk_end_time-clk_start_time;
      clk_frequency = 1000/clk_period;
      $display("\nINFO-VERILOG: openMSP430 System clock frequency %f MHz\n", clk_frequency);
      $display("\nINFO-VERILOG: CPU version %d \n", dbg_cpu_version);

      //---------------------------------------
      // Wait for the end of C-code execution
      //---------------------------------------
      @(posedge p4_dout[0] or posedge sleep);
 
    stimulus_done = 1;
    #1;
	$fclose(dmem_trace);
	$fclose(pmem_trace);
      $display(" ===============================================");
      $display("|               SIMULATION DONE                 |");
      $display("|       (stopped through verilog stimulus)      |");
      $display(" ===============================================");
      $finish;

   end

initial begin
      //---------------------------------------
      // Measure Test run time
      //---------------------------------------
     @(posedge p3_dout[0]);
      start_time = $time;
      $timeformat(-3, 3, " ms", 10);
      $display("\nINFO-VERILOG: Test started at %t ", start_time);
 	

      // Detect end of run
      @(negedge p3_dout[0]);
      end_time = $time;
      $timeformat(-3, 3, " ms", 10);
      $display("INFO-VERILOG: Test ended   at %t ",   end_time);

end
integer counter;
initial counter = 0;
always @(negedge clk) begin
    counter <= counter + 1;
    if (counter > 50000) begin
      $display("Time %t ",   $time);
      counter <= 0;
      end
    else counter <= counter + 1;
end

// Sensor input
integer fh;
initial begin
	fh = $fopen(`HEADER(`IN), "r");
	//#1; // Overide first sensor input
	//$fscanf(fh, "%d\n", sensor_data_next);
end
integer fh2;
initial begin
	fh2 = $fopen(`HEADER(`OUT), "r");
	//#1; // Overide first sensor input
    $fscanf(fh2, "%04x\n", gt_next);
end

always @(negedge clk) begin
        if (per_en &(!per_we) && (per_addr == (14'h0040 >> 1))) begin
                 $fscanf(fh, "%04x\n", sensor_data_next);
                 $display("FIFO Sent %x", sensor_data_next);
        end
        if (per_en & per_we && (per_addr == (14'h0040 >> 1))) begin
                 $display("FIFO Recieved %x", per_din);
                 if (per_din != gt_next) begin
                    $display("Data Missmatch %x:%x", gt_next, per_din);
                 end
                 #1 $fscanf(fh2, "%04x\n", gt_next);
        end
end


// Display stuff from the C-program
always @(p2_dout[0])
  begin
     $display("C: char=%s dec=%d hex=%h", p1_dout, p1_dout, p1_dout);
  end

// Record memory traces

initial begin
	dmem_trace = $fopen("dmem.trace", "w");
	pmem_trace = $fopen("pmem.trace", "w");
    //$timeformat(-9, 1, " ns", 10);
end

wire dmem_clk, pmem_clk;
wire dmem_cen, pmem_cen;
wire dmem_wen, pmem_wen;
wire [12:0] pmem_addr;
wire [10:0] dmem_addr;
wire [15:0] dmem_dout, pmem_dout, dmem_din, pmem_din;
/*
assign dmem_cen = dut.ms.DATA.rram_ctrl.EN;
assign pmem_cen = dut.ms.INSTR.rram_ctrl.EN;
assign dmem_wen = dut.ms.DATA.rram_ctrl.WE;
assign pmem_wen = dut.ms.INSTR.rram_ctrl.WE;
assign dmem_addr = dut.ms.DATA.rram_ctrl.A;
assign pmem_addr = dut.ms.INSTR.rram_ctrl.A;
assign dmem_dout = dut.ms.DATA.rram_ctrl.DO;
assign pmem_dout = dut.ms.INSTR.rram_ctrl.DO;
assign dmem_din = dut.ms.DATA.rram_ctrl.DI;
assign pmem_din = dut.ms.INSTR.rram_ctrl.DI;
assign dmem_clk = dut.ms.DATA.rram_ctrl.CK;
assign pmem_clk = dut.ms.INSTR.rram_ctrl.CK;
assign dmem_loading = dut.ms.DATA.loading;
assign pmem_loading = dut.ms.INSTR.loading;

always @(negedge pgr_clk) begin
	if (dmem_cen & !dmem_loading) begin
		if (!dmem_wen) $fwrite(dmem_trace, "%t:0%h%h\n", $time, dmem_addr, dmem_dout);
		else $fwrite(dmem_trace, "%t:1%h%h\n", $time, dmem_addr, dmem_din);
	end
end

always @(posedge pmem_clk) begin
	if (pmem_cen & !pmem_loading) begin
		if (!pmem_wen) $fwrite(pmem_trace, "%t:0%h%h\n", $time, pmem_addr, pmem_dout);
		else $fwrite(pmem_trace, "%t:1%h%h\n", $time, pmem_addr, pmem_din);
	end
end
*/
