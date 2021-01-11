# Source code for LSTM

Modified from the golden model to appropriately use the hardware and provide message passing. 

See makefile for example compilation and possible compilation flags.

TIME: If set, include code for RLT timing. Do not use in real operatoin
MODETARGET: If set, compile to single ideal chip (with weight overlapping)
MODEILLUSIONSM: If set, compile to 4 chip illusion system (with weight overlapping)
MODEILLUSION: If set, compile to full 8 chip Illusion system
NV: If set, compile to write to the non-volatile memory (used for endurance trace generation)

CHIP#: Set to compile the code for a given chip number

Directory map:
- `model.c` : Quantized weights given in C structs
- `tensor.c` : Tensor operations for CNN inference
- `test.c` : Main model inference execution with message passing
- `main.c` : Wakeup and self-shutdown handling
- `makefile` : Example compilation flow
