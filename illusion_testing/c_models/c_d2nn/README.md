# Source code for D2NN

Directory map:
- `model.c` : Quantized weights given in C structs
- `tensor.c` : Tensor operations for CNN inference
- `main.c` : Main model inference execution with accuracy evaluation and IO printout generation
- `data.c` : Quantized input data

To run:
`./compile_run.sh [INPUT_START] [INPUT_END] [PRINT_MODE]`

Print mode listed in main.c
