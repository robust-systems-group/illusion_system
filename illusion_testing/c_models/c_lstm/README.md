# Source code for LSTM

Directory map:
- `model.c` : Quantized weights given in C structs
- `model_chunked_LSTM.c` : Quantized weights given in C structs, needed index swap for Illusion partitioning, so testing chunking
- `tensor.c` : Tensor operations for CNN inference
- `main.c` : Main model inference execution with accuracy evaluation and IO printout generation
- `data.c` : Quantized input data
- `chunk_model.py` : Create manipulated C struct with correct indexing for Illusion partition

To run:
`./compile_run.sh [NUMBER_INPUTS] [PRINT_MODE]`

Print mode listed in main.c
