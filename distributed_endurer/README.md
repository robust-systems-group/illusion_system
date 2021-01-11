# Distributed ENDURER Testing Code and Data
Directory map:
- `c_models` : golden models
- `c_models_test` : Evaluates accuracy using weight & activation masks from different time points
- `endurer_tests` : FPGA scripts 
- `fpga_endurer` : FPGA version of endurer
- `data` : Measured data reported in [1]
- `plotting` : Plotting scripts from appendix for [1]


## Code setup 
Code was run on a CentOS server with GCC 4.8:
```
gcc (GCC) 4.8.5 20150623 (Red Hat 4.8.5-39)
Copyright (C) 2015 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

```

The Python components have been tested with both Python 2.7 and Python 3.

