#
# Copyright (C) 2020 by The Board of Trustees of Stanford University
# This program is free software: you can redistribute it and/or modify it under
# the terms of the Modified BSD-3 License as published by the Open Source
# Initiative.
# If you use this program in your research, we request that you reference the
# Illusion paper, and that you send us a citation of your work.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the BSD-3 License for more details.
# You should have received a copy of the Modified BSD-3 License along with this
# program. If not, see <https://opensource.org/licenses/BSD-3-Clause>.
#


gcc -lm main.c tensor.c -o  d2nn_inference
./d2nn_inference 0 0 1 > mode_target_io_low.txt
./d2nn_inference 0 0 2 > mode_illusion_sm_io_low.txt
./d2nn_inference 0 0 3 > mode_illusion_io_low.txt
./d2nn_inference 17 17 1 > mode_target_io_high.txt
./d2nn_inference 17 17 2 > mode_illusion_sm_io_high.txt
./d2nn_inference 17 17 3 > mode_illusion_io_high.txt
