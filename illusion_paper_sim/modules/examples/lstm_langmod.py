""" $lic$
Copyright (C) 2016-2020 by The Board of Trustees of Stanford University

This program is free software: you can redistribute it and/or modify it under
the terms of the Modified BSD-3 License as published by the Open Source
Initiative.

If you use this program in your research, we request that you reference the
TETRIS paper ("TETRIS: Scalable and Efficient Neural Network Acceleration with
3D Memory", in ASPLOS'17. April, 2017), and that you send us a citation of your
work.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the BSD-3 License for more details.

You should have received a copy of the Modified BSD-3 License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>.
"""

'''
LSTM Language Model 1 Billion
'''

from collections import OrderedDict

from nn_dataflow import Layer, FCLayer

LAYERS = OrderedDict()

LAYERS['embed1'] = FCLayer(1, 793470, 1)
LAYERS['act_1'] = FCLayer(2048, 32768, 1)
LAYERS['proj_1'] = FCLayer(8192, 1024, 1) 
