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
VGGNet-16

Simonyan and Zisserman, 2014
'''

from collections import OrderedDict

from nn_dataflow import Layer, FCLayer

LAYERS = OrderedDict()

LAYERS['conv01'] = Layer(3, 64, 224, 3)
LAYERS['conv02'] = Layer(64, 64, 224, 3)

LAYERS['conv03'] = Layer(64, 128, 112, 3)
LAYERS['conv04'] = Layer(128, 128, 112, 3)

LAYERS['conv05'] = Layer(128, 256, 56, 3)
LAYERS['conv06'] = Layer(256, 256, 56, 3)
LAYERS['conv07'] = Layer(256, 256, 56, 3)

LAYERS['conv08'] = Layer(256, 512, 28, 3)
LAYERS['conv09'] = Layer(512, 512, 28, 3)
LAYERS['conv10'] = Layer(512, 512, 28, 3)

LAYERS['conv11'] = Layer(512, 512, 14, 3)
LAYERS['conv12'] = Layer(512, 512, 14, 3)
LAYERS['conv13'] = Layer(512, 512, 14, 3)

LAYERS['fc1'] = FCLayer(512, 4096, 7)
LAYERS['fc2'] = FCLayer(4096, 4096, 1)
LAYERS['fc3'] = FCLayer(4096, 1000, 1)

