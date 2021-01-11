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


from qtorch import FixedPoint, FloatingPoint
from qtorch.quant import Quantizer, quantizer
from qtorch.optim import OptimLP

ubit_8 = FixedPoint(8,7)
ubit_8a = FixedPoint(8,4)
ubit_16 = FixedPoint(16,11)

#ubit_16 = FixedPoint(16,12)
#ubit_32 = FixedPoint(32,24)

weight_quant = quantizer(forward_number=ubit_8,
                        forward_rounding="nearest")

act_error_quant = lambda : Quantizer(forward_number=ubit_8a, backward_number=None,
                        forward_rounding="nearest", backward_rounding="stochastic")

act2_error_quant = lambda : Quantizer(forward_number=ubit_16, backward_number=None,
                        forward_rounding="nearest", backward_rounding="stochastic")
