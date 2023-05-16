# coding=utf-8
from core.factors.binary_class import BinaryClassObservationFactor
from core.inference.gbp.layers.softmax_class import GBPSoftmaxClassObservationLayer
from core.variables import CoeffVariable, PixelVariable


class GBPBinaryClassObservationLayer(GBPSoftmaxClassObservationLayer):
    def __init__(self,
                 binary_factor: BinaryClassObservationFactor,
                 input_vars: [CoeffVariable, PixelVariable]):
        # No output/coeff variables
        super(GBPBinaryClassObservationLayer, self).__init__(input_vars=input_vars,
                                                             softmax_factor=binary_factor)
