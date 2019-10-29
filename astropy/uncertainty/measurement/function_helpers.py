# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Numpy function helpers fo the Measurement class.
"""

import numpy as np

from .core import Measurement
from .helpers import chain_derivatives
from .uncertainty import DerivedUncertainty


FUNCTION_HELPERS = {}


def concatenate(arrays, axis=0, out=None):
    if out is not None:
        return NotImplemented

    nominals = [(array.nominal if isinstance(array, Measurement) else array)
                for array in arrays]
    nominal_result = np.concatenate(nominals, axis=axis)
    derivatives = []
    parts = []
    for array in arrays:
        shape = [1] * nominal_result.ndim
        shape[axis] = array.shape[axis]
        parts.append(np.zeros(shape))

    for array, part in zip(arrays, parts):
        if isinstance(array, Measurement):
            part[:] = 1.
            derivative = np.concatenate(parts, axis=axis)
            part[:] = 0.
        else:
            derivative = None

        derivatives.append(derivative)

    new_derivs = chain_derivatives(arrays, derivatives)
    result = Measurement(nominal_result)
    result._uncertainty = DerivedUncertainty(new_derivs)
    return result


FUNCTION_HELPERS[np.concatenate] = concatenate
