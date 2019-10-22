# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Measurement subclasses for which special treatment is helpful.
"""
import numpy as np

from astropy.units import Quantity
from astropy.constants import Constant

from .core import Measurement
from .uncertainty import Uncertainty


class QuantityMeasurement(Quantity, Measurement):
    # Define subclass just so that one can pass in uncertainty with units.
    def _set_uncertainty(self, uncertainty):
        uncertainty = self._to_own_unit(uncertainty)
        super()._set_uncertainty(uncertainty)


class ConstantMeasurement(Constant, Measurement):
    # Constant uses uncertainty and _uncertainty as well, but not
    # quite in the same way.  Fortunately, we never get back to this
    # class through a view, so we can be quite sloppy.

    # TODO: Let Constant use Measurement inside?

    _uncertainty_cls = Quantity

    def _set_uncertainty(self, uncertainty):
        raise TypeError('cannot set uncertainty on a ConstantMeasurement.')

    def __array_finalize__(self, obj):
        if obj is None or obj.__class__ is np.ndarray:
            return
        super().__array_finalize__(obj)
        # Correct _uncertainty from float to Uncertainty instance.
        self._uncertainty = Uncertainty(np.array(self._uncertainty))

    @property
    def uncertainty(self):
        # Override Constant.property
        return super(Constant, self).uncertainty

    @property
    def nominal(self):
        nominal = super().nominal
        # unarrayify _uncertainty
        nominal._uncertainty = nominal._uncertainty().item()
        return nominal
