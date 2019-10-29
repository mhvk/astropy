# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Measurement class and associated machinery.
"""
import numpy as np

from .uncertainty import Uncertainty, DerivedUncertainty
from .helpers import UFUNC_DERIVATIVES, chain_derivatives


__all__ = ['Measurement']


class Measurement(np.ndarray):
    """Measured value with an associated, possibly derived uncertainty.

    Parameters
    ----------
    value : array-like
        The nominal value of the measurement.
    uncertainty : array-like
        The associated measurement uncertainty.  This is assumed to have no
        dependences on any other measurements.
    copy : bool
        Whether or not to copy the input data.  Default: `True`.
    """
    _generated_subclasses = {}
    _nominal_cls = _uncertainty_cls = np.ndarray
    _uncertainty = None

    def __init_subclass__(cls, **kwargs):
        subclass = cls.__mro__[1]
        super().__init_subclass__(**kwargs)
        # If not explicitly defined for this class, use defaults.
        # (TODO: metaclass better in this case?)
        if '_nominal_cls' not in cls.__dict__:
            cls._nominal_cls = subclass
        if '_uncertainty_cls' not in cls.__dict__:
            cls._uncertainty_cls = cls._nominal_cls
        cls._generated_subclasses[subclass] = cls

    def __new__(cls, value, uncertainty=None, copy=True):
        value = np.array(value, dtype=float, subok=True, copy=copy)
        subclass = cls._get_subclass(value.__class__)
        self = value.view(subclass)
        if uncertainty is not None:
            uncertainty = np.array(uncertainty, dtype=float, subok=True,
                                   copy=copy)
        self._set_uncertainty(uncertainty)
        return self

    def _set_uncertainty(self, uncertainty):
        if uncertainty is None:
            uncertainty = np.zeros(self.shape, dtype=float)
        elif type(uncertainty) is not np.ndarray:
            raise ValueError('uncertainty should be plain ndarray')
        elif uncertainty.shape != self.shape:
            uncertainty = np.broadcast_to(uncertainty, self.shape).copy()
        self._uncertainty = Uncertainty(uncertainty)

    @classmethod
    def _get_subclass(cls, subclass):
        if subclass is np.ndarray:
            return Measurement
        if subclass is None:
            return cls
        if issubclass(subclass, Measurement):
            return subclass
        if not issubclass(subclass, np.ndarray):
            raise ValueError('can only pass in an ndarray subtype.')

        measurement_subclass = cls._generated_subclasses.get(subclass)
        if measurement_subclass is None:
            # Create (and therefore register) new Measurement subclass.
            new_name = subclass.__name__ + 'Measurement'
            # Walk through MRO and find closest generated class
            # (e.g., MeasurementQuantity for Angle).
            for mro_item in subclass.__mro__:
                base_cls = cls._generated_subclasses.get(mro_item)
                if base_cls is not None:
                    break
            else:
                base_cls = Measurement
            measurement_subclass = type(new_name, (subclass, base_cls), {})

        return measurement_subclass

    def view(self, dtype=None, type=None):
        if type is None and issubclass(dtype, np.ndarray):
            type = dtype
            dtype = None
        elif dtype is not None:
            raise ValueError('Measurement cannot be viewed with new dtype.')
        return super().view(self._get_subclass(type))

    @property
    def nominal(self):
        return super().view(self._nominal_cls)

    @property
    def uncertainty(self):
        """Uncertainty associated with the measurement.

        Returns either the measurement uncertainty, or the uncertainty derived
        by propagating the uncertainties of the underlying measurements.
        """
        uncertainty = np.broadcast_to(self._uncertainty(), self.shape)
        uncertainty = uncertainty.view(self._uncertainty_cls)
        if callable(uncertainty.__array_finalize__):
            uncertainty.__array_finalize__(self)
        return uncertainty

    def __getitem__(self, item):
        result = super().__getitem__(item)
        if not isinstance(result, self.__class__):
            result = result[...].view(self.__class__)
        result._uncertainty = self._uncertainty[item]
        return result

    def __array_finalize__(self, obj):
        if obj is None or obj.__class__ is np.ndarray:
            return

        # Check whether super().__array_finalize should be called
        # (sadly, ndarray.__array_finalize__ is None; we cannot be sure
        # what is above us).
        super_array_finalize = super().__array_finalize__
        if super_array_finalize is not None:
            super_array_finalize(obj)

        if self._uncertainty is None:
            self._uncertainty = getattr(obj, '_uncertainty', None)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Evaluate a function on the nominal value, deriving its uncertainty.

        The derivation is done by error propagation, using the derivatives of
        the given function.
        """
        # TODO: this is too general; need to allow bool type, etc.!
        if ufunc not in UFUNC_DERIVATIVES:
            if ufunc in (np.equal, np.not_equal):
                # TODO: maybe makes more sense to return Masked array?
                diff = np.subtract(*inputs)
                result = ufunc(diff.nominal, 0., **kwargs)
                result_unc = ufunc(diff.uncertainty, 0.)
                if ufunc is np.equal:
                    result &= result_unc
                else:
                    result |= result_unc
                return result

            return NotImplemented
        # No support yet for ufunc methods (at, reduce, reduceat, outer, etc.).
        assert method == '__call__'
        # No support yet for output arguments.
        assert 'out' not in kwargs
        # Apply the function to the nominal values.
        values = [(arg.nominal if isinstance(arg, Measurement) else arg)
                  for arg in inputs]
        value = super().__array_ufunc__(ufunc, method, *values, **kwargs)

        # Set up output as a Measurement that contains a derived uncertainty.
        if not isinstance(value, np.ndarray):
            value = np.array(value)

        # TODO: probably better to instantiate Measurand class here.
        result = value.view(type(self))
        derivatives = chain_derivatives(ufunc, inputs, values)
        result._uncertainty = DerivedUncertainty(derivatives)
        return result

    def __array_function__(self, function, types, args, kwargs):
        if function is np.array2string:
            # Complete hack.
            if self.shape == ():
                return str(self)
            kwargs.setdefault('formatter',
                              {'all': Measurement.__str__})

        return super().__array_function__(function, types, args, kwargs)

    def copy(self):
        copy = self.nominal.copy().view(type(self))
        copy._uncertainty = DerivedUncertainty(self._uncertainty.derivatives)
        return copy

    def __str__(self):
        return '{0}±{1}'.format(self.nominal, self.uncertainty)

    def __repr__(self):
        return '{0}(value={1}, uncertainty={2})'.format(
            type(self).__name__, self.nominal, self.uncertainty)
