# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Measurement class and associated machinery.
"""
import numpy as np

from astropy.units import Quantity
from astropy.constants import Constant

from .uncertainty import Uncertainty, DerivedUncertainty


__all__ = ['Measurement']


# Derivatives of ufuncs relative to their input(s).
UFUNC_DERIVATIVES = {
    np.add: (lambda x, y: 1.,
             lambda x, y: 1.),
    np.subtract: (lambda x, y: 1.,
                  lambda x, y: -1.),
    np.multiply: (lambda x, y: y,
                  lambda x, y: x),
    np.true_divide: (lambda x, y: 1/y,
                     lambda x, y: -x/y**2),
    np.arccos: lambda x: -1/np.sqrt(1-x**2),
    np.arccosh: lambda x: 1/np.sqrt(x**2-1),
    np.arcsin: lambda x: 1/np.sqrt(1-x**2),
    np.arcsinh: lambda x: 1/np.sqrt(1+x**2),
    np.arctan: lambda x: 1/(1+x**2),
    np.arctan2: (lambda y, x: x/(x**2+y**2),
                 lambda y, x: -y/(x**2+y**2)),
    np.arctanh: lambda x: 1/(1-x**2),
    np.copysign: (lambda x, y: np.where(x >= 0, np.copysign(1, y),
                                        -np.copysign(1, y)),
                  lambda x, y: 0),
    np.cos: lambda x: -np.sin(x),
    np.cosh: np.sinh,
    np.degrees: lambda x: np.degrees(1.),
    np.exp: np.exp,
    np.expm1: np.exp,
    np.fabs: lambda x: np.where(x >= 0., 1., -1.),
    np.hypot: (lambda x, y: x / np.hypot(x, y),
               lambda x, y: y / np.hypot(x, y)),
    np.log: lambda x: 1/x,
    np.log10: lambda x: 1/x/np.log(10),
    np.log1p: lambda x: 1/(1+x),
    np.power: (lambda x, y: x**(y-1) * y,   # explicitly zero y==0?
               lambda x, y: np.log(x)*x**y),
    # above: explicityly zero np.logical_and(x==0, y>0)?
    np.radians: lambda x: np.radians(1),
    np.sin: np.cos,
    np.sinh: np.cosh,
    np.sqrt: lambda x: 0.5/np.sqrt(x),
    np.square: lambda x: 2.0*x,
    np.tan: lambda x: 1+np.tan(x)**2,
    np.tanh: lambda x: 1-np.tanh(x)**2}

UFUNC_DERIVATIVES[np.divide] = UFUNC_DERIVATIVES[np.true_divide]
UFUNC_DERIVATIVES[np.abs] = UFUNC_DERIVATIVES[np.fabs]
UFUNC_DERIVATIVES[np.cbrt] = lambda x: 1./(3.*np.cbrt(x)**2)


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
        if type(uncertainty) is not np.ndarray:
            raise ValueError('uncertainty should be plain ndarray')
        if uncertainty.shape != self.shape:
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
        if self._uncertainty is None:
            uncertainty = np.broadcast_to(np.array(0.), self.shape)
        else:
            uncertainty = self._uncertainty()
        uncertainty = uncertainty[...].view(self._uncertainty_cls)
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
        # No support yet for ufunc methods (at, reduce, reduceat, outer, etc.).
        assert method == '__call__'
        # No support yet for output arguments.
        assert 'out' not in kwargs
        # Apply the function to the nominal values.
        values = [(arg.nominal if isinstance(arg, Measurement) else arg)
                  for arg in inputs]
        value = ufunc(*values, **kwargs)
        # Set up the output as a Measurement that contains a derived uncertainty.
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        result = value.view(type(self))
        return result._add_derivatives(ufunc, inputs, values)

    def __array_function__(self, function, types, args, kwargs):
        if function is np.array2string:
            # Complete hack.
            if self.shape == ():
                return str(self)
            kwargs.setdefault('formatter',
                              {'all': Measurement.__str__})

        return super().__array_function__(function, types, args, kwargs)

    def _add_derivatives(self, ufunc, inputs, nominals):
        # get the functions that calculate derivatives of the ufunc
        # relative to the nominal input values.
        ufunc_derivs = UFUNC_DERIVATIVES[ufunc]
        if len(inputs) == 1:
            ufunc_derivs = (ufunc_derivs,)

        # For Measurement inputs, get possible derivatives to other measurements.
        deriv_dicts = [getattr(getattr(arg, '_uncertainty', None),
                               'derivatives', {}) for arg in inputs]
        # Set up a new derivatives dictionary, with entries to all measurements
        # the result depends on.
        derivatives = {}
        for deriv_dict in deriv_dicts:
            for k, [unc_id, derivative] in deriv_dict.items():
                derivatives[k] = [unc_id, None]
        # Add to derivatives using chain rule.
        for ufunc_deriv, deriv_dict in zip(ufunc_derivs, deriv_dicts):
            if deriv_dict:
                new_deriv = ufunc_deriv(*nominals)
            for unc_id, (uncertainty, derivative) in deriv_dict.items():
                if derivatives[unc_id][1] is None:
                    derivatives[unc_id][1] = new_deriv * derivative
                else:
                    derivatives[unc_id][1] += new_deriv * derivative
        self._uncertainty = DerivedUncertainty(derivatives)
        return self

    def __str__(self):
        return '{0}±{1}'.format(self.nominal, self.uncertainty)

    def __repr__(self):
        return '{0}(value={1}, uncertainty={2})'.format(
            type(self).__name__, self.nominal, self.uncertainty)


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
