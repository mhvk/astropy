# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Helpers fo the Measurement class.

In particular derivatives to the various ufuncs.
"""

import numpy as np

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
    np.tanh: lambda x: 1-np.tanh(x)**2,
    np.cbrt: lambda x: 1./(3.*np.cbrt(x)**2)}

UFUNC_DERIVATIVES[np.divide] = UFUNC_DERIVATIVES[np.true_divide]
UFUNC_DERIVATIVES[np.abs] = UFUNC_DERIVATIVES[np.fabs]


def chain_derivatives(ufunc, inputs, nominals):
    """Calculate derivatives to all the base Measurements."""
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

    # Remove any that are zero.
    zero_ids = [unc_id for unc_id, (_, derivative)
                in derivatives.items() if np.all(derivative == 0)]
    for unc_id in zero_ids:
        derivatives.pop(unc_id)

    return derivatives
