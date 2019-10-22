# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Uncertainty and DerivedUncertainty classes.

For use in the Measurement class.
"""
import numpy as np

from astropy.utils.misc import isiterable


class Uncertainty:
    """Measurement Uncertainty.

    Parameters
    ----------
    uncertainty : `~numpy.ndarray`
        Measurement uncertainties
    check : Bool
        Whether to check all uncertainties are positive (default: ``True``).
        Mostly for internal use in slicing, etc.

    Notes
    -----
    This class is for internal use by `~astropy.uncertainties.Measurement`.
    """
    def __init__(self, uncertainty, check=True):
        if check and np.any(uncertainty < 0):
            raise ValueError("The uncertainty cannot be negative.")
        self.uncertainty = uncertainty
        # Set up an ID that uses the memory location and the array strides;
        # the latter helps keep track of slices.
        self.id = (uncertainty.__array_interface__['data'][0],
                   uncertainty.__array_interface__['strides'])
        # Any derived uncertainty will use this dictionary, ensuring there is
        # a link that keeps the value in memory until it is not used any more.
        self.derivatives = {self.id: [self, 1.]} if np.any(uncertainty > 0) else {}

    def __call__(self):
        """Get the uncertainty.  Returns simply the stored value."""
        return self.uncertainty

    def __getitem__(self, item):
        # Particular items are stored in a new instance, with a new ID.
        return self.__class__(uncertainty=self.uncertainty[item], check=False)

    def __repr__(self):
        return '{0}({1})'.format(type(self).__name__, self.uncertainty)


class DerivedUncertainty:
    """Derived Uncertainty.

    Parameters
    ----------
    derivatives : dict
        A dictionary keyed on uncertainty IDs, containing both the uncertainty
        and the derivative with which a variable depends on it.

    Notes
    -----
    This class is for internal use by `~astropy.uncertainties.Measurement`.
    """
    def __init__(self, derivatives):
        self.derivatives = derivatives

    @property
    def uncertainty_components(self):
        """Iterator over the different components of the uncertainty."""
        for unc_id, (uncertainty, derivative) in self.derivatives.items():
            uncertainty = uncertainty.uncertainty
            component = np.abs(derivative * uncertainty)
            # The component is generally an array; if any derivatives are
            # undefined, but the uncertainty is 0, the component is zero too.
            # [Twice faster is to do: np.isnan(np.max(a))]
            if np.any(np.isnan(derivative)):
                component[np.logical_and(np.isnan(derivative),
                                         uncertainty == 0.)] = 0.
            yield component

    def __call__(self):
        """Calculate the uncertainty as the rms of the components."""
        variance = None
        for delta in self.uncertainty_components:
            variance = delta**2 if variance is None else variance + delta**2
        return np.sqrt(variance)

    def __getitem__(self, item):
        # Get the item from each of the uncertainties that contribute.
        # TODO: this needs proper support for slicing, incl. broadcasting, etc.
        derivatives = {}
        for unc_id, (uncertainty, derivative) in self.derivatives.items():
            uncertainty = uncertainty[item]
            if isiterable(derivative):
                derivative = derivative[item]
            derivatives[uncertainty.id] = [uncertainty, derivative]
        return self.__class__(derivatives)

    def __repr__(self):
        return '{0}({1})'.format(type(self).__name__, self.derivatives)
