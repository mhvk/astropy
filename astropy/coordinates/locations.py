# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This module contains functionality pertaining to locations.  Nominally, these
are locations on Earth, but could in the future be generalized to other
celestial bodies.
"""

from .. import units as u
from .coordsystems import SphericalCoordinatesBase

__all__ = ['EarthLocation']


class EarthLocation(SphericalCoordinatesBase):
    """
    A point relative to the surface of the Earth's geoid.


    Parameters
    ----------
    {params}
    elevation : `~astropy.time.Distance`
        The elevation above the reference surface (the Earth's geoid).
        Roughly speaking, this means elevation above sea level. Defaults
        to 0.

    Notes
    -----
    Strictly, this is intended to be a point on the
    `ITRF <http://www.iers.org/nn_10402/IERS/EN/DataProducts/ITRF/itrf.html>`_,
    but for most astronomical purposes that can be considered a point on
    the `WGS84 <http://en.wikipedia.org/wiki/World_Geodetic_System>`_ ellipsoid
    (i.e., GPS coordinates).

    """

    __doc__ = __doc__.format(params=SphericalCoordinatesBase._init_docstring_param_templ.format(lonnm='lon', latnm='lat'))

    #from WGS84 reference ellipsoid
    _WGS84_EQUATORIAL_RAD = 6378137 * u.meter
    _WGS84_INVERSE_FLATTENING = 298.257223563  # 1/f
    # average of major and minor axis
    _WGS84_MEAN_RAD = _WGS84_EQUATORIAL_RAD * (2 - 1 / _WGS84_INVERSE_FLATTENING) / 2

    def __init__(self, *args, **kwargs):
        from .distances import Distance

        if 'distance' in kwargs:
            raise ValueError('cannot give `distance` for creating an EarthLocation - instead use `elevation`')
        elev = kwargs.pop('elevation', None)

        super(EarthLocation, self).__init__()
        super(EarthLocation, self)._initialize_latlon('lon', 'lat', args, kwargs)

        self.elevation = elev

    def __repr__(self):
        msg = "<{0} lat={1:.5f} deg, lon={2:.5f} deg elev={3}>"
        return msg.format(self.__class__.__name__, self.lat.degree,
                          self.lon.degree, str(self.elevation))

    @property
    def lonangle(self):
        return self.lon

    @property
    def latangle(self):
        return self.lat

    @property
    def elevation(self):
        from .distances import Distance
        d = self._distance
        return Distance(d.m - self._WGS84_MEAN_RAD.value, u.m)

    @elevation.setter
    def elevation(self, value):
        from .distances import Distance

        if value is None:  # no initial value provided - assume surface
            self._distance = Distance(self._WGS84_MEAN_RAD)
        else:
            d = Distance(value)
            self._distance = Distance(d.m + self._WGS84_MEAN_RAD.value, u.m)

    @classmethod
    def from_name(cls, name):
        """
        `from_name` is invalid for an `EarthLocation` and will raise a
        `NotImplementedError`
        """
        raise NotImplementedError('from_name is not sensible for locations')
        #TODO: reorganize `coordsystems` heirarchy or move `from_name` so that this is unnecessary

    #TODO: implement transformations from this to ITRF coordinates once those are defined
    #TODO: correctly account for this being an ellipse, rather than a