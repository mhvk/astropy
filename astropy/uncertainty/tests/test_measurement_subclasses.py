# coding: utf-8
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from astropy import units as u
from astropy import constants as c
from astropy.units import Quantity
from astropy.constants import Constant
from astropy.coordinates import Angle, Longitude, representation as r

from .. import Measurement


def test_initialisation():
    v1 = Quantity(Measurement(5, 2), u.km)
    assert isinstance(v1, Quantity)
    assert isinstance(v1, Measurement)
    assert v1.unit == u.km
    assert v1.nominal == 5 * u.km
    assert v1.uncertainty == 2 * u.km
    v1_value = v1.value
    assert type(v1_value) is Measurement
    assert v1_value.nominal == 5
    assert v1_value.uncertainty == 2
    v2 = Measurement(5, 2) * u.km
    assert v2.unit == u.km
    assert v2.nominal == 5 * u.km
    assert v2.uncertainty == 2 * u.km
    v3 = Measurement(5 * u.km, 2000 * u.m)
    assert v3.unit == u.km
    assert v3.nominal == 5 * u.km
    assert v3.uncertainty == 2 * u.km
    v4 = Measurement(np.arange(5.) << u.km,
                     np.array([1., 2., 1., 2., 1.]) << u.km)
    assert v4.unit == u.km
    assert np.all(v4.nominal == np.arange(5.) << u.km)
    assert np.all(v4.uncertainty == np.array([1., 2., 1., 2., 1.]) << u.km)

    a1 = Measurement(Angle('1d'), Angle('0d1m'))
    assert isinstance(a1, Angle)
    assert isinstance(a1, Measurement)
    assert isinstance(a1.nominal, Angle)
    assert isinstance(a1.uncertainty, Angle)
    assert a1.nominal == 1*u.degree
    assert a1.uncertainty == 1.*u.arcminute

    l1 = Measurement(Longitude('3h'), Angle('0d1m'))
    assert isinstance(l1, Longitude)
    assert isinstance(l1, Measurement)
    assert isinstance(l1.nominal, Longitude)
    assert isinstance(l1.uncertainty, Angle)
    assert a1.nominal == 1*u.degree
    assert a1.uncertainty == 1.*u.arcminute

    c1 = Measurement(c.G)
    assert isinstance(c1, Constant)
    assert isinstance(c1, Measurement)
    assert isinstance(c1.nominal, Constant)
    assert isinstance(c1.uncertainty, Quantity)
    assert c1.nominal == c.G
    assert c1.uncertainty == c.G.uncertainty << c.G.unit


class TestBasics():
    def setup(self):
        self.v = Measurement(5., 2.) << u.km
        self.a = Measurement(np.arange(1., 5.), 1.) << u.s
        self.b = Measurement(np.array([1., 2., 3.]),
                             np.array([0.1, 0.2, 0.1])) << u.m

    def test_unit_change(self):
        v_m = self.v.to(u.m)
        assert v_m.unit == u.m
        assert v_m.nominal.value == self.v.nominal.to_value(u.m)
        assert v_m.uncertainty.value == self.v.uncertainty.to_value(u.m)

    def test_to_value(self):
        v_in_m = self.v.to_value(u.m)
        assert v_in_m.nominal == self.v.nominal.to_value(u.m)
        assert v_in_m.uncertainty == self.v.uncertainty.to_value(u.m)

    def test_addition(self):
        unit = self.v.unit
        c1 = self.v + Measurement(12, 5) * unit
        assert c1.nominal == self.v.nominal + 12*unit
        assert c1.unit == u.km
        # Uncertainties under addition add in quadrature
        assert u.allclose(c1.uncertainty,
                          np.sqrt(self.v.uncertainty**2 + (5*unit)**2))
        # now with different units
        c2 = self.v + (Measurement(12000., 5000.) << u.m)
        assert c2.nominal == self.v.nominal + 12*u.km
        assert c2.unit == u.km
        assert u.allclose(c2.uncertainty,
                          np.sqrt(self.v.uncertainty**2 + (5*u.km)**2))
        # try array
        c3 = self.v + self.b
        assert np.all(c3.nominal == self.v.nominal + self.b.nominal)
        assert u.allclose(c3.uncertainty,
                          np.sqrt(self.v.uncertainty**2 +
                                  self.b.uncertainty**2))
        # try adding regular Quantity
        q = 10. * self.v.unit
        c4 = self.v + q
        assert c4.nominal == self.v.nominal + q
        assert c4.uncertainty == self.v.uncertainty

    def test_subtraction(self):
        unit = self.v.unit
        c1 = self.v - Measurement(12, 5) * unit
        assert c1.unit == u.km
        assert c1.nominal == self.v.nominal - 12*unit
        # Uncertainties under addition add in quadrature
        assert u.allclose(c1.uncertainty,
                          np.sqrt(self.v.uncertainty**2 + (5*unit)**2))

    def test_multiplication(self):
        c1 = self.v * self.a
        assert np.all(c1.nominal == self.v.nominal * self.a.nominal)

        # Fractional uncertainties under multiplication add in quadrature
        assert u.allclose(c1.uncertainty/c1.nominal, np.sqrt(
            (self.v.uncertainty/self.v.nominal)**2 +
            (self.a.uncertainty/self.a.nominal)**2))
        # Test multiplication with straight Quantity
        q = 10. * u.s
        c2 = self.a * q
        assert c2.unit == self.a.unit * u.s
        assert np.all(c2.nominal == self.a.nominal * q)
        assert np.all(c2.uncertainty == self.a.uncertainty * q)

    def test_division(self):
        c1 = self.v / self.a
        assert u.allclose(c1.nominal, self.v.nominal / self.a.nominal)
        # Fractional uncertainties under division add in quadrature
        assert u.allclose(c1.uncertainty/c1.nominal, np.sqrt(
            (self.v.uncertainty/self.v.nominal)**2 +
            (self.a.uncertainty/self.a.nominal)**2))


def test_more_complex():
    G = Measurement(c.G)
    m1 = Measurement(1e15, 1e5) * u.kg
    m2 = Measurement(100, 10) * u.kg
    r = Measurement(10000, 500) * u.m
    F = G * (m1 * m2) / r**2
    assert np.allclose(F.nominal.value, c.G.si.value * (1e15 * 100) / 10000**2)
    assert F.unit == u.N
    # Uncertainties calculated using partial derivative method
    assert u.allclose(F.uncertainty, np.sqrt(
        (m1.nominal*m2.nominal/(r.nominal**2)*G.uncertainty)**2 +
        (G.nominal*m2.nominal/(r.nominal**2)*m1.uncertainty)**2 +
        (G.nominal*m1.nominal/(r.nominal**2)*m2.uncertainty)**2 +
        (-2*G.nominal*m1.nominal*m2.nominal/(r.nominal**3)*r.uncertainty)**2))


class TestRepresentations:
    def setup(self):
        self.x = 5.
        self.y = 12.
        self.z = 0.
        self.c = r.CartesianRepresentation(self.x, self.y, self.z)
        self.mx = Measurement(self.x, 2.) << u.m
        self.my = Measurement(self.y, 3.) << u.m
        self.mz = Measurement(self.z, 2.) << u.m
        self.mc = r.CartesianRepresentation(self.mx, self.my, self.mz)

    def test_initialization(self):
        assert self.mc.x == self.mx
        assert self.mc.y == self.my
        assert self.mc.z == self.mz

    @pytest.mark.xfail
    def test_norm(self):
        # Need stacking and erfa override.
        norm = self.mc.norm()
        assert norm.nominal == self.c.norm()
