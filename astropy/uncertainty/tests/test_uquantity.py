# coding: utf-8
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from ... import units as u
from ... import constants as c

from .uncertainty import Variable


def test_initialisation():
    v1 = u.Quantity(Variable(5, 2), u.km)
    assert v1.value == Variable(5, 2)
    assert v1.unit == u.km
    assert v1.nominal == 5 * u.km
    assert v1.uncertainty == 2 * u.km
    v2 = Variable(5, 2) * u.km
    assert v2.value == Variable(5, 2)
    assert v2.unit == u.km
    assert v2.nominal == 5 * u.km
    assert v2.uncertainty == 2 * u.km
    assert v1 == v2
    v3 = Variable(5 * u.km, 2000 * u.m)
    assert v3.value == Variable(5, 2)
    assert v3.unit == u.km
    assert v3.nominal == 5 * u.km
    assert v3.uncertainty == 2 * u.km
    v4 = Variable(np.arange(5.) << u.km,
                  np.array([1., 2., 1., 2., 1.]) << u.km)
    assert v4.unit == u.km
    assert np.all(v4.nominal == np.arange(5.) << u.km)
    assert np.all(v4.uncertainty == np.array([1., 2., 1., 2., 1.]) << u.km)


class TestBasics():
    def setup(self):
        self.v = Variable(5., 2.) << u.km
        self.a = Variable(np.arange(1., 5.), 1.) << u.s
        self.b = Variable(np.array([1., 2., 3.]),
                          np.array([0.1, 0.2, 0.1])) << u.m

    def test_addition(self):
        unit = self.v.unit
        c1 = self.v + Variable(12, 5) << unit
        assert c1.nominal == self.v.nominal + 12*unit
        assert c1.unit == u.km
        # Uncertainties under addition add in quadrature
        assert np.allclose(c1.uncertainty,
                           np.sqrt(self.v.uncertainty**2 + (5*unit)**2))
        # now with different units
        c2 = self.v + Variable(12000., 5000.) << u.m
        assert c2.value == self.v.value + 12*u.m
        assert c2.unit == u.km
        assert np.allclose(c2.uncertainty,
                           np.sqrt(self.v.uncertainty**2 + (5*u.m)**2))
        # try array
        c3 = self.v + self.b
        assert np.all(c3.nominal == self.v.nominal + self.b.nominal)
        assert np.allclose(c3.uncertainty,
                           np.sqrt(self.v.uncertainty**2 +
                                   self.b.uncertainty**2))
        # try adding regular Quantity
        q = 10. * self.v.unit
        c4 = self.v + q
        assert c4.value == Variable(self.v.value + 10.)
        assert c4.nominal == self.v.nominal + q
        assert c4.uncertainty == self.v.uncertainty

    def test_subtraction(self):
        unit = self.v.unit
        c1 = self.v - Variable(12, 5) * unit
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
        assert np.all(c2.value == self.a.value * 10.)
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
    G = Variable(c.G.value, c.G.uncertainty) * c.G.unit
    m1 = Variable(1e15, 1e5) * u.kg
    m2 = Variable(100, 10) * u.kg
    r = Variable(10000, 500) * u.m
    F = G * (m1 * m2) / r**2
    assert np.allclose(F.nominal.value, c.G.si.value * (1e15 * 100) / 10000**2)
    assert F.unit == u.N
    # Uncertainties calculated using partial derivative method
    assert u.allclose(F.uncertainty, np.sqrt(
        (m1.nominal*m2.nominal/(r.nominal**2)*G.uncertainty)**2 +
        (G.nominal*m2.nominal/(r.nominal**2)*m1.uncertainty)**2 +
        (G.nominal*m1.nominal/(r.nominal**2)*m2.uncertainty)**2 +
        (-2*G.nominal*m1.nominal*m2.nominal/(r.nominal**3)*r.uncertainty)**2))
