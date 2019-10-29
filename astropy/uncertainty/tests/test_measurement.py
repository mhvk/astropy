# coding: utf-8
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from .. import Measurement


def test_initialisation():
    v1 = Measurement(5, 2)
    assert v1.nominal == 5
    assert v1.uncertainty == 2
    # No uncertainty.
    v2 = Measurement(5)
    assert v2.nominal == 5
    assert v2.uncertainty == 0
    v4 = Measurement(np.arange(5.), 2.)
    assert np.all(v4.nominal == np.arange(5.))
    assert v4.uncertainty.shape == (5,)
    assert np.all(v4.uncertainty == 2)
    v5 = Measurement(np.arange(5.), np.array([1., 2., 1., 2., 1.]))
    assert np.all(v5.nominal == np.arange(5.))
    assert np.all(v5.uncertainty == np.array([1., 2., 1., 2., 1.]))


class TestBasics():
    def setup(self):
        self.v = Measurement(5., 2.)
        self.a = Measurement(np.arange(1., 5.), 1.)
        self.b = Measurement(np.array([1., 2., 3.]), np.array([0.1, 0.2, 0.1]))

    def test_addition(self):
        c = self.v + Measurement(12, 5)
        assert c.nominal == self.v.nominal + 12
        # Uncertainties under addition add in quadrature
        assert c.uncertainty == np.sqrt(self.v.uncertainty**2 + 5**2)
        # try array
        c3 = self.v + self.b
        assert np.all(c3.nominal == self.v.nominal + self.b.nominal)
        assert np.allclose(c3.uncertainty, np.sqrt(self.v.uncertainty**2 +
                                                   self.b.uncertainty**2))
        # Try adding a regular number.
        c4 = self.v + 10.
        assert c4.nominal == self.v.nominal + 10.
        assert c4.uncertainty == self.v.uncertainty
        # And a measurement without an uncertainty.
        c5 = self.v + Measurement(10.)
        assert c5.nominal == self.v.nominal + 10.
        assert len(c5._uncertainty.derivatives) == 1
        assert c5.uncertainty == self.v.uncertainty

    def test_subtraction(self):
        c = self.v - Measurement(12, 5)
        assert c.nominal == self.v.nominal - 12
        # Uncertainties under addition add in quadrature
        assert c.uncertainty == np.sqrt(self.v.uncertainty**2 + 5**2)
        c2 = self.b - Measurement(12, 5)
        assert np.all(c2.nominal == self.b.nominal - 12)
        c2_uncertainty = c2.uncertainty
        assert c2_uncertainty.shape == c2.shape
        assert np.all(c2_uncertainty == np.sqrt(self.b.uncertainty**2 + 5**2))

    def test_subtraction_nothing_left(self):
        c = self.v - self.v
        assert c.nominal == 0
        assert c.uncertainty == 0
        assert c._uncertainty.derivatives == {}

    def test_multiplication(self):
        c = self.v * self.a
        assert np.all(c.nominal == self.v.nominal * self.a.nominal)

        # Fractional uncertainties under multiplication add in quadrature
        assert np.allclose(c.uncertainty / c.nominal, np.sqrt(
            (self.v.uncertainty / self.v.nominal)**2 +
            (self.a.uncertainty / self.a.nominal)**2))

        # Test multiplication with straight number
        c2 = self.a * 10.
        assert np.all(c2.nominal == self.a.nominal * 10.)
        assert np.all(c2.uncertainty == self.a.uncertainty * 10.)

    def test_division(self):
        c = self.v / self.a
        assert np.allclose(c.nominal, self.v.nominal / self.a.nominal)
        # Fractional uncertainties under division add in quadrature
        assert np.allclose(c.uncertainty/c.nominal, np.sqrt(
            (self.v.uncertainty / self.v.nominal)**2 +
            (self.a.uncertainty / self.a.nominal)**2))

    def test_tracking(self):
        c = Measurement(12, 5) + self.v
        assert c.uncertainty == np.sqrt(5**2 + self.v.uncertainty**2)
        c = c - self.v
        assert c.nominal == 12
        assert c.uncertainty == 5
        c2 = self.v - self.v
        assert c2.nominal == 0
        assert c2.uncertainty == 0
