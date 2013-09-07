# Licensed under a 3-clause BSD style license - see LICENSE.rst
import functools
import itertools

import numpy as np

from ...tests.helper import pytest
from .. import Time
from ..core import PRECESSION_MODELS, PRECESSION_NUTATION_MODELS

allclose_hours = functools.partial(np.allclose, rtol=1e-15, atol=1e-9)
# 1 nanosec atol
within_1_second = functools.partial(np.allclose, rtol=1., atol=1./3600.)
within_2_seconds = functools.partial(np.allclose, rtol=1., atol=2./3600.)


class TestGST():
    """Test Greenwich Sidereal Time"""

    t = Time(['2012-06-30 12:00:00', '2012-06-30 23:59:59',
              '2012-06-30 23:59:60', '2012-07-01 00:00:00',
              '2012-07-01 12:00:00'], scale='utc')

    def test_gmst(self):
        """Compare Greenwich Mean Sidereal Time with what was found earlier.
        Ideally, would compare with a reference implementation.  Tried
        http://www.erdrotation.de/ERIS/EN/Eris/InteractiveTools/Sofa/sofa.html
        on Sep'13, but website not work.
        """
        gmst_compare = np.array([6.5968497894730564, 18.629426164144697,
                                 18.629704702452862, 18.629983240761003,
                                 6.6628381828899643])
        gmst = self.t.gmst()
        assert allclose_hours(gmst.value, gmst_compare)

    def test_gst(self):
        """Compare Greenwich Apparent Sidereal Time with what was found
        earlier. Ideally, would compare with a reference implementation...
        """
        gst_compare = np.array([6.5971168570494854, 18.629694220878296,
                                18.62997275921186, 18.630251297545389,
                                6.6631074284018244])
        gst = self.t.gst()
        assert allclose_hours(gst.value, gst_compare)

    def test_gmst_gst_close(self):
        """Check that Mean and Apparent are within a few seconds."""
        gmst = self.t.gmst()
        gst = self.t.gst()
        assert within_2_seconds(gst.value, gmst.value)


class TestLST():
    """Test Local Sidereal Time"""

    t = Time(['2012-06-30 12:00:00', '2012-06-30 23:59:59',
              '2012-06-30 23:59:60', '2012-07-01 00:00:00',
              '2012-07-01 12:00:00'], scale='utc', lon='120d', lat='10d')

    def test_lmst(self):
        """Compare Local Mean Sidereal Time with what was found earlier,
        as well as with what is expected from GMST
        """
        lmst_compare = np.array([14.596849789473058, 2.629426164144693,
                                 2.6297047024528588, 2.6299832407610033,
                                 14.662838182889967])

        gmst = self.t.gmst()
        lmst = self.t.lmst()
        assert allclose_hours(lmst.value, lmst_compare)
        assert allclose_hours((lmst-gmst).wrap_at('12h').value,
                              self.t.lon.to('hourangle').value)

    def test_lst(self):
        """Compare Local Apparent Sidereal Time with what was found
        earlier. Ideally, would compare with a reference implementation...
        """
        lst_compare = np.array([14.597116857049487, 2.6296942208782959,
                                2.6299727592118565, 2.6302512975453887,
                                14.663107428401826])

        gst = self.t.gst()
        lst = self.t.lst()
        assert allclose_hours(lst.value, lst_compare)
        assert allclose_hours((lst-gst).wrap_at('12h').value,
                              self.t.lon.to('hourangle').value)

    def test_lmst_lst_close(self):
        """This really follows from above, but for completeness..."""
        lmst = self.t.lmst()
        lst = self.t.lst()
        assert within_2_seconds(lst.value, lmst.value)

    def test_lst_needs_lon(self):
        t = Time('2012-02-02', scale='utc')
        with pytest.raises(ValueError):
            t.lmst()
        with pytest.raises(ValueError):
            t.lst()


class TestModelInterpretation():
    """Check that models are different, and that wrong models are recognized"""
    t = Time(['2012-06-30 12:00:00'], scale='utc', lon='120d', lat='10d')

    def test_precession_model_uniqueness(self):
        """Check models give different answers, yet are close."""
        for model1, model2 in itertools.combinations(
                PRECESSION_MODELS.keys(), 2):
            gmst1 = self.t.gmst(model1)
            gmst2 = self.t.gmst(model2)
            assert np.all(gmst1.value != gmst2.value)
            assert within_1_second(gmst1.value, gmst2.value)
            lmst1 = self.t.lmst(model1)
            lmst2 = self.t.lmst(model2)
            assert np.all(lmst1.value != lmst2.value)
            assert within_1_second(lmst1.value, lmst2.value)

    def test_precession_nutation_model_uniqueness(self):
        for model1, model2 in itertools.combinations(
                PRECESSION_NUTATION_MODELS.keys(), 2):
            gst1 = self.t.gst(model1)
            gst2 = self.t.gst(model2)
            assert np.all(gst1.value != gst2.value)
            assert within_1_second(gst1.value, gst2.value)
            lst1 = self.t.lst(model1)
            lst2 = self.t.lst(model2)
            assert np.all(lst1.value != lst2.value)
            assert within_1_second(lst1.value, lst2.value)

    def test_wrong_models_raise_exceptions(self):

        with pytest.raises(ValueError):
            self.t.gmst('nonsense')

        for model in (set(PRECESSION_NUTATION_MODELS.keys()) -
                      set(PRECESSION_MODELS.keys())):
            with pytest.raises(ValueError):
                self.t.gmst(model)
            with pytest.raises(ValueError):
                self.t.lmst(model)

        with pytest.raises(ValueError):
            self.t.gst('nonsense')

        for model in (set(PRECESSION_MODELS.keys()) -
                      set(PRECESSION_NUTATION_MODELS.keys())):
            with pytest.raises(ValueError):
                self.t.gst(model)
            with pytest.raises(ValueError):
                self.t.lst(model)
