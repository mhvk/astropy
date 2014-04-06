# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from numpy.ma import masked
from ..ma import MaskedArray

from ....tests.helper import pytest
from ....units import Quantity
from ....coordinates import Angle, Longitude, Distance
from .... import units as u


@pytest.fixture(scope='module', params=[Quantity, Distance, Longitude])
def quantity_type(request):
    return request.param


@pytest.mark.usefixtures('quantity_type')
class TestMaskedArrayWithQuantity(object):
    """Test one can construct a masked array from a Quantity"""

    def _setup(self, quantity_type):
        self.quantity_type = quantity_type
        unit = {Quantity: u.m, Longitude: u.deg, Distance: u.kpc}
        self.q = quantity_type(np.arange(3., 6.), unit[quantity_type])
        self.mq = MaskedArray(self.q.copy(), mask=[True, False, False])

    def test_baseclass(self, quantity_type):
        self._setup(quantity_type)
        assert self.mq._baseclass is type(self.q)
        assert np.all(self.mq.data == self.q)

    def test_access(self, quantity_type):
        self._setup(quantity_type)
        mq_sel0 = self.mq[0]
        assert mq_sel0 is masked
        mq_sel1 = self.mq[1]
        q_sel1 = self.q[1]
        assert mq_sel1 == q_sel1
        mq_sel2 = self.mq[:2]
        q_sel2 = self.q[:2]
        mask_sel2 = self.mq.mask[:2]
        assert np.all(mq_sel2 == MaskedArray(q_sel2, mask_sel2))
        assert np.all(mq_sel2.data == q_sel2)
        assert np.all(mq_sel2.mask == mask_sel2)

    def test_setter(self, quantity_type):
        self._setup(quantity_type)
        self.mq[1] = 120. * self.q.unit
        assert self.mq.data[1] == 120. * self.q.unit
        assert np.all((self.mq.data == self.q) ==
                      np.array([True, False, True]))

    def test_filled(self, quantity_type):
        self._setup(quantity_type)
        assert np.all(self.mq.filled() == self.q.unit *
                      np.where(self.mq.mask, self.mq.fill_value, self.q.value))

    def test_representation(self, quantity_type):
        self._setup(quantity_type)
        if issubclass(quantity_type, Angle):
            assert str(self.mq) == "[-- '4d00m00s' '5d00m00s']"
            assert repr(self.mq) == (
                "masked_{0}(data = [-- '4d00m00s' '5d00m00s'],\n"
                "{1}        mask = [ True False False],\n"
                "{1}  fill_value = 1e+20)\n"
                .format(self.quantity_type.__name__,
                        " " * len(self.quantity_type.__name__)))
        else:
            assert str(self.mq) == "[-- 4.0 5.0] {0}".format(self.q.unit)
            assert repr(self.mq) == (
                "masked_{0}(data = [-- 4.0 5.0] {1},\n"
                "{2}        mask = [ True False False],\n"
                "{2}  fill_value = 1e+20)\n"
                .format(self.quantity_type.__name__, self.q.unit,
                        " " * len(self.quantity_type.__name__)))
