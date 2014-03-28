# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from numpy.ma import masked
from ..ma import MaskedArray

from ....tests.helper import pytest
from ....units import Quantity
from .... import units as u


class TestMaskedArrayWithQuantity():
    """Test one can construct a masked array from a Quantity"""

    def setup(self):
        self.q = Quantity(np.arange(3., 6.), u.m)
        self.mq = MaskedArray(self.q, mask=[True, False, False])

    def test_baseclass(self):
        assert np.all(self.mq.data == self.q)

    def test_access(self):
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

    def test_setter(self):
        q = np.arange(4., 7.) * u.m
        mq = MaskedArray(np.arange(4., 7.) * u.m, mask=[False, False, False])
        mq[1] = 1. * u.km
        assert mq.data[1] == 1. * u.km
        assert np.all((mq.data == q) == np.array([True, False, True]))

    def test_filled(self):
        assert np.all(self.mq.filled() == self.q.unit *
                      np.where(self.mq.mask, self.mq.fill_value, self.q.value))

    def test_representation(self):
        assert str(self.mq) == '[-- 4.0 5.0] m'
        assert repr(self.mq) == (
            'masked_Quantity(data = [-- 4.0 5.0] m,\n'
            '                mask = [ True False False],\n'
            '          fill_value = 1e+20)\n')
