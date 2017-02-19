# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.ma import masked
from ..ma import MaskedArray

from .....units import Quantity
from .....coordinates import Angle, Longitude, Distance
from ..... import units as u


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
        self.q2 = quantity_type(np.arange(0., 3.), unit[quantity_type])
        self.mq = MaskedArray(self.q.copy(), mask=[True, False, False])
        self.mq2 = MaskedArray(self.q2.copy(), mask=[True, False, False])

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
        assert np.all(mq_sel2 == mq_sel2.__class__(q_sel2, mask=mask_sel2))
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

    def test_str(self, quantity_type):
        self._setup(quantity_type)
        if issubclass(quantity_type, Angle):
            assert str(self.mq) == "[-- '4d00m00s' '5d00m00s']"
        else:
            assert str(self.mq) == "[-- 4.0 5.0] {0}".format(self.q.unit)

    def test_repr(self, quantity_type):
        self._setup(quantity_type)
        if issubclass(quantity_type, Angle):
            assert repr(self.mq) == (
                "masked_{0}(data = [-- '4d00m00s' '5d00m00s'],\n"
                "{1}        mask = [ True False False],\n"
                "{1}  fill_value = 1e+20)\n"
                .format(self.quantity_type.__name__,
                        " " * len(self.quantity_type.__name__)))
        else:
            assert repr(self.mq) == (
                "masked_{0}(data = [-- 4.0 5.0] {1},\n"
                "{2}        mask = [ True False False],\n"
                "{2}  fill_value = 1e+20)\n"
                .format(self.quantity_type.__name__, self.q.unit,
                        " " * len(self.quantity_type.__name__)))


@pytest.mark.usefixtures('quantity_type')
class TestMaskedQuantity(TestMaskedArrayWithQuantity):
    """Test one can construct a mixin class of MaskedArray and Quantity"""
    # this will run all tests from TestMaskedArrayWithQuantity
    # plus the new ones that checks the Quantity methods
    def _setup(self, quantity_type):
        super(TestMaskedQuantity, self)._setup(quantity_type)

        # setup class following ~numpy.ma.tests.test_subclassing.MSubArray
        class MQ(MaskedArray, quantity_type):
            def __new__(cls, *args, **kwargs):
                mask = kwargs.pop('mask', np.ma.nomask)
                q = quantity_type(*args, **kwargs)
                return MaskedArray.__new__(cls, data=q, mask=mask)

        self.MQ = MQ
        self.mq = MQ(self.q.copy(), mask=[True, False, False])
        self.mq2 = MQ(self.q2.copy(), mask=[True, False, False])

    def test_addition_subtraction(self, quantity_type):
        self._setup(quantity_type)
        qsum = self.q + self.q2
        mqsum = self.mq + self.mq2
        assert np.all(mqsum.data == qsum)
        qdiff = self.q - self.q2
        mqdiff = self.mq - self.mq2
        assert np.all(mqdiff.data == qdiff)

    def test_multiplication_division(self, quantity_type):
        self._setup(quantity_type)
        if quantity_type is Longitude:
            pass
            # this needs attribute_wrapper
            # with pytest.raises(TypeError):
            #     mqproduct = self.mq * self.mq2
            # with pytest.raises(TypeError):
            #     mqratio = self.mq / self.mq2
        else:
            qproduct = self.q * self.q2
            mqproduct = self.mq * self.mq2
            assert np.all(mqproduct.data == qproduct)
            qratio = self.q / self.q2
            mqratio = self.mq / self.mq2
            assert np.all((mqratio.data == qratio) |
                          mqratio.mask & np.isinf(qratio))
