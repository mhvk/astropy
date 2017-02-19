# coding: utf-8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Test broadcast_arrays replacement on Quantity class.
"""
from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

import pytest
import numpy as np

from ..ma import (MaskedArray, PR4576, PR4585, PR4586,
                  mvoid, PR4866)
#                  PRTBD, PRTBD2)


def test_import():
    """Check that what is imported from code is what we are testing."""
    from ... import numpy as anp
    assert anp.ma.MaskedArray is MaskedArray
    assert anp.ma.mvoid is mvoid
    # assert anp.ma.getdata is getdata


@pytest.mark.parametrize(('item', 'pr'),
                         ((MaskedArray, PR4576),
                          (MaskedArray, PR4585),
                          (MaskedArray, PR4586),
                          (mvoid, PR4866)))
def test_PR(item, pr):
    """Test the test functions

    The possibly patched versions of `item` should always be OK.
    The numpy version may be, in which case we just use it, or it may not,
    it which case we use the patched version.
    """
    assert pr(item)
    if pr():
        assert item is getattr(np.ma, item.__name__)
    else:
        assert item is not getattr(np.ma, item.__name__)
