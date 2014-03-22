from numpy import ndarray
from numpy.ma.core import (MaskedArray as NumpyMaskedArray,
                           masked, nomask, masked_print_option,
                           _recursive_make_descr, _recursive_printoption,
                           _print_templates)


class MaskedArray(NumpyMaskedArray):
    def _get_data(self):
        data = super(MaskedArray, self)._get_data()
        if hasattr(data, '__dict__'):
            data.__dict__.update(self._optinfo)
        return data
    _data = property(fget=_get_data)
    data = property(fget=_get_data)

    def __str__(self):
        """String representation.

        """
        if masked_print_option.enabled():
            f = masked_print_option
            if self is masked:
                return str(f)
            m = self._mask
            if m is nomask:
                res = self._data
            else:
                if m.shape == ():
                    if m.dtype.names:
                        m = m.view((bool, len(m.dtype)))
                        if m.any():
                            return str(tuple((f if _m else _d) for _d, _m in
                                             zip(self._data.tolist(), m)))
                        else:
                            return str(self._data)
                    elif m:
                        return str(f)
                    else:
                        return str(self._data)
                # convert to object array to make filled work
                names = self.dtype.names
                if names is None:
                    res = self._data.astype("O")
                    res.view(ndarray)[m] = f
                else:
                    rdtype = _recursive_make_descr(self.dtype, "O")
                    res = self._data.astype(rdtype)
                    _recursive_printoption(res, m, f)
        else:
            res = self.filled(self.fill_value)
        return str(res)

    def __repr__(self):
        """Literal string representation.

        """
        n = len(self.shape)
        name = ('array' if type(self) is ndarray else
                self._baseclass.__name__)
        parameters = dict(name=name, nlen=" " * len(name),
                          data=str(self), mask=str(self._mask),
                          fill=str(self.fill_value), dtype=str(self.dtype))
        if self.dtype.names:
            if n <= 1:
                return _print_templates['short_flx'] % parameters
            return _print_templates['long_flx'] % parameters
        elif n <= 1:
            return _print_templates['short_std'] % parameters
        return _print_templates['long_std'] % parameters
