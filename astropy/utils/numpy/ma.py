from astropy.extern import six
from numpy import ndarray
from numpy.ma.core import (MaskedArray as NumpyMaskedArray,
                           masked, nomask, masked_print_option,
                           _recursive_make_descr, _recursive_printoption,
                           _print_templates, getmask, make_mask_none)


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

    def __setitem__(self, indx, value):
        """x.__setitem__(i, y) <==> x[i]=y

        Set item described by index. If value is masked, masks those
        locations.

        """
        # astropy changes:
        # _data = self.data (instead of view)
        # ndarray.__setitem__(_data, indx, value)
        # -> _data[indx] = value, so that _data setter can do its work
        if self is masked:
            raise MaskError('Cannot alter the masked element.')
        # This test is useful, but we should keep things light...
#        if getmask(indx) is not nomask:
#            msg = "Masked arrays must be filled before they can be used as indices!"
#            raise IndexError(msg)
        _data = self.data
        _mask = ndarray.__getattribute__(self, '_mask')
        if isinstance(indx, six.text_type):
            _data[indx] = value
            if _mask is nomask:
                self._mask = _mask = make_mask_none(self.shape, self.dtype)
            _mask[indx] = getmask(value)
            return
        #........................................
        _dtype = ndarray.__getattribute__(_data, 'dtype')
        nbfields = len(_dtype.names or ())
        #........................................
        if value is masked:
            # The mask wasn't set: create a full version...
            if _mask is nomask:
                _mask = self._mask = make_mask_none(self.shape, _dtype)
            # Now, set the mask to its value.
            if nbfields:
                _mask[indx] = tuple([True] * nbfields)
            else:
                _mask[indx] = True
            if not self._isfield:
                self._sharedmask = False
            return
        #........................................
        # Get the _data part of the new value
        dval = value
        # Get the _mask part of the new value
        mval = getattr(value, '_mask', nomask)
        if nbfields and mval is nomask:
            mval = tuple([False] * nbfields)
        if _mask is nomask:
            # Set the data, then the mask
            _data[indx] = dval
            if mval is not nomask:
                _mask = self._mask = make_mask_none(self.shape, _dtype)
                _mask[indx] = mval
        elif not self._hardmask:
            # Unshare the mask if necessary to avoid propagation
            if not self._isfield:
                self.unshare_mask()
                _mask = ndarray.__getattribute__(self, '_mask')
            # Set the data, then the mask
            _data[indx] = dval
            _mask[indx] = mval
        elif hasattr(indx, 'dtype') and (indx.dtype == MaskType):
            indx = indx * umath.logical_not(_mask)
            _data[indx] = dval
        else:
            if nbfields:
                err_msg = "Flexible 'hard' masks are not yet supported..."
                raise NotImplementedError(err_msg)
            mindx = mask_or(_mask[indx], mval, copy=True)
            dindx = self._data[indx]
            if dindx.size > 1:
                np.copyto(dindx, dval, where=~mindx)
            elif mindx is nomask:
                dindx = dval
            _data[indx] = dindx
            _mask[indx] = mindx
        return
