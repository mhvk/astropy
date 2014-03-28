from astropy.extern import six
import numpy as np
from numpy import ndarray
import numpy.core.umath as umath
from numpy.ma.core import (MaskedArray as NumpyMaskedArray,
                           MaskedIterator as NumpyMaskedIterator,
                           masked, nomask, masked_print_option,
                           _recursive_make_descr, _recursive_printoption,
                           _print_templates, getmask, make_mask_none, mask_or,
                           MaskType, MaskError)


class MaskedIterator(NumpyMaskedIterator):
    def __getitem__(self, indx):
        result = self.dataiter.__getitem__(indx).view(type(self.ma))
        result._update_from(self.ma._data)
        if self.maskiter is not None:
            _mask = self.maskiter.__getitem__(indx)
            _mask.shape = result.shape
            result._mask = _mask
        return result


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

    def __getitem__(self, indx):
        """x.__getitem__(y) <==> x[y]

        Return the item described by i, as a masked array.

        """
        # astropy: _data = self.data (instead of view)

        # This test is useful, but we should keep things light...
#        if getmask(indx) is not nomask:
#            msg = "Masked arrays must be filled before they can be used as indices!"
#            raise IndexError(msg)
        dout = self.data[indx]
        # We could directly use ndarray.__getitem__ on self...
        # But then we would have to modify __array_finalize__ to prevent the
        # mask of being reshaped if it hasn't been set up properly yet...
        # So it's easier to stick to the current version
        _mask = self._mask
        if not getattr(dout, 'ndim', False):
            # A record ................
            if isinstance(dout, np.void):
                mask = _mask[indx]
                # We should always re-cast to mvoid, otherwise users can
                # change masks on rows that already have masked values, but not
                # on rows that have no masked values, which is inconsistent.
                dout = mvoid(dout, mask=mask, hardmask=self._hardmask)
            # Just a scalar............
            elif _mask is not nomask and _mask[indx]:
                return masked
        else:
            # Force dout to MA ........
            dout = dout.view(type(self))
            # Inherit attributes from self
            dout._update_from(self)
            # Check the fill_value ....
            if isinstance(indx, six.text_type):
                if self._fill_value is not None:
                    dout._fill_value = self._fill_value[indx]
                dout._isfield = True
            # Update the mask if needed
            if _mask is not nomask:
                dout._mask = _mask[indx]
                dout._sharedmask = True
#               Note: Don't try to check for m.any(), that'll take too long...
        return dout

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

    def _get_flat(self):
        "Return a flat iterator."
        return MaskedIterator(self)

    def _set_flat(self, value):
        "Set a flattened version of self to value."
        y = self.ravel()
        y[:] = value
    #
    flat = property(fget=_get_flat, fset=_set_flat,
                    doc="Flat version of the array.")
