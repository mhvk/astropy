from astropy.extern import six
import numpy as np

from distutils import version
NUMPY_VERSION = version.LooseVersion(np.__version__)

BUG3907 = NUMPY_VERSION < version.LooseVersion('9.9.9')
BUG4576 = NUMPY_VERSION < version.LooseVersion('1.9.0')
BUG4585 = NUMPY_VERSION < version.LooseVersion('1.9.0')
BUG4586 = NUMPY_VERSION < version.LooseVersion('9.9.9')
BUGTBD = True

# not essential for astropy
# BUG4617 = NUMPY_VERSION < version.LooseVersion('9.9.9')


from functools import reduce
from numpy import ndarray
import numpy.core.umath as umath
from numpy.ma.core import (
    MaskedArray as NumpyMaskedArray, MaskedIterator as NumpyMaskedIterator,
    masked, nomask, masked_print_option,
    _recursive_make_descr, _recursive_printoption,
    _print_templates, getmask, make_mask_none, mask_or, MaskType, MaskError,
    _MaskedBinaryOperation as Numpy_MaskedBinaryOperation,
    _DomainedBinaryOperation as Numpy_DomainedBinaryOperation,
    get_masked_subclass, ufunc_domain, ufunc_fills, filled, _DomainSafeDivide)


if BUG3907:
    def getdata(a, subok=True):
        """
        Return the data of a masked array as an ndarray.

        Return the data of `a` if `a` is a ``MaskedArray``, or return `a` as an
        ndarray if it is not.  Depending on `subok`, subclasses are passed on.

        Parameters
        ----------
        a : array_like
            Input ``MaskedArray``, alternatively a ndarray or a subclass thereof.
        subok : bool
            Whether to allow the output to be a ndarray subclass (True, default),
            or force it to be a `pure` ndarray (False).  In either case, input
            that would lead to an ndarray with object dtype (i.e., that cannot be
            well-represented as an ndarray) is returned as is.

        See Also
        --------
        getmask : Return the mask of a masked array, or nomask.
        getmaskarray : Return the mask of a masked array, or full array of False.

        Examples
        --------
        >>> import numpy.ma as ma
        >>> a = ma.masked_equal([[1,2],[3,4]], 2)
        >>> a
        masked_array(data =
         [[1 --]
         [3 4]],
              mask =
         [[False  True]
         [False False]],
              fill_value=999999)
        >>> ma.getdata(a)
        array([[1, 2],
               [3, 4]])

        Equivalently use the ``MaskedArray`` `data` attribute.

        >>> a.data
        array([[1, 2],
               [3, 4]])

        """
        try:
            data = a._data
        except AttributeError:
            data = np.array(a, copy=False, subok=subok)
            if data.dtype == object:
                return a
        else:
            if not subok:
                return data.view(ndarray)
        return data
    get_data = getdata


    def getmaskarray(arr):
        """
        Return the mask of a masked array, or full boolean array of False.

        Return the mask of `arr` as an ndarray if `arr` is a `MaskedArray` and
        the mask is not `nomask`, else return a full boolean array of False of
        the same shape as `arr`.

        Parameters
        ----------
        arr : array_like
            Input `MaskedArray` for which the mask is required.

        See Also
        --------
        getmask : Return the mask of a masked array, or nomask.
        getdata : Return the data of a masked array as an ndarray.

        Examples
        --------
        >>> import numpy.ma as ma
        >>> a = ma.masked_equal([[1,2],[3,4]], 2)
        >>> a
        masked_array(data =
         [[1 --]
         [3 4]],
              mask =
         [[False  True]
         [False False]],
              fill_value=999999)
        >>> ma.getmaskarray(a)
        array([[False,  True],
               [False, False]], dtype=bool)

        Result when mask == ``nomask``

        >>> b = ma.masked_array([[1,2],[3,4]])
        >>> b
        masked_array(data =
         [[1 2]
         [3 4]],
              mask =
         False,
              fill_value=999999)
        >>> >ma.getmaskarray(b)
        array([[False, False],
               [False, False]], dtype=bool)

        """
        mask = getmask(arr)
        if mask is nomask:
            mask = make_mask_none(np.shape(arr), getattr(arr, 'dtype', None))
        return mask

    class _MaskedBinaryOperation(Numpy_MaskedBinaryOperation):
        # https://github.com/numpy/numpy/pull/3907
        def __call__(self, a, b, *args, **kwargs):
            "Execute the call behavior."
            # Get the data, as ndarray
            (da, db) = (getdata(a), getdata(b))
            # Get the result
            with np.errstate():
                np.seterr(divide='ignore', invalid='ignore')
                result = self.f(da, db, *args, **kwargs)
            # check it worked
            if result is NotImplemented:
                return NotImplemented
            # Get the mask for the result
            (ma, mb) = (getmask(a), getmask(b))
            if ma is nomask:
                if mb is nomask:
                    m = nomask
                else:
                    m = umath.logical_or(getmaskarray(a), mb)
            elif mb is nomask:
                m = umath.logical_or(ma, getmaskarray(b))
            else:
                m = umath.logical_or(ma, mb)
            # Case 1. : scalar
            if not result.ndim:
                if m:
                    return masked
                return result
            # Case 2. : array
            # Revert result to da where masked
            if m.any():
                # any errors, just abort; impossible to guarantee masked values
                try:
                    np.copyto(result, 0, casting='unsafe', where=m)
                    # avoid using "*" since this may be overlaid
                    masked_da = umath.multiply(m, da)
                    # only add back if it can be cast safely
                    if np.can_cast(masked_da.dtype, result.dtype, casting='safe'):
                        result += masked_da
                except:
                    pass
            # Transforms to a (subclass of) MaskedArray
            masked_result = result.view(get_masked_subclass(a, b))
            masked_result._mask = m
            masked_result._update_from(result)
            return masked_result

        def outer(self, a, b):
            """Return the function applied to the outer product of a and b.

            """
            (da, db) = (getdata(a), getdata(b))
            d = self.f.outer(da, db)
            # check it worked
            if d is NotImplemented:
                return NotImplemented
            ma = getmask(a)
            mb = getmask(b)
            if ma is nomask and mb is nomask:
                m = nomask
            else:
                ma = getmaskarray(a)
                mb = getmaskarray(b)
                m = umath.logical_or.outer(ma, mb)
            if (not m.ndim) and m:
                return masked
            if m is not nomask:
                np.copyto(d, da, where=m)
            if d.shape:
                d = d.view(get_masked_subclass(a, b))
                d._mask = m
            return d


    class _DomainedBinaryOperation(Numpy_DomainedBinaryOperation):
        # https://github.com/numpy/numpy/pull/3907
        def __call__(self, a, b, *args, **kwargs):
            "Execute the call behavior."
            # Get the data
            (da, db) = (getdata(a), getdata(b))
            # Get the result
            with np.errstate():
                np.seterr(divide='ignore', invalid='ignore')
                result = self.f(da, db, *args, **kwargs)
            # check it worked
            if result is NotImplemented:
                return NotImplemented
            # Get the mask as a combination of the source masks and invalid
            m = ~umath.isfinite(result)
            m |= getmask(a)
            m |= getmask(b)
            # Apply the domain
            domain = ufunc_domain.get(self.f, None)
            if domain is not None:
                m |= filled(domain(da.view(np.ndarray),
                                   db.view(np.ndarray)), True)
            # Take care of the scalar case first
            if (not m.ndim):
                if m:
                    return masked
                else:
                    return result
            # When the mask is True, put back da if possible
            # any errors, just abort; impossible to guarantee masked values
            try:
                np.copyto(result, 0, casting='unsafe', where=m)
                # avoid using "*" since this may be overlaid
                masked_da = umath.multiply(m, da)
                # only add back if it can be cast safely
                if np.can_cast(masked_da.dtype, result.dtype, casting='safe'):
                    result += masked_da
            except:
                pass

            # Transforms to a (subclass of) MaskedArray
            masked_result = result.view(get_masked_subclass(a, b))
            masked_result._mask = m
            masked_result._update_from(result)
            return masked_result


    # Binary ufuncs ...............................................................
    # def attribute_wrapper(attr, umath_default):
    #     def wrapper(da, db, *args, **kwargs):
    #         if hasattr(da, attr):
    #             return getattr(da, attr)(db, *args, **kwargs)
    #         else:
    #             return umath_default(da, db, *args, **kwargs)
    #     return wrapper

    add = _MaskedBinaryOperation(umath.add)
    subtract = _MaskedBinaryOperation(umath.subtract)
    # add = _MaskedBinaryOperation(attribute_wrapper('__add__', umath.add))
    # subtract = _MaskedBinaryOperation(
    #     attribute_wrapper('__sub__', umath.subtract))
    multiply = _MaskedBinaryOperation(umath.multiply, 1, 1)
    # multiply = _MaskedBinaryOperation(
    #     attribute_wrapper('__mul__', umath.multiply), 1, 1)
    arctan2 = _MaskedBinaryOperation(umath.arctan2, 0.0, 1.0)
    equal = _MaskedBinaryOperation(umath.equal)
    equal.reduce = None
    not_equal = _MaskedBinaryOperation(umath.not_equal)
    not_equal.reduce = None
    less_equal = _MaskedBinaryOperation(umath.less_equal)
    less_equal.reduce = None
    greater_equal = _MaskedBinaryOperation(umath.greater_equal)
    greater_equal.reduce = None
    less = _MaskedBinaryOperation(umath.less)
    less.reduce = None
    greater = _MaskedBinaryOperation(umath.greater)
    greater.reduce = None
    logical_and = _MaskedBinaryOperation(umath.logical_and)
    alltrue = _MaskedBinaryOperation(umath.logical_and, 1, 1).reduce
    logical_or = _MaskedBinaryOperation(umath.logical_or)
    sometrue = logical_or.reduce
    logical_xor = _MaskedBinaryOperation(umath.logical_xor)
    bitwise_and = _MaskedBinaryOperation(umath.bitwise_and)
    bitwise_or = _MaskedBinaryOperation(umath.bitwise_or)
    bitwise_xor = _MaskedBinaryOperation(umath.bitwise_xor)
    hypot = _MaskedBinaryOperation(umath.hypot)
    # Domained binary ufuncs ......................................................
    divide = _DomainedBinaryOperation(umath.divide, _DomainSafeDivide(), 0, 1)
    # divide = _DomainedBinaryOperation(
    #     attribute_wrapper('__divide__', umath.divide),
    #     _DomainSafeDivide(), 0, 1)
    true_divide = _DomainedBinaryOperation(umath.true_divide,
                                           _DomainSafeDivide(), 0, 1)
    # true_divide = _DomainedBinaryOperation(
    #     attribute_wrapper('__truediv__', umath.true_divide),
    #     _DomainSafeDivide(), 0, 1)
    floor_divide = _DomainedBinaryOperation(umath.floor_divide,
                                            _DomainSafeDivide(), 0, 1)
    remainder = _DomainedBinaryOperation(umath.remainder,
                                         _DomainSafeDivide(), 0, 1)
    fmod = _DomainedBinaryOperation(umath.fmod, _DomainSafeDivide(), 0, 1)
    mod = _DomainedBinaryOperation(umath.mod, _DomainSafeDivide(), 0, 1)


if BUG4585:
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

    if BUG4576:
        def __str__(self):
            """String representation.

            """
            # see https://github.com/numpy/numpy/pull/4576  MERGED
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
            # see https://github.com/numpy/numpy/pull/4576  MERGED
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

    if BUG4586:
        def __getitem__(self, indx):
            """x.__getitem__(y) <==> x[y]

            Return the item described by i, as a masked array.

            """
            # see https://github.com/numpy/numpy/pull/4585

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
            # see https://github.com/numpy/numpy/pull/4585
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

    if BUG4585:  # only to ensure we use our own MaskedIterator
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

    if BUG3907:  # only to ensure redef's get picked up
        def __add__(self, other):
            "Add other to self, and return a new masked array."
            return add(self, other)
        #
        def __radd__(self, other):
            "Add other to self, and return a new masked array."
            return add(self, other)
        #
        def __sub__(self, other):
            "Subtract other to self, and return a new masked array."
            return subtract(self, other)
        #
        def __rsub__(self, other):
            "Subtract other to self, and return a new masked array."
            return subtract(other, self)
        #
        def __mul__(self, other):
            "Multiply other by self, and return a new masked array."
            return multiply(self, other)
        #
        def __rmul__(self, other):
            "Multiply other by self, and return a new masked array."
            return multiply(self, other)
        #
        def __div__(self, other):
            "Divide other into self, and return a new masked array."
            return divide(self, other)
        #
        def __truediv__(self, other):
            "Divide other into self, and return a new masked array."
            return true_divide(self, other)
        #
        def __rtruediv__(self, other):
            "Divide other into self, and return a new masked array."
            return true_divide(other, self)
        #
        def __floordiv__(self, other):
            "Divide other into self, and return a new masked array."
            return floor_divide(self, other)
        #
        def __rfloordiv__(self, other):
            "Divide other into self, and return a new masked array."
            return floor_divide(other, self)
        #
        def __pow__(self, other):
            "Raise self to the power other, masking the potential NaNs/Infs"
            return power(self, other)
        #
        def __rpow__(self, other):
            "Raise self to the power other, masking the potential NaNs/Infs"
            return power(other, self)
        #............................................

    if BUGTBD:
        def __array_wrap__(self, obj, context=None):
            """
            Special hook for ufuncs.
            Wraps the numpy array and sets the mask according to context.
            """
            result = obj.view(type(self))
            result._update_from(self)
            #..........
            if context is not None:
                result._mask = result._mask.copy()
                (func, args, _) = context
                m = reduce(mask_or, [getmaskarray(arg) for arg in args])
                # Get the domain mask................
                domain = ufunc_domain.get(func, None)
                if domain is not None:
                    # Take the domain, and make sure it's a ndarray
                    args_array = [arg.view(np.ndarray) for arg in args]
                    if len(args) > 2:
                        d = filled(reduce(domain, args_array), True)
                    else:
                        d = filled(domain(*args_array), True)
                    # Fill the result where the domain is wrong
                    try:
                        # Binary domain: take the last value
                        fill_value = ufunc_fills[func][-1]
                    except TypeError:
                        # Unary domain: just use this one
                        fill_value = ufunc_fills[func]
                    except KeyError:
                        # Domain not recognized, use fill_value instead
                        fill_value = self.fill_value
                    result = result.copy()
                    np.copyto(result, fill_value, where=d)
                    # Update the mask
                    if m is nomask:
                        if d is not nomask:
                            m = d
                    else:
                        # Don't modify inplace, we risk back-propagation
                        m = (m | d)
                # Make sure the mask has the proper size
                if result.shape == () and m:
                    return masked
                else:
                    result._mask = m
                    result._sharedmask = False
            #....
            return result
