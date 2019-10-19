from itertools import count, product
from decimal import Decimal

import pytest

import numpy as np

from astropy.time import Time, TimeFormat, TimeNumeric


class SpecificException(ValueError):
    pass


@pytest.fixture
def custom_format_name():
    for i in count():
        if not i:
            custom = f"custom_format_name"
        else:
            custom = f"custom_format_name_{i}"
        if custom not in Time.FORMATS:
            break
    yield custom
    Time.FORMATS.pop(custom, None)


def test_custom_time_format_set_jds_exception(custom_format_name):
    class Custom(TimeFormat):
        name = custom_format_name

        def set_jds(self, val, val2):
            raise SpecificException

    try:
        Time(7.0, format=custom_format_name)
    except ValueError as e:
        assert hasattr(e, "__cause__") and isinstance(e.__cause__, SpecificException)


def test_custom_time_format_val_type_exception(custom_format_name):
    class Custom(TimeFormat):
        name = custom_format_name

        def _check_val_type(self, val, val2):
            raise SpecificException

    try:
        Time(7.0, format=custom_format_name)
    except ValueError as e:
        assert hasattr(e, "__cause__") and isinstance(e.__cause__, SpecificException)


def test_custom_time_format_value_exception(custom_format_name):
    class Custom(TimeFormat):
        name = custom_format_name

        def set_jds(self, val, val2):
            self.jd1, self.jd2 = val, val2

        @property
        def value(self):
            raise SpecificException

    t = Time.now()
    with pytest.raises(SpecificException):
        getattr(t, custom_format_name)


def test_custom_time_format_fine(custom_format_name):
    class Custom(TimeFormat):
        name = custom_format_name

        def set_jds(self, val, val2):
            self.jd1, self.jd2 = val, val2

        @property
        def value(self):
            return self.jd1 + self.jd2

    t = Time.now()
    getattr(t, custom_format_name)


def test_custom_time_format_forgot_property(custom_format_name):
    class Custom(TimeFormat):
        name = custom_format_name

        def set_jds(self, val, val2):
            self.jd1, self.jd2 = val, val2

        def value(self):
            return self.jd1, self.jd2

    t = Time.now()
    with pytest.raises(AttributeError):
        getattr(t, custom_format_name)

    t.format = custom_format_name
    with pytest.raises(AttributeError):
        t.value

    with pytest.raises(AttributeError):
        Time(7, 9, format=custom_format_name).value


def test_custom_time_format_problematic_name():
    assert "sort" not in Time.FORMATS, "problematic name in default FORMATS!"
    assert hasattr(Time, "sort")

    try:

        class Custom(TimeFormat):
            name = "sort"

            def set_jds(self, val, val2):
                self.jd1, self.jd2 = val, val2

            @property
            def value(self):
                return self.jd1, self.jd2

        t = Time.now()
        assert t.sort() == t, "bogus time format clobbers everyone's Time objects"

        t.format = "sort"
        if not isinstance(t.value, tuple):
            pytest.xfail("No good way to detect that `sort` is invalid")

        assert Time(7, 9, format="sort").value == (7, 9)

    finally:
        Time.FORMATS.pop("sort", None)


@pytest.mark.parametrize(
    "n, i",
    product(
        [
            "bogus with_underscore",
            "bogus with_two_underscores",
            "bogus with_str",
            "bogus with_long",
            "bogus with_decimal",
        ],
        [58000, 58000.0, np.longdouble(58000.0), "58000", Decimal(58000)],
    ),
)
def test_custom_time_format_name_with_underscore_like_subfmt(n, i):
    assert n not in Time.FORMATS, "problematic name in default FORMATS!"

    try:

        class Custom(TimeNumeric):
            name = n

            def set_jds(self, val, val2):
                self.jd1, self.jd2 = val, val2

        t = Time(i, format=n, scale="tai")

        assert isinstance(getattr(t, n), float)
        assert isinstance(getattr(t, n + "_str"), str)
        assert isinstance(getattr(t, n + "_long"), np.longdouble)
        assert isinstance(getattr(t, n + "_decimal"), Decimal)

    finally:
        Time.FORMATS.pop(n, None)
