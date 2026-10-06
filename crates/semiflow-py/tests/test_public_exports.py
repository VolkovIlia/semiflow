"""Every class/function the native module registers is re-exported by `semiflow`.

`ReverseHeat1D` was registered natively and documented, yet `from semiflow import
ReverseHeat1D` raised ImportError because `__init__.py` never re-exported it.
"""

import numpy as np

import semiflow
from semiflow import semiflow as native  # pyright: ignore[reportMissingImports]


def test_native_names_are_reexported() -> None:
    public = {n for n in dir(native) if not n.startswith("_")}
    missing = sorted(n for n in public if not hasattr(semiflow, n))
    assert not missing, f"registered natively but not re-exported: {missing}"


def test_all_matches_exports() -> None:
    assert all(hasattr(semiflow, n) for n in semiflow.__all__)


def test_error_has_kind_attribute() -> None:
    try:
        semiflow.Heat1D(1.0, 0.0, 8, np.zeros(8))  # xmin > xmax
    except semiflow.SemiflowError as err:
        assert isinstance(err.kind, str) and err.kind
        assert str(err).startswith(f"[{err.kind}]")
    else:
        raise AssertionError("expected SemiflowError")


def test_adr_0202_names_are_public() -> None:
    """ADR-0202: SpdResolvent + phi_combination are re-exported and listed in __all__."""
    for name in ("SpdResolvent", "phi_combination"):
        assert hasattr(semiflow, name), name
        assert name in semiflow.__all__, name
        assert hasattr(native, name), name
