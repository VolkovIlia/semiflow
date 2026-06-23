"""Type stubs for semiflow — PyO3 bindings to semiflow-core.

These stubs mirror the actual #[pymethods] signatures defined in
crates/semiflow-py/src/state.rs / src/error.rs / src/lib.rs.
"""

from typing import Any, Callable, Literal, Union, final

import numpy as np
from numpy.typing import NDArray

BoundaryLiteral = Literal["reflect", "periodic", "zero", "linear"]

@final
class Heat1D:
    """1-D heat equation state with unit diffusion (a = 1).

    Solves du/dt = d^2u/dx^2 on [xmin, xmax] with n uniformly-spaced nodes.
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        n: int,
        u0: NDArray[np.float64],
        *,
        boundary: BoundaryLiteral = "reflect",
        dtype: Literal["f64", "f32"] = "f64",
    ) -> None:
        """Create state on [xmin, xmax] with n grid nodes and initial datum u0.

        Parameters
        ----------
        xmin : float
            Left boundary.
        xmax : float
            Right boundary (must be > xmin).
        n : int
            Number of grid nodes (must be >= 4).
        u0 : NDArray[np.float64]
            Initial condition; must have length n and contain only finite values.
        boundary : str, optional
            Boundary policy (keyword-only).  One of ``"reflect"`` (default,
            mirror/Neumann zero-flux), ``"periodic"`` (wrap), ``"zero"``
            (Dirichlet zero), ``"linear"`` (extrapolate).
        dtype : str, optional
            ``"f64"`` (default) or ``"f32"``.  When ``"f32"``, each call to
            ``evolve()`` runs the generic f32 Chernoff kernel; the internal
            state is always stored as float64.  fp16 is REJECTED (ADR-0115).

        Raises
        ------
        SemiflowError
            kind='GridMismatch' if xmin >= xmax or n < 4 or len(u0) != n.
            kind='NanInf' if u0 contains NaN or Inf.
            kind='OutOfDomain' if boundary or dtype is not a recognised string.
        """
        ...

    def evolve(self, t: float, n_steps: int = 100) -> None:
        """Advance the state by time t using n_steps Chernoff iterations.

        Mutates self in-place; returns None.  The GIL is released during the
        inner Rust compute loop (ADR-0031) so concurrent Python threads make
        progress during long calls.

        Parameters
        ----------
        t : float
            Time to advance.  Must be non-negative and finite.
        n_steps : int, optional
            Number of Chernoff steps (default 100).  Must be >= 1.

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' if t < 0, t is non-finite, or n_steps == 0.
        """
        ...

    def evolve_chunked(
        self,
        t: float,
        total_steps: int,
        chunk_steps: int,
        progress: object | None = None,
    ) -> None:
        """Chunked GIL-cooperative evolve (ADR-0141, v8.1.0 A-4).

        Runs ``total_steps`` Chernoff iterations in chunks of ``chunk_steps``,
        releasing the GIL for each chunk's pure-Rust compute and re-acquiring
        it to call the optional ``progress(done, total)`` callback.

        The result is **bit-identical (0 ULP)** to ``evolve(t, total_steps)``.
        Chunking only changes *when* the GIL is released, not the arithmetic.

        GIL-safety: ``progress`` is called only while the GIL is held, never
        from inside ``py.detach``.

        Parameters
        ----------
        t : float
            Time to advance.  Must be non-negative and finite.
        total_steps : int
            Total number of Chernoff steps.  Must be >= 1.
        chunk_steps : int
            Steps per GIL-release chunk.  Must be >= 1.  The last chunk may
            be smaller if ``total_steps`` is not divisible.
        progress : callable | None, optional
            If provided, called as ``progress(done: int, total: int)`` after
            each completed chunk (GIL held).  A Python exception raised inside
            ``progress`` propagates and stops the evolution cleanly.

        Raises
        ------
        SemiflowError
            ``kind='OutOfDomain'`` if ``t < 0``, ``total_steps == 0``,
            or ``chunk_steps == 0``.
        """
        ...

    def values(self) -> NDArray[np.float64]:
        """Return the current grid values as a 1-D float64 numpy array.

        Returns a *copy* of the internal state; mutations to the returned
        array do not affect this Heat1D object.
        """
        ...

    def __len__(self) -> int:
        """Return the number of grid nodes."""
        ...

    @staticmethod
    def with_a_function(
        xmin: float,
        xmax: float,
        n: int,
        a: object,
        a_prime: object,
        a_double_prime: object,
        a_norm_bound: float,
        u0: NDArray[np.float64],
        *,
        boundary: BoundaryLiteral = "reflect",
    ) -> "Heat1D":
        """Create a Heat1D with a variable diffusion coefficient via callables.

        Parameters
        ----------
        xmin, xmax : float
            Domain boundaries.
        n : int
            Number of grid nodes (>= 4).
        a : callable
            ``a(x: float) -> float``; must be positive on ``[xmin, xmax]``.
        a_prime : callable
            ``a'(x: float) -> float``.
        a_double_prime : callable
            ``a''(x: float) -> float``.
        a_norm_bound : float
            Upper bound on ``‖a‖_∞``; must be > 0.
        u0 : NDArray[np.float64]
            Initial condition; length n, all finite.
        boundary : str, optional
            Boundary policy (keyword-only); default ``"reflect"``.

        Raises
        ------
        SemiflowError
            kind='GridMismatch' if n < 4 or len(u0) != n.
            kind='NanInf' if u0 contains NaN or Inf.
            kind='OutOfDomain' if boundary is not recognised.
        """
        ...

    @staticmethod
    def with_a_array(
        xmin: float,
        xmax: float,
        n: int,
        a: NDArray[np.float64],
        u0: NDArray[np.float64],
        *,
        a_prime: NDArray[np.float64] | None = None,
        a_double_prime: NDArray[np.float64] | None = None,
        a_norm_bound: float | None = None,
        boundary: BoundaryLiteral = "reflect",
    ) -> "Heat1D":
        """Create a Heat1D from pre-sampled diffusion-coefficient arrays.

        **GIL-zero-cost advantage**: unlike ``with_a_function``, the coefficient
        closures produced here are pure-Rust backed by ``Arc<Vec<f64>>``.
        They do not re-acquire the GIL during ``evolve``, so the three-phase
        GIL-release pattern (ADR-0031) is fully effective.  For large grids or
        many time steps this can be 10× faster than the callback path.

        Parameters
        ----------
        xmin, xmax : float
            Domain boundaries.
        n : int
            Number of grid nodes (>= 4).
        a : NDArray[np.float64]
            Pre-sampled ``a(x_i)`` values; length n; all finite.
        u0 : NDArray[np.float64]
            Initial condition; length n, all finite.
        a_prime : NDArray[np.float64] | None, optional
            Pre-sampled ``a'(x_i)``.  Computed via 4th-order FD if ``None``.
        a_double_prime : NDArray[np.float64] | None, optional
            Pre-sampled ``a''(x_i)``.  Computed via 4th-order FD if ``None``.
        a_norm_bound : float | None, optional
            Upper bound on ``‖a‖_∞``.  Set to ``1.1 * max(a)`` if ``None``.
        boundary : str, optional
            Boundary policy (keyword-only); default ``"reflect"``.

        Raises
        ------
        SemiflowError
            kind='GridMismatch' if any array has length != n.
            kind='NanInf' if any array contains NaN or Inf.
            kind='OutOfDomain' if boundary is not recognised.
        """
        ...


@final
class Heat1D4th:
    """1-D heat equation with 4th-order Chernoff diffusion kernel.

    Solves du/dt = d^2u/dx^2 (or variable a(x) via ``with_a_array``) using
    a Diffusion4thChernoff kernel (order 4, formula (6) Remizov 2025).
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        n: int,
        u0: NDArray[np.float64],
        *,
        boundary: BoundaryLiteral = "reflect",
    ) -> None:
        """Create state on [xmin, xmax] with n grid nodes and initial datum u0.

        Parameters
        ----------
        xmin : float
            Left boundary.
        xmax : float
            Right boundary (must be > xmin).
        n : int
            Number of grid nodes (must be >= 4).
        u0 : NDArray[np.float64]
            Initial condition; must have length n and contain only finite values.
        boundary : str, optional
            Boundary policy (keyword-only).  One of ``"reflect"`` (default),
            ``"periodic"``, ``"zero"``, ``"linear"``.

        Raises
        ------
        SemiflowError
            kind='GridMismatch' if xmin >= xmax or n < 4 or len(u0) != n.
            kind='NanInf' if u0 contains NaN or Inf.
            kind='OutOfDomain' if boundary is not recognised.
        """
        ...

    def evolve(self, t: float, n_steps: int = 100) -> None:
        """Advance the state by time t using n_steps Chernoff iterations.

        Mutates self in-place; returns None.  The GIL is released during the
        inner Rust compute loop (ADR-0031).

        Parameters
        ----------
        t : float
            Time to advance.  Must be non-negative and finite.
        n_steps : int, optional
            Number of Chernoff steps (default 100).  Must be >= 1.

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' if t < 0, t is non-finite, or n_steps == 0.
        """
        ...

    def values(self) -> NDArray[np.float64]:
        """Return the current grid values as a 1-D float64 numpy array (copy)."""
        ...

    def __len__(self) -> int:
        """Return the number of grid nodes."""
        ...

    @staticmethod
    def with_a_array(
        xmin: float,
        xmax: float,
        n: int,
        a: NDArray[np.float64],
        u0: NDArray[np.float64],
        *,
        a_prime: NDArray[np.float64] | None = None,
        a_double_prime: NDArray[np.float64] | None = None,
        a_norm_bound: float | None = None,
        boundary: BoundaryLiteral = "reflect",
    ) -> "Heat1D4th":
        """Create a Heat1D4th from pre-sampled diffusion-coefficient arrays.

        Parameters
        ----------
        xmin, xmax : float
            Domain boundaries.
        n : int
            Number of grid nodes (>= 4).
        a : NDArray[np.float64]
            Pre-sampled ``a(x_i)`` values; length n; all finite and positive.
        u0 : NDArray[np.float64]
            Initial condition; length n, all finite.
        a_prime : NDArray[np.float64] | None, optional
            Pre-sampled ``a'(x_i)``.  Computed via 4th-order FD if ``None``.
        a_double_prime : NDArray[np.float64] | None, optional
            Pre-sampled ``a''(x_i)``.  Computed via 4th-order FD if ``None``.
        a_norm_bound : float | None, optional
            Upper bound on ``‖a‖_∞``.  Set to ``1.1 * max(a)`` if ``None``.
        boundary : str, optional
            Boundary policy (keyword-only); default ``"reflect"``.

        Raises
        ------
        SemiflowError
            kind='GridMismatch' if any array has length != n.
            kind='NanInf' if any array contains NaN or Inf.
            kind='OutOfDomain' if boundary is not recognised.
        """
        ...


@final
class Heat1D6th:
    """1-D heat equation with 6th-order Chernoff diffusion kernel.

    Solves du/dt = d^2u/dx^2 (or variable a(x) via ``with_a_array``) using
    a Diffusion6thChernoff kernel (order 6, formula (6) Remizov 2025).
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        n: int,
        u0: NDArray[np.float64],
        *,
        boundary: BoundaryLiteral = "reflect",
    ) -> None:
        """Create state on [xmin, xmax] with n grid nodes and initial datum u0.

        Parameters
        ----------
        xmin : float
            Left boundary.
        xmax : float
            Right boundary (must be > xmin).
        n : int
            Number of grid nodes (must be >= 4).
        u0 : NDArray[np.float64]
            Initial condition; must have length n and contain only finite values.
        boundary : str, optional
            Boundary policy (keyword-only).  One of ``"reflect"`` (default),
            ``"periodic"``, ``"zero"``, ``"linear"``.

        Raises
        ------
        SemiflowError
            kind='GridMismatch' if xmin >= xmax or n < 4 or len(u0) != n.
            kind='NanInf' if u0 contains NaN or Inf.
            kind='OutOfDomain' if boundary is not recognised.
        """
        ...

    def evolve(self, t: float, n_steps: int = 100) -> None:
        """Advance the state by time t using n_steps Chernoff iterations.

        Mutates self in-place; returns None.  The GIL is released during the
        inner Rust compute loop (ADR-0031).

        Parameters
        ----------
        t : float
            Time to advance.  Must be non-negative and finite.
        n_steps : int, optional
            Number of Chernoff steps (default 100).  Must be >= 1.

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' if t < 0, t is non-finite, or n_steps == 0.
        """
        ...

    def values(self) -> NDArray[np.float64]:
        """Return the current grid values as a 1-D float64 numpy array (copy)."""
        ...

    def __len__(self) -> int:
        """Return the number of grid nodes."""
        ...

    @staticmethod
    def with_a_array(
        xmin: float,
        xmax: float,
        n: int,
        a: NDArray[np.float64],
        u0: NDArray[np.float64],
        *,
        a_prime: NDArray[np.float64] | None = None,
        a_double_prime: NDArray[np.float64] | None = None,
        a_norm_bound: float | None = None,
        boundary: BoundaryLiteral = "reflect",
    ) -> "Heat1D6th":
        """Create a Heat1D6th from pre-sampled diffusion-coefficient arrays.

        Parameters
        ----------
        xmin, xmax : float
            Domain boundaries.
        n : int
            Number of grid nodes (>= 4).
        a : NDArray[np.float64]
            Pre-sampled ``a(x_i)`` values; length n; all finite and positive.
        u0 : NDArray[np.float64]
            Initial condition; length n, all finite.
        a_prime : NDArray[np.float64] | None, optional
            Pre-sampled ``a'(x_i)``.  Computed via 4th-order FD if ``None``.
        a_double_prime : NDArray[np.float64] | None, optional
            Pre-sampled ``a''(x_i)``.  Computed via 4th-order FD if ``None``.
        a_norm_bound : float | None, optional
            Upper bound on ``‖a‖_∞``.  Set to ``1.1 * max(a)`` if ``None``.
        boundary : str, optional
            Boundary policy (keyword-only); default ``"reflect"``.

        Raises
        ------
        SemiflowError
            kind='GridMismatch' if any array has length != n.
            kind='NanInf' if any array contains NaN or Inf.
            kind='OutOfDomain' if boundary is not recognised.
        """
        ...


@final
class DriftReaction1D:
    """1-D drift + reaction equation: du/dt = b(x) du/dx + c(x) u.

    Uses DriftReactionChernoff kernel (order 2, formula (6) Remizov 2025).
    Default coefficients: b(x) = 0.5, c(x) = 0.0.
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        n: int,
        u0: NDArray[np.float64],
        *,
        b: float = 0.5,
        c: float = 0.0,
        boundary: BoundaryLiteral = "reflect",
    ) -> None:
        """Create state with constant drift b and reaction c on [xmin, xmax].

        Parameters
        ----------
        xmin : float
            Left boundary.
        xmax : float
            Right boundary (must be > xmin).
        n : int
            Number of grid nodes (must be >= 4).
        u0 : NDArray[np.float64]
            Initial condition; must have length n and contain only finite values.
        b : float, optional
            Constant drift coefficient (default 0.5).
        c : float, optional
            Constant reaction coefficient (default 0.0).
        boundary : str, optional
            Boundary policy (keyword-only).  One of ``"reflect"`` (default),
            ``"periodic"``, ``"zero"``, ``"linear"``.

        Raises
        ------
        SemiflowError
            kind='GridMismatch' if xmin >= xmax or n < 4 or len(u0) != n.
            kind='NanInf' if u0 contains NaN or Inf.
            kind='OutOfDomain' if boundary is not recognised.
        """
        ...

    def evolve(self, t: float, n_steps: int = 100) -> None:
        """Advance the state by time t using n_steps Chernoff iterations.

        Mutates self in-place; returns None.  The GIL is released during the
        inner Rust compute loop (ADR-0031).

        Parameters
        ----------
        t : float
            Time to advance.  Must be non-negative and finite.
        n_steps : int, optional
            Number of Chernoff steps (default 100).  Must be >= 1.

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' if t < 0, t is non-finite, or n_steps == 0.
        """
        ...

    def values(self) -> NDArray[np.float64]:
        """Return the current grid values as a 1-D float64 numpy array (copy)."""
        ...

    def __len__(self) -> int:
        """Return the number of grid nodes."""
        ...

    @staticmethod
    def with_arrays(
        xmin: float,
        xmax: float,
        n: int,
        b: NDArray[np.float64],
        c: NDArray[np.float64],
        c_norm_bound: float,
        u0: NDArray[np.float64],
        *,
        boundary: BoundaryLiteral = "reflect",
    ) -> "DriftReaction1D":
        """Create a DriftReaction1D from pre-sampled variable-coefficient arrays.

        **GIL-zero-cost advantage**: coefficient closures are pure-Rust backed
        by ``Arc<Vec<f64>>`` Catmull-Rom interpolants; no GIL re-acquire during
        ``evolve`` (ADR-0031).

        Parameters
        ----------
        xmin, xmax : float
            Domain boundaries.
        n : int
            Number of grid nodes (>= 4).
        b : NDArray[np.float64]
            Pre-sampled ``b(x_i)`` drift values; length n; all finite.
        c : NDArray[np.float64]
            Pre-sampled ``c(x_i)`` reaction values; length n; all finite.
        c_norm_bound : float
            Upper bound on ``‖b‖_∞ + ‖c‖_∞ · dt`` (stability factor).
            Must be > 0.
        u0 : NDArray[np.float64]
            Initial condition; length n, all finite.
        boundary : str, optional
            Boundary policy (keyword-only); default ``"reflect"``.

        Raises
        ------
        SemiflowError
            kind='GridMismatch' if any array has length != n.
            kind='NanInf' if any array contains NaN or Inf.
            kind='OutOfDomain' if boundary is not recognised.
        """
        ...

    def evolve_with_time_schedule(
        self,
        t_final: float,
        n_steps_per_segment: int,
        b_schedule: NDArray[np.float64],
        *,
        c: float = 0.0,
    ) -> None:
        """Evolve with a piecewise-constant-in-time ``b`` schedule (D3 — ADR-0113).

        The ``b_schedule`` array contains ``n_segments`` constant values of the
        drift coefficient ``b(t)`` — one per segment on a uniform partition of
        ``[0, t_final]``.  ``c`` is a spatially constant scalar.

        **GIL policy (ADR-0034)**: ``b_schedule`` is pre-sampled; no Python
        callback enters the Rust loop.

        **Scope note**: genuine joint space-time ``b(x, t)`` is out of scope.
        This covers time-varying-but-spatially-constant (piecewise) schedules.

        Parameters
        ----------
        t_final : float
            Total evolution time.  Must be finite and >= 0.
        n_steps_per_segment : int
            Number of Chernoff steps per segment.  Must be >= 1.
        b_schedule : NDArray[np.float64]
            Drift coefficient for each segment; length = n_segments.
        c : float, optional
            Constant reaction coefficient (default 0.0).

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' on invalid parameters or empty schedule.
        """
        ...


@final
class Shift1D:
    """Universal 1-D PDE: du/dt = a(x) d^2u/dx^2 + b(x) du/dx + c(x) u.

    Uses ShiftChernoff1D kernel (order 1, formula (6) Remizov 2025).
    Default coefficients: a(x) = 0.5, b(x) = 0.0, c(x) = 0.0.
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        n: int,
        u0: NDArray[np.float64],
        *,
        a: float = 0.5,
        b: float = 0.0,
        c: float = 0.0,
        boundary: BoundaryLiteral = "reflect",
    ) -> None:
        """Create state with constant a, b, c on [xmin, xmax].

        Parameters
        ----------
        xmin : float
            Left boundary.
        xmax : float
            Right boundary (must be > xmin).
        n : int
            Number of grid nodes (must be >= 4).
        u0 : NDArray[np.float64]
            Initial condition; must have length n and contain only finite values.
        a : float, optional
            Constant diffusion coefficient (default 0.5).
        b : float, optional
            Constant drift coefficient (default 0.0).
        c : float, optional
            Constant reaction coefficient (default 0.0).
        boundary : str, optional
            Boundary policy (keyword-only).  One of ``"reflect"`` (default),
            ``"periodic"``, ``"zero"``, ``"linear"``.

        Raises
        ------
        SemiflowError
            kind='GridMismatch' if xmin >= xmax or n < 4 or len(u0) != n.
            kind='NanInf' if u0 contains NaN or Inf.
            kind='OutOfDomain' if boundary is not recognised.
        """
        ...

    def evolve(self, t: float, n_steps: int = 100) -> None:
        """Advance the state by time t using n_steps Chernoff iterations.

        Mutates self in-place; returns None.  The GIL is released during the
        inner Rust compute loop (ADR-0031).

        Parameters
        ----------
        t : float
            Time to advance.  Must be non-negative and finite.
        n_steps : int, optional
            Number of Chernoff steps (default 100).  Must be >= 1.

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' if t < 0, t is non-finite, or n_steps == 0.
        """
        ...

    def values(self) -> NDArray[np.float64]:
        """Return the current grid values as a 1-D float64 numpy array (copy)."""
        ...

    def __len__(self) -> int:
        """Return the number of grid nodes."""
        ...

    @staticmethod
    def with_arrays(
        xmin: float,
        xmax: float,
        n: int,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        c: NDArray[np.float64],
        c_norm_bound: float,
        u0: NDArray[np.float64],
        *,
        boundary: BoundaryLiteral = "reflect",
    ) -> "Shift1D":
        """Create a Shift1D from pre-sampled variable-coefficient arrays.

        **GIL-zero-cost advantage**: coefficient closures are pure-Rust backed
        by ``Arc<Vec<f64>>`` Catmull-Rom interpolants; no GIL re-acquire during
        ``evolve`` (ADR-0031).

        Parameters
        ----------
        xmin, xmax : float
            Domain boundaries.
        n : int
            Number of grid nodes (>= 4).
        a : NDArray[np.float64]
            Pre-sampled ``a(x_i)`` diffusion values; length n; all finite.
        b : NDArray[np.float64]
            Pre-sampled ``b(x_i)`` drift values; length n; all finite.
        c : NDArray[np.float64]
            Pre-sampled ``c(x_i)`` reaction values; length n; all finite.
        c_norm_bound : float
            Upper bound on the combined norm (stability factor).  Must be > 0.
        u0 : NDArray[np.float64]
            Initial condition; length n, all finite.
        boundary : str, optional
            Boundary policy (keyword-only); default ``"reflect"``.

        Raises
        ------
        SemiflowError
            kind='GridMismatch' if any array has length != n.
            kind='NanInf' if any array contains NaN or Inf.
            kind='OutOfDomain' if boundary is not recognised.
        """
        ...

    def evolve_with_time_schedule(
        self,
        t_final: float,
        n_steps_per_segment: int,
        a_schedule: NDArray[np.float64],
        *,
        b: float = 0.0,
        c: float = 0.0,
    ) -> None:
        """Evolve with a piecewise-constant-in-time ``a`` schedule (D3 — ADR-0113).

        The ``a_schedule`` array contains ``n_segments`` constant values of the
        diffusion coefficient ``a(t)`` — one per segment on a uniform partition
        of ``[0, t_final]``.  ``b`` and ``c`` are spatially constant scalars.

        **GIL policy (ADR-0034)**: ``a_schedule`` is pre-sampled; no Python
        callback enters the Rust loop.  Safe for long time horizons.

        **Scope note**: genuine joint space-time ``a(x, t)`` is out of scope
        (core closures are purely spatial).  This covers time-varying-but-
        spatially-constant (piecewise) schedules.

        Parameters
        ----------
        t_final : float
            Total evolution time.  Must be finite and >= 0.
        n_steps_per_segment : int
            Number of Chernoff steps per segment.  Must be >= 1.
        a_schedule : NDArray[np.float64]
            Diffusion coefficient for each segment; length = n_segments.
            All values must be > 0.
        b : float, optional
            Constant drift coefficient (default 0.0).
        c : float, optional
            Constant reaction coefficient (default 0.0).

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' on invalid parameters or empty schedule.
        """
        ...


@final
class Schrodinger1D:
    """1-D Schrödinger equation state: ``iψ_t = (−Δ + V(x))ψ``.

    Backed by ``SchrodingerChernoff<f64>`` — palindromic Strang splitting,
    globally order 2, unitary by construction (ADR-0057).
    The GIL is released during :meth:`evolve` (ADR-0031).
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        n: int,
        psi0: NDArray[np.complex128],
        *,
        boundary: BoundaryLiteral = "reflect",
    ) -> None:
        """Create a free-particle state (V = 0) from a complex128 initial array.

        Parameters
        ----------
        xmin : float
            Left boundary.
        xmax : float
            Right boundary (must be > xmin).
        n : int
            Number of grid nodes (must be >= 4).
        psi0 : NDArray[np.complex128]
            Initial wavefunction; length n; all finite.
        boundary : str, optional
            Boundary policy (keyword-only); one of ``"reflect"`` (default),
            ``"periodic"``, ``"zero"``, ``"linear"``.

        Raises
        ------
        SemiflowError
            kind='GridMismatch' if xmin >= xmax or n < 4 or len(psi0) != n.
            kind='NanInf' if psi0 contains NaN or Inf.
            kind='OutOfDomain' if boundary is not recognised.
        """
        ...

    @staticmethod
    def from_parts(
        xmin: float,
        xmax: float,
        n: int,
        psi_re: NDArray[np.float64],
        psi_im: NDArray[np.float64],
        *,
        boundary: BoundaryLiteral = "reflect",
    ) -> "Schrodinger1D":
        """Create a free-particle state from separate real/imaginary float64 arrays.

        Equivalent to ``Schrodinger1D(xmin, xmax, n, psi_re + 1j*psi_im, ...)``.

        Parameters
        ----------
        psi_re : NDArray[np.float64]
            Real part of ψ₀; length n; all finite.
        psi_im : NDArray[np.float64]
            Imaginary part of ψ₀; length n; all finite.
        boundary : str, optional
            Boundary policy (keyword-only); default ``"reflect"``.

        Raises
        ------
        SemiflowError
            kind='GridMismatch' / 'NanInf' / 'OutOfDomain' on invalid inputs.
        """
        ...

    @staticmethod
    def with_potential(
        xmin: float,
        xmax: float,
        n: int,
        v: NDArray[np.float64],
        psi0: NDArray[np.complex128],
        *,
        boundary: BoundaryLiteral = "reflect",
    ) -> "Schrodinger1D":
        """Create state with a pre-sampled potential V(x) and complex128 ψ₀.

        Parameters
        ----------
        v : NDArray[np.float64]
            Pre-sampled ``V(x_i)`` values; length n; all finite.
        psi0 : NDArray[np.complex128]
            Initial wavefunction; length n; all finite.
        boundary : str, optional
            Boundary policy (keyword-only); default ``"reflect"``.

        Raises
        ------
        SemiflowError
            kind='GridMismatch' / 'NanInf' / 'OutOfDomain' on invalid inputs.
        """
        ...

    @staticmethod
    def with_potential_parts(
        xmin: float,
        xmax: float,
        n: int,
        v: NDArray[np.float64],
        psi_re: NDArray[np.float64],
        psi_im: NDArray[np.float64],
        *,
        boundary: BoundaryLiteral = "reflect",
    ) -> "Schrodinger1D":
        """Create state with potential V(x) and separate real/imag float64 arrays.

        Convenience variant combining ``with_potential`` and ``from_parts``.

        Parameters
        ----------
        v : NDArray[np.float64]
            Pre-sampled ``V(x_i)`` values; length n; all finite.
        psi_re : NDArray[np.float64]
            Real part of ψ₀; length n; all finite.
        psi_im : NDArray[np.float64]
            Imaginary part of ψ₀; length n; all finite.
        boundary : str, optional
            Boundary policy (keyword-only); default ``"reflect"``.

        Raises
        ------
        SemiflowError
            kind='GridMismatch' / 'NanInf' / 'OutOfDomain' on invalid inputs.
        """
        ...

    def evolve(self, t: float, n_steps: int = 200) -> None:
        """Advance the wavefunction by time t using n_steps Chernoff steps.

        Mutates self in-place; returns None.  The GIL is released during the
        inner Rust compute loop (ADR-0031 three-phase pattern).

        **Negative t (D2 — ADR-0113)**: t may be negative for backward
        (time-reversed) unitary evolution.  The palindromic Strang kernel
        satisfies S(−τ) = S(τ)⁻¹ exactly (verified round-trip residual
        1.19e-13 for Schrodinger1D).  Norm is preserved to machine precision.

        Parameters
        ----------
        t : float
            Time to advance.  Must be finite; may be negative for backward
            unitary evolution.
        n_steps : int, optional
            Number of Chernoff steps (default 200).  Must be >= 1.

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' if t is non-finite or n_steps == 0.
        """
        ...

    def values(self) -> NDArray[np.complex128]:
        """Return current wavefunction as a complex128 numpy array of length n.

        Returns a copy of the internal state; mutations do not affect this object.
        The returned dtype is ``numpy.complex128`` (= ``numpy.cdouble``).
        """
        ...

    def values_parts(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (psi_re, psi_im) as two float64 numpy arrays of length n.

        Each returned array is a copy.
        ``values().real == values_parts()[0]`` and
        ``values().imag == values_parts()[1]`` hold exactly (no rounding).
        """
        ...

    def __len__(self) -> int:
        """Return the number of grid nodes n."""
        ...

    def norm_squared(self) -> float:
        """Return Σ |ψ_i|² · dx — grid-spacing-weighted squared L2 norm.

        Equals 1.0 for a normalised wavefunction.  Used to verify unitarity:
        ``abs(sch.norm_squared() / norm0 - 1) < 1e-6`` after evolution.
        """
        ...


@final
class Heat2D:
    """2-D heat equation state with unit diffusion (a = 1).

    Solves du/dt = d^2u/dx^2 + d^2u/dy^2 on [xmin,xmax] x [ymin,ymax]
    via palindromic Strang splitting.  Output is flat x-fastest row-major.
    """

    def __init__(
        self,
        xmin: float, xmax: float, nx: int,
        ymin: float, ymax: float, ny: int,
        *,
        boundary: BoundaryLiteral = "reflect",
    ) -> None:
        """Create state on the 2D grid.  Each axis must have >= 4 nodes.

        Parameters
        ----------
        boundary : str, optional
            Boundary policy (keyword-only); applied to both axes.
            One of ``"reflect"`` (default), ``"periodic"``, ``"zero"``,
            ``"linear"``.

        Raises
        ------
        SemiflowError
            kind='GridMismatch' if any axis < 4 nodes or xmin >= xmax.
            kind='OutOfDomain' if boundary is not recognised.
        """
        ...

    def evolve(
        self,
        u0: NDArray[np.float64],
        tau: float,
        n_steps: int,
    ) -> NDArray[np.float64]:
        """Evolve u0 (flat, length nx*ny) by n_steps of size tau.

        Returns flat x-fastest row-major array, shape (nx*ny,).
        Raises SemiflowError on grid mismatch or invalid parameters.
        """
        ...


@final
class Heat3D:
    """3-D heat equation state with unit diffusion (a = 1).

    Solves du/dt = d^2u/dx^2 + d^2u/dy^2 + d^2u/dz^2 on a cuboid domain
    via palindromic Strang3D splitting.  Output is flat x-fastest row-major
    (I-T1-3D convention): values[k*nx*ny + j*nx + i] = u(x_i, y_j, z_k).
    """

    def __init__(
        self,
        xmin: float, xmax: float, nx: int,
        ymin: float, ymax: float, ny: int,
        zmin: float, zmax: float, nz: int,
        *,
        boundary: BoundaryLiteral = "reflect",
    ) -> None:
        """Create state on the 3D grid.  Each axis must have >= 4 nodes.

        Parameters
        ----------
        boundary : str, optional
            Boundary policy (keyword-only); applied to all three axes.
            One of ``"reflect"`` (default), ``"periodic"``, ``"zero"``,
            ``"linear"``.

        Raises
        ------
        SemiflowError
            kind='GridMismatch' if any axis < 4 nodes or xmin >= xmax.
            kind='OutOfDomain' if boundary is not recognised.
        """
        ...

    def evolve(
        self,
        u0: NDArray[np.float64],
        tau: float,
        n_steps: int,
    ) -> NDArray[np.float64]:
        """Evolve u0 (flat, length nx*ny*nz) by n_steps of size tau.

        Returns flat x-fastest row-major array, shape (nx*ny*nz,).
        Raises SemiflowError on grid mismatch or invalid parameters.
        """
        ...


class SemiflowError(Exception):
    """Single discriminated exception raised by all semiflow operations.

    Attribute kind identifies the error category; matches the C-ABI
    SemiflowStatus names from semiflow-ffi.
    """

    kind: str
    """Discriminator string; one of: GridMismatch, NanInf, OutOfDomain,
    BoundaryFailure, CflViolated, ConvergenceFailed, Unsupported, Panic."""


def version() -> str:
    """Return the semiflow-py crate version string (e.g. '0.10.0')."""
    ...


# ---------------------------------------------------------------------------
# Wave C — Graph PDE bindings (ADR-0059, v2.2)
# ---------------------------------------------------------------------------

from collections.abc import Callable

@final
class GraphPath:
    """Path graph P_n on ``n_nodes`` nodes with unit edge weights.

    Represents the graph ``0 — 1 — 2 — … — (n-1)``.  Used as the topology
    for :class:`GraphHeat` and :class:`MagnusGraphHeat`.
    """

    def __init__(self, n_nodes: int) -> None:
        """Create a path graph on ``n_nodes`` nodes.

        Parameters
        ----------
        n_nodes : int
            Number of nodes.  Must be >= 1.

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' if ``n_nodes == 0``.
        """
        ...

    def n_nodes(self) -> int:
        """Return the number of nodes in the graph."""
        ...

    def n_directed_edges(self) -> int:
        """Return the number of directed edge entries (= 2 × undirected edges)."""
        ...


@final
class GraphHeat:
    """Order-1 graph heat equation state: ``∂ₜu = −L_G u``.

    Uses the Chernoff function ``S(τ)f = f − τ L_G f`` driven by the
    combinatorial Laplacian of a :class:`GraphPath` or :class:`Graph`.

    The GIL is released during :meth:`evolve` (ADR-0031).
    """

    def __init__(
        self,
        graph: Union[GraphPath, "Graph", None] = None,
        laplacian: Union["Laplacian", None] = None,
        *,
        rho_bar: float,
        dtype: Literal["f64", "f32"] = "f64",
    ) -> None:
        """Create a graph heat state from a graph or Laplacian.

        Provide one of ``graph`` or ``laplacian``; ``laplacian`` takes
        precedence.

        Parameters
        ----------
        graph : GraphPath or Graph, optional
            Topology graph.  Combinatorial Laplacian assembled internally.
        laplacian : Laplacian, optional
            Pre-assembled Laplacian.  Takes precedence over ``graph``.
        rho_bar : float
            Gershgorin spectral-radius bound ``ρ̄ ≥ ρ(L_G)``.  Must be > 0.
        dtype : str, optional
            ``"f64"`` (default) or ``"f32"``.  When ``"f32"``, the Chernoff
            kernel runs in single precision; ``evolve()`` returns ``float32``.
            fp16 is REJECTED (dep-budget violation, ADR-0115).

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' if ``rho_bar <= 0``, neither argument given,
            or ``dtype`` is not ``"f64"``/``"f32"``.
        """
        ...

    def evolve(
        self,
        t_final: float,
        n_steps: int,
        f0: NDArray[np.float64],
    ) -> Union[NDArray[np.float64], NDArray[np.float32]]:
        """Evolve ``f0`` from ``t=0`` to ``t=t_final`` using ``n_steps`` Chernoff steps.

        The GIL is released during the inner pure-Rust compute loop (ADR-0031).

        Parameters
        ----------
        t_final : float
            Time horizon.  Must be finite and >= 0.
        n_steps : int
            Number of Chernoff steps.  Must be >= 1.
        f0 : NDArray[np.float64]
            Initial condition; 1-D float64 array of length ``graph.n_nodes()``.

        Returns
        -------
        NDArray[np.float64] or NDArray[np.float32]
            Result at ``t = t_final``; dtype matches the ``dtype`` set at
            construction.

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' if ``t_final < 0`` or ``n_steps == 0``.
            kind='GridMismatch' if ``len(f0) != graph.n_nodes()``.
        """
        ...


@final
class MagnusGraphHeat:
    """Magnus K=4 graph heat equation: ``∂ₜu = −L_G(t) u``.

    Uses ``MagnusGraphHeatChernoff`` (fourth-order Magnus expansion with
    two-point Gauss-Legendre quadrature) for time-varying graph Laplacians.
    For time-independent problems, :class:`GraphHeat` is faster.

    The ``lap_at_t`` callable is invoked via ``Python::attach`` inside the
    GIL-released window (ADR-0059 R2): at most 4 GIL re-acquires per step.

    **D1 (ADR-0113)**: constructor now mirrors :class:`MagnusGraphHeat6` —
    accepts ``graph``/``laplacian`` + keyword-only ``rho_bar_max``,
    ``convergence_check``.  ``lap_at_t`` may return Laplacians with varying
    edge weights (topology must remain fixed).
    """

    def __init__(
        self,
        graph: Union[GraphPath, "Graph", None] = None,
        laplacian: Union["Laplacian", None] = None,
        *,
        lap_at_t: "Callable[[float], Union[GraphPath, Graph, Laplacian]]",
        rho_bar_max: float,
        convergence_check: bool = True,
        dtype: Literal["f64", "f32"] = "f64",
    ) -> None:
        """Create a Magnus K=4 graph heat state.

        Parameters
        ----------
        graph : Graph or GraphPath, optional
            Fixed-topology graph.  Either ``graph`` or ``laplacian`` is required.
        laplacian : Laplacian, optional
            Pre-assembled Laplacian for the topology.
        lap_at_t : callable
            ``t: float -> Graph | Laplacian | GraphPath`` — return the Laplacian
            (or graph) at absolute time ``t``.  The topology MUST match ``graph``
            at every ``t``.  Edge weights may vary.
        rho_bar_max : float
            Upper bound on ``ρ̄(L_G(t))`` for all ``t``.  Must be > 0.
        convergence_check : bool, optional
            If ``True`` (default), each step checks ``rho_bar_max * tau < π/2``.
        dtype : str, optional
            ``"f64"`` (default) or ``"f32"``.  When ``"f32"``, evolve returns
            ``float32``; fp16 is REJECTED (ADR-0115).

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' if ``rho_bar_max <= 0``, no graph provided,
            or ``dtype`` is not ``"f64"``/``"f32"``.
            kind='ConvergenceFailed' if convergence-radius condition violated.
        """
        ...

    def evolve(
        self,
        t_final: float,
        n_steps: int,
        f0: NDArray[np.float64],
    ) -> Union[NDArray[np.float64], NDArray[np.float32]]:
        """Evolve ``f0`` from ``t=0`` to ``t=t_final`` using ``n_steps`` Magnus steps.

        The GIL is released during the Rust compute loop; ``lap_at_t`` is
        called via ``Python::attach`` inside the released window (ADR-0031 /
        ADR-0059 R2 pattern).

        Parameters
        ----------
        t_final : float
            Time horizon.  Must be finite and >= 0.
        n_steps : int
            Number of Magnus steps.  Must be >= 1.
        f0 : NDArray[np.float64]
            Initial condition; 1-D float64 array of length ``graph.n_nodes()``.

        Returns
        -------
        NDArray[np.float64] or NDArray[np.float32]
            Result at ``t = t_final``; dtype matches the ``dtype`` set at construction.

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' / 'ConvergenceFailed' on invalid parameters.
        """
        ...

# ---------------------------------------------------------------------------
# Phase 4 — v2.3 ADR-0058 / ADR-0055 / ADR-0044
# ---------------------------------------------------------------------------

@final
class NonSeparable2D:
    """Non-separable anisotropic 2-D diffusion with mixed-derivative coupling.

    Solves ``∂_t u = ∂_{xx}u + ∂_{yy}u + β(x,y)·∂_x∂_y u`` on
    ``[xmin, xmax] × [ymin, ymax]`` via the palindromic 5-leg Chernoff
    operator (math.md §10.7-ter, ADR-0058).

    Parameters
    ----------
    xmin, xmax : float
        X-axis domain; must satisfy ``xmin < xmax``.
    nx : int
        Number of X-axis grid nodes (>= 4).
    ymin, ymax : float
        Y-axis domain; must satisfy ``ymin < ymax``.
    ny : int
        Number of Y-axis grid nodes (>= 4).
    u0 : NDArray[np.float64]
        Flat row-major initial condition of length ``nx * ny``.
    c : float, optional
        Constant scalar coupling coefficient (default 0.0 — pure Strang2D).
    boundary : BoundaryLiteral, optional
        Boundary policy; default ``"reflect"``.

    Raises
    ------
    SemiflowError
        ``kind='GridMismatch'`` for size mismatches.
        ``kind='NanInf'`` if ``u0`` contains NaN or Inf.
        ``kind='CflViolated'`` if the coupling violates CFL.
    """

    def __new__(
        cls,
        xmin: float,
        xmax: float,
        nx: int,
        ymin: float,
        ymax: float,
        ny: int,
        u0: NDArray[np.float64],
        *,
        c: float = 0.0,
        boundary: BoundaryLiteral = "reflect",
    ) -> "NonSeparable2D": ...

    @staticmethod
    def with_beta_array(
        xmin: float,
        xmax: float,
        nx: int,
        ymin: float,
        ymax: float,
        ny: int,
        beta_values: NDArray[np.float64],
        u0: NDArray[np.float64],
        *,
        beta_norm_bound: float | None = None,
        boundary: BoundaryLiteral = "reflect",
    ) -> "NonSeparable2D":
        """Construct ``NonSeparable2D`` from a pre-sampled ``(nx, ny)`` beta array.

        Parameters
        ----------
        beta_values : NDArray[np.float64]
            Shape ``(nx, ny)`` row-major array of ``β(x_i, y_j)`` values.
        beta_norm_bound : float or None
            Upper bound on ``‖β‖_∞``. Auto-computed as ``1.1 * max(|β|)`` if None.
        """
        ...

    def evolve(
        self,
        t: float,
        n_steps: int = 100,
    ) -> NDArray[np.float64]:
        """Evolve the current state by time ``t`` using ``n_steps`` Chernoff steps.

        The GIL is released during the inner Rust compute loop (ADR-0031).

        Parameters
        ----------
        t : float
            Time step.  Must be finite and > 0.
        n_steps : int, optional
            Number of Chernoff substeps (default 100).

        Returns
        -------
        NDArray[np.float64]
            Flat row-major state at ``t``; length ``nx * ny``.

        Raises
        ------
        SemiflowError
            ``kind='OutOfDomain'`` if ``t <= 0`` or ``n_steps < 1``.
        """
        ...

    def __len__(self) -> int:
        """Return ``nx * ny`` (total number of state values)."""
        ...


@final
class Adjoint:
    """Adjoint semigroup wrapper for any supported 1-D Chernoff kernel.

    Evolves the adjoint (dual) semigroup ``exp(τA*)`` for the chosen inner
    kernel.  For self-adjoint kernels (``"heat2"``, ``"heat4"``, ``"heat6"``)
    the wrapper is zero-overhead.  For ``"drift"`` and ``"shift"`` the
    bounded-perturbation expansion is used (math.md §15.1, ADR-0055).

    Parameters
    ----------
    xmin, xmax : float
        Domain boundaries; ``xmin < xmax``.
    n : int
        Number of grid nodes (>= 4).
    u0 : NDArray[np.float64]
        Initial condition of length ``n``.
    kernel : str, optional
        Inner kernel.  One of ``"heat2"`` (default), ``"heat4"``,
        ``"heat6"``, ``"drift"``, ``"shift"``.
    self_adjoint : bool, optional
        If ``True``, skip the dual correction even for non-symmetric kernels
        (caller asserts self-adjointness).  Default ``False``.
    boundary : BoundaryLiteral, optional
        Boundary policy; default ``"reflect"``.

    Raises
    ------
    SemiflowError
        ``kind='GridMismatch'`` for invalid grid or IC-length mismatches.
        ``kind='NanInf'`` if ``u0`` contains NaN or Inf.
    """

    def __new__(
        cls,
        xmin: float,
        xmax: float,
        n: int,
        u0: NDArray[np.float64],
        *,
        kernel: str = "heat2",
        self_adjoint: bool = False,
        boundary: BoundaryLiteral = "reflect",
    ) -> "Adjoint": ...

    def evolve(
        self,
        t: float,
        n_steps: int = 100,
    ) -> NDArray[np.float64]:
        """Evolve the current state by time ``t`` using ``n_steps`` Chernoff steps.

        The GIL is released during the inner Rust compute loop (ADR-0031).

        Parameters
        ----------
        t : float
            Time step.  Must be finite and > 0.
        n_steps : int, optional
            Number of Chernoff substeps (default 100).

        Returns
        -------
        NDArray[np.float64]
            State array at ``t``; length ``n``.

        Raises
        ------
        SemiflowError
            ``kind='OutOfDomain'`` if ``t <= 0`` or ``n_steps < 1``.
        """
        ...

    def values(self) -> NDArray[np.float64]:
        """Return the current grid values as a 1-D float64 numpy array (copy)."""
        ...

    def order(self) -> int:
        """Return the approximation order of the wrapped adjoint kernel."""
        ...

    def is_self_adjoint(self) -> bool:
        """Return ``True`` if the inner kernel is declared self-adjoint."""
        ...

    def __len__(self) -> int:
        """Return the number of grid nodes ``n``."""
        ...

class GraphAdjoint:
    """Graph state-adjoint for the truncated Magnus K=4 map (Issue #2, ADR-0115).

    Computes the backward costate sweep ``λ_0 = S⋆_1 ⋯ S⋆_n · λ_n`` where
    each step applies ``S⋆(τ) = Σ_{m=0..4} (Ω₄ᵀ)^m/m!`` — the transpose of
    the degree-4 Taylor map (math.md §42 Theorem 42.1, sign-flipped commutator).

    Parameters
    ----------
    graph : Graph or GraphPath, optional
        Fixed-topology graph.
    laplacian : Laplacian, optional
        Pre-assembled Laplacian for the topology.
    lap_at_t : callable
        ``t: float -> Graph | Laplacian | GraphPath``
    rho_bar : float
        Upper bound on ``ρ̄(L_G(t))``.
    a : callable, optional
        ``t: float -> list[float]`` — node weights.
        Required when ``kernel="varcoef_magnus_graph"``.
    kernel : str, optional
        ``"magnus_graph"`` (default) or ``"varcoef_magnus_graph"``.
    convergence_check : bool, optional
        Enable convergence-radius guard (default ``True``).
    """

    def __new__(
        cls,
        graph: "Graph | GraphPath | None" = None,
        laplacian: "Laplacian | None" = None,
        *,
        lap_at_t: "Callable[[float], Any]",
        rho_bar: float,
        a: "Callable[[float], list[float]] | None" = None,
        kernel: str = "magnus_graph",
        convergence_check: bool = True,
    ) -> "GraphAdjoint": ...

    def evolve_state_adjoint(
        self,
        lambda_n: NDArray[np.float64],
        t: float,
        n_steps: int = 100,
    ) -> NDArray[np.float64]:
        """Backward costate sweep: ``n_steps`` adjoint steps of size ``t/n_steps``.

        Terminal costate ``lambda_n`` → initial costate ``λ_0``.
        GIL released during Rust compute loop (ADR-0031).

        Parameters
        ----------
        lambda_n : NDArray[np.float64]
            Terminal costate; length ``n_nodes``.
        t : float
            Total time (positive, finite).
        n_steps : int, optional
            Number of backward steps (default 100).

        Returns
        -------
        NDArray[np.float64]
            λ₀; length ``n_nodes``.
        """
        ...

    def n_nodes(self) -> int:
        """Return the number of graph nodes."""
        ...

# ---------------------------------------------------------------------------
# Phase 5 — Graph topology & advanced graph kernels
# ---------------------------------------------------------------------------

@final
class Graph:
    """Undirected weighted graph used as the spatial domain for graph PDEs.

    Parameters
    ----------
    None — use the factory class methods to construct graphs.

    Examples
    --------
    >>> g = Graph.path(64)
    >>> g.n_nodes
    64
    """

    @staticmethod
    def path(n: int) -> "Graph":
        """Create a path graph on ``n`` nodes (0-1-2-..-(n-1)).

        Parameters
        ----------
        n : int
            Number of nodes; must be >= 1.

        Returns
        -------
        Graph
        """
        ...

    @staticmethod
    def cycle(n: int) -> "Graph":
        """Create a cycle graph on ``n`` nodes.

        Parameters
        ----------
        n : int
            Number of nodes; must be >= 3.

        Returns
        -------
        Graph
        """
        ...

    @staticmethod
    def from_edges(n: int, edges: Union[list[tuple[int, int, float]], NDArray[np.float64]]) -> "Graph":
        """Create a graph from an explicit edge list.

        Parameters
        ----------
        n : int
            Number of nodes.
        edges : list of (src, dst, weight) tuples OR flat float64 numpy array
            Undirected edges; each edge appears once.  Weights must be > 0.
            Flat array form: ``[src0, dst0, w0, src1, dst1, w1, ...]`` (length
            must be divisible by 3).

        Returns
        -------
        Graph
        """
        ...

    @staticmethod
    def erdos_renyi(n: int, p: float, seed: int) -> "Graph":
        """Create a random Erdős–Rényi G(n, p) graph.

        Parameters
        ----------
        n : int
            Number of nodes.
        p : float
            Edge probability in [0, 1].
        seed : int
            RNG seed for reproducibility.

        Returns
        -------
        Graph
        """
        ...

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the graph."""
        ...

    @property
    def n_directed_edges(self) -> int:
        """Number of directed half-edges (2 × undirected edges)."""
        ...

    def degree(self, node: int) -> float:
        """Return the (weighted) degree of ``node``.

        Parameters
        ----------
        node : int
            Node index; must be in ``[0, n_nodes)``.

        Returns
        -------
        float
        """
        ...


@final
class Laplacian:
    """Assembled graph Laplacian matrix (combinatorial or normalized).

    Parameters
    ----------
    None — use the factory class methods to construct.

    Notes
    -----
    The Laplacian is stored in CSR format and pre-computes a spectral bound
    used by the Chernoff kernels.
    """

    @staticmethod
    def combinatorial(graph: "Graph") -> "Laplacian":
        """Assemble the combinatorial Laplacian ``L = D - A``.

        Parameters
        ----------
        graph : Graph

        Returns
        -------
        Laplacian
        """
        ...

    @staticmethod
    def normalized(graph: "Graph") -> "Laplacian":
        """Assemble the normalized Laplacian ``L = I - D^{-1/2} A D^{-1/2}``.

        Parameters
        ----------
        graph : Graph

        Returns
        -------
        Laplacian
        """
        ...

    @property
    def n_nodes(self) -> int:
        """Number of nodes (rows/columns of the matrix)."""
        ...

    @property
    def is_combinatorial(self) -> bool:
        """True if this is a combinatorial Laplacian."""
        ...

    @property
    def is_normalized(self) -> bool:
        """True if this is a normalized Laplacian."""
        ...

    @property
    def spectral_bound(self) -> float:
        """Pre-computed spectral radius upper bound ``ρ̄(L)``."""
        ...

    def to_dense(self) -> "np.ndarray[Any, np.dtype[np.float64]]":
        """Dense ``n × n`` float64 matrix reconstructed from CSR (row-major copy).

        Memory: O(n²).  Raises ``SemiflowError(OutOfDomain)`` if ``n*n``
        overflows.
        """
        ...

    def row_ptr(self) -> "np.ndarray[Any, np.dtype[np.int64]]":
        """CSR row-pointer array (copy), length ``n_nodes + 1``, dtype int64."""
        ...

    def col_idx(self) -> "np.ndarray[Any, np.dtype[np.int64]]":
        """CSR column-index array (copy), length ``n_directed_edges``, dtype int64."""
        ...

    def vals(self) -> "np.ndarray[Any, np.dtype[np.float64]]":
        """CSR values array (copy), length ``n_directed_edges``, dtype float64."""
        ...


@final
class GraphHeat4th:
    """Fourth-order graph heat equation: ``∂ₜu = −L_G u``.

    Uses the fourth-order ``GraphHeat4thChernoff`` kernel.
    For time-varying Laplacians, use :class:`MagnusGraphHeat6`.

    Parameters
    ----------
    laplacian : Laplacian, optional
        Pre-assembled Laplacian.  Either ``laplacian`` or ``graph`` required.
    graph : Graph, optional
        Graph from which the combinatorial Laplacian is assembled automatically.
    rho_bar : float
        Spectral radius upper bound ``ρ̄(L_G)``.  Must be > 0.

    Raises
    ------
    SemiflowError
        ``kind='OutOfDomain'`` if ``rho_bar <= 0`` or no input provided.
    """

    def __new__(
        cls,
        *,
        laplacian: Union["Laplacian", None] = None,
        graph: Union["Graph", None] = None,
        rho_bar: float,
    ) -> "GraphHeat4th": ...

    def evolve(
        self,
        t_final: float,
        n_steps: int,
        f0: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evolve ``f0`` to ``t = t_final`` using ``n_steps`` fourth-order steps.

        Parameters
        ----------
        t_final : float
            Time horizon.  Must be finite and >= 0.
        n_steps : int
            Number of steps.  Must be >= 1.
        f0 : NDArray[np.float64]
            Initial condition; length ``n_nodes``.

        Returns
        -------
        NDArray[np.float64]
            Result at ``t = t_final``; length ``n_nodes``.

        Raises
        ------
        SemiflowError
            ``kind='OutOfDomain'`` on invalid parameters.
        """
        ...


@final
class VarCoefGraphHeat:
    """Variable-coefficient graph heat equation: ``∂ₜu = −L_{a,G} u``.

    Uses ``VarCoefGraphHeatChernoff``.  The edge weight is modulated by a
    node-local coefficient vector ``a``.

    Parameters
    ----------
    graph : Graph
        Graph topology.
    a : NDArray[np.float64]
        Coefficient vector; length ``n_nodes``, all entries > 0.
    rho_bar : float
        Spectral radius upper bound.  Must be > 0.

    Raises
    ------
    SemiflowError
        ``kind='OutOfDomain'`` if ``rho_bar <= 0`` or ``a`` has invalid values.
        ``kind='GridMismatch'`` if ``len(a) != n_nodes``.
    """

    def __new__(
        cls,
        graph: "Graph",
        a: NDArray[np.float64],
        *,
        rho_bar: float,
        dtype: Literal["f64", "f32"] = "f64",
    ) -> "VarCoefGraphHeat": ...

    def evolve(
        self,
        t_final: float,
        n_steps: int,
        f0: NDArray[np.float64],
    ) -> Union[NDArray[np.float64], NDArray[np.float32]]:
        """Evolve ``f0`` to ``t = t_final`` using ``n_steps`` steps.

        Parameters
        ----------
        t_final : float
            Time horizon.  Must be finite and >= 0.
        n_steps : int
            Number of steps.  Must be >= 1.
        f0 : NDArray[np.float64]
            Initial condition; length ``n_nodes``.

        Returns
        -------
        NDArray[np.float64] or NDArray[np.float32]
            Result at ``t = t_final``; dtype matches construction ``dtype``.

        Raises
        ------
        SemiflowError
            ``kind='OutOfDomain'`` on invalid parameters.
        """
        ...


@final
class MagnusGraphHeat6:
    """Magnus K=6 graph heat equation: ``∂ₜu = −L_G(t) u``.

    Uses ``MagnusGraphHeat6thChernoff`` (sixth-order GL₆ three-point expansion)
    for time-varying graph Laplacians.

    Parameters
    ----------
    graph : Graph, optional
        Fixed-topology graph.  Either ``graph`` or ``laplacian`` is required.
    laplacian : Laplacian, optional
        Pre-assembled Laplacian for the topology.
    lap_at_t : Callable[[float], Union[GraphPath, Graph, Laplacian]]
        Returns the Laplacian (or graph) at absolute time ``t``.
        The topology MUST match ``graph`` at every ``t``.
    rho_bar_max : float
        Upper bound on ``ρ̄(L_G(t))`` for all ``t``.  Must be > 0.
    convergence_check : bool, optional
        If ``True`` (default), each step checks ``rho_bar_max * tau < π/2``.

    Raises
    ------
    SemiflowError
        ``kind='OutOfDomain'`` if ``rho_bar_max <= 0`` or no graph provided.
        ``kind='ConvergenceFailed'`` if convergence-radius condition violated.
    """

    def __new__(
        cls,
        *,
        graph: Union["Graph", None] = None,
        laplacian: Union["Laplacian", None] = None,
        lap_at_t: "Callable[[float], Union[GraphPath, Graph, Laplacian]]",
        rho_bar_max: float,
        convergence_check: bool = True,
    ) -> "MagnusGraphHeat6": ...

    def evolve(
        self,
        t_final: float,
        n_steps: int,
        f0: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evolve ``f0`` from ``t=0`` to ``t=t_final`` using ``n_steps`` Magnus K=6 steps.

        Parameters
        ----------
        t_final : float
            Time horizon.  Must be finite and >= 0.
        n_steps : int
            Number of Magnus K=6 steps.  Must be >= 1.
        f0 : NDArray[np.float64]
            Initial condition; length ``n_nodes``.

        Returns
        -------
        NDArray[np.float64]
            Result at ``t = t_final``; length ``n_nodes``.

        Raises
        ------
        SemiflowError
            ``kind='OutOfDomain'`` / ``kind='ConvergenceFailed'`` on invalid parameters.
        """
        ...

@final
class GraphHeat6:
    """Sixth-order static graph heat equation: ``∂ₜu = −L_G u`` (ADR-0062).

    Uses the ``GraphHeat6thChernoff`` kernel.
    For time-varying Laplacians use :class:`MagnusGraphHeat6`.

    Parameters
    ----------
    laplacian : Laplacian, optional
        Pre-assembled Laplacian.  Either ``laplacian`` or ``graph`` is required.
    graph : Graph, optional
        Graph from which the combinatorial Laplacian is assembled automatically.
    rho_bar : float
        Spectral radius upper bound ``ρ̄(L_G)``.  Must be > 0.

    Raises
    ------
    SemiflowError
        ``kind='OutOfDomain'`` if ``rho_bar <= 0`` or no input provided.
    """

    def __new__(
        cls,
        *,
        laplacian: Union["Laplacian", None] = None,
        graph: Union["Graph", None] = None,
        rho_bar: float,
    ) -> "GraphHeat6": ...

    def evolve(
        self,
        t_final: float,
        n_steps: int,
        f0: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evolve ``f0`` to ``t = t_final`` using ``n_steps`` sixth-order steps.

        Parameters
        ----------
        t_final : float
            Time horizon.  Must be finite and >= 0.
        n_steps : int
            Number of steps.  Must be >= 1.
        f0 : NDArray[np.float64]
            Initial condition; length ``n_nodes``.

        Returns
        -------
        NDArray[np.float64]
            Result at ``t = t_final``; length ``n_nodes``.

        Raises
        ------
        SemiflowError
            ``kind='OutOfDomain'`` on invalid parameters.
        """
        ...

    @property
    def n_nodes(self) -> int:
        """Number of nodes the kernel acts on."""
        ...


@final
class VarCoefMagnusGraph:
    """Variable-coefficient × time-dependent graph Magnus K=4 (ADR-0063).

    ``∂_t u = −L_a(t) u``, where ``L_a(t) = sqrt(a(t)) ⊙ L_G(t) ⊙ sqrt(a(t))``.
    Order-4 Magnus expansion with GL₂ quadrature.

    Parameters
    ----------
    n_nodes : int
        Number of graph nodes (must be >= 1).
    lap_at_t : callable
        ``t: float -> Graph | Laplacian | GraphPath``.  Topology must be
        invariant in ``t``.
    a_at_t : callable
        ``t: float -> NDArray[float64]``.  Node-local coefficient; length
        ``n_nodes`` at every sampled ``t``; all entries >= 0.
    rho_bar_max : float
        Upper bound on ``ρ̄(L_G(t))`` for all ``t``.  Must be > 0.
    a_sup_max : float
        Upper bound on ``‖a(t)‖_∞`` for all ``t``.  Must be > 0.
    convergence_check : bool, optional
        If ``True`` (default), each step validates the convergence-radius
        inequality ``rho_bar_max * a_sup_max² * τ < π/2``.

    Raises
    ------
    SemiflowError
        ``kind='OutOfDomain'`` if ``rho_bar_max <= 0`` or ``a_sup_max <= 0``.
        ``kind='ConvergenceFailed'`` if the convergence-radius condition fires.
    """

    def __new__(
        cls,
        n_nodes: int,
        *,
        lap_at_t: "Callable[[float], Union[GraphPath, Graph, Laplacian]]",
        a_at_t: "Callable[[float], NDArray[np.float64]]",
        rho_bar_max: float,
        a_sup_max: float,
        convergence_check: bool = True,
    ) -> "VarCoefMagnusGraph": ...

    def evolve(
        self,
        t_final: float,
        n_steps: int,
        f0: NDArray[np.float64],
        *,
        t_start: float = 0.0,
    ) -> NDArray[np.float64]:
        """Evolve ``f0`` from ``t_start`` to ``t_start + t_final`` using Magnus K=4.

        Parameters
        ----------
        t_final : float
            Duration; must be finite and >= 0.
        n_steps : int
            Number of Magnus K=4 steps; must be >= 1.
        f0 : NDArray[np.float64]
            Initial condition; length ``n_nodes``.
        t_start : float, optional
            Absolute start time (default 0.0); used for stitched trajectories.

        Returns
        -------
        NDArray[np.float64]
            Result at ``t = t_start + t_final``; length ``n_nodes``.

        Raises
        ------
        SemiflowError
            ``kind='OutOfDomain'`` / ``kind='ConvergenceFailed'`` on invalid params.
        """
        ...

    @property
    def n_nodes(self) -> int:
        """Number of graph nodes."""
        ...

    @staticmethod
    def compute_rho_bar(
        lap_at_t: "Callable[[float], Union[GraphPath, Graph, Laplacian]]",
        a_at_t: "Callable[[float], NDArray[np.float64]]",
        t0: float,
        t1: float,
        n_nodes: int,
        *,
        n_samples: int = 32,
    ) -> tuple[float, float]:
        """Estimate ``(rho_bar_max, a_sup_max)`` over ``[t0, t1]``.

        Returns
        -------
        tuple[float, float]
            ``(rho_bar_max, a_sup_max)`` suitable for direct use as constructor kwargs.
        """
        ...


@final
class AdaptivePI:
    """PI-controller adaptive-step integrator for any supported 1-D kernel.

    Unlike fixed-step Chernoff products, ``AdaptivePI`` selects substep sizes
    automatically to meet the mixed tolerance
    ``tol_abs + tol_rel * ‖u‖`` (ADR-0044, math.md §11.1.bis).

    Parameters
    ----------
    xmin, xmax : float
        Domain boundaries; ``xmin < xmax``.
    n : int
        Number of grid nodes (>= 4).
    u0 : NDArray[np.float64]
        Initial condition of length ``n``.
    kernel : str, optional
        Inner kernel; one of ``"heat2"`` (default), ``"heat4"``,
        ``"heat6"``, ``"drift"``, ``"shift"``.
    tol_abs : float, optional
        Absolute tolerance component (default 1e-6).
    tol_rel : float, optional
        Relative tolerance component (default 1e-4).
    boundary : BoundaryLiteral, optional
        Boundary policy; default ``"reflect"``.

    Raises
    ------
    SemiflowError
        ``kind='GridMismatch'`` for invalid grid or IC-length mismatches.
        ``kind='NanInf'`` if ``u0`` contains NaN or Inf.
        ``kind='OutOfDomain'`` if ``t <= 0`` or non-finite.
        ``kind='CflViolated'`` if adaptive integration exceeds ``max_substeps``.
    """

    def __new__(
        cls,
        xmin: float,
        xmax: float,
        n: int,
        u0: NDArray[np.float64],
        *,
        kernel: str = "heat2",
        tol_abs: float = 1e-6,
        tol_rel: float = 1e-4,
        boundary: BoundaryLiteral = "reflect",
    ) -> "AdaptivePI": ...

    def evolve(
        self,
        t: float,
    ) -> NDArray[np.float64]:
        """Evolve the current state by time ``t`` using adaptive PI substeps.

        The integrator selects substep sizes automatically based on the
        tolerance parameters set at construction.  The GIL is released during
        the adaptive loop (ADR-0031).

        Parameters
        ----------
        t : float
            Time horizon.  Must be finite and > 0.

        Returns
        -------
        NDArray[np.float64]
            State array at ``t``; length ``n``.

        Raises
        ------
        SemiflowError
            ``kind='OutOfDomain'`` if ``t <= 0`` or non-finite.
            ``kind='CflViolated'`` if ``max_substeps`` is exceeded.
        """
        ...

    def __len__(self) -> int:
        """Return the number of grid nodes ``n``."""
        ...

# ---------------------------------------------------------------------------
# Wave E — v3.0 Evolver surface (ADR-0076)
# ---------------------------------------------------------------------------

@final
class GrowthV3:
    """Growth bound ``‖S(τ)‖ ≤ M · exp(ω · τ)`` (v3.0 pyclass, ADR-0074 / ADR-0076).

    Attributes
    ----------
    multiplier : float
        ``M ≥ 1.0``.  For unit-diffusion: ``1.0``.
    omega : float
        ``ω`` (finite).  For unit-diffusion: ``0.0``.

    Notes
    -----
    Returned by :meth:`EvolverHeat1DUnitV3.growth`.  Named-field access only;
    tuple unpacking (``m, w = g``) is NOT supported (the Rust side is a
    ``#[pyclass]``, not a ``PyTuple``).
    """

    multiplier: float
    omega: float


@final
class EvolverHeat1DUnitV3:
    """v3.0 Evolver for the unit-diffusion heat equation (ADR-0076, Wave E).

    Solves ``∂_t u = ∂²u`` on ``[domain_lo, domain_hi]`` with ``n_grid`` nodes
    and ``n_chernoff`` Chernoff iterations per :meth:`evolve_into` call.

    This is the **v3-native** pyclass wrapping ``Evolver<DiffusionChernoff<f64>>``
    directly (zero-alloc ``apply_into`` hot path).  For the v2 allocating API,
    use :class:`Heat1D` which is preserved for 12-month compat per ADR-0035 §9.

    Parameters
    ----------
    domain_lo : float
        Left boundary; must be finite.
    domain_hi : float
        Right boundary; must be finite and ``> domain_lo``.
    n_grid : int
        Number of grid nodes; must be ``≥ 4``.
    u0 : NDArray[np.float64]
        Initial state; 1-D float64 array of length ``n_grid``.
    n_chernoff : int
        Chernoff iteration count; must be ``≥ 1``.

    Raises
    ------
    SemiflowError
        kind='GridMismatch' — invalid geometry or ``len(u0) != n_grid``.
        kind='NanInf' — ``u0`` contains NaN or Inf.
        kind='OutOfDomain' — ``n_chernoff == 0``.
    """

    def __init__(
        self,
        domain_lo: float,
        domain_hi: float,
        n_grid: int,
        u0: NDArray[np.float64],
        n_chernoff: int,
    ) -> None: ...

    def evolve_into(self, t: float, buf: NDArray[np.float64]) -> None:
        """Evolve the current state in-place by time ``t``.

        Writes evolved values into ``buf`` (zero-alloc).  The internal state
        is updated to the result.  The GIL is released during the pure-Rust
        compute loop (ADR-0031 three-phase pattern).

        Parameters
        ----------
        t : float
            Time to advance.  Must be non-negative and finite.
        buf : NDArray[np.float64]
            Output buffer of length ``size()``.  Modified in-place.

        Raises
        ------
        SemiflowError
            kind='GridMismatch' — ``len(buf) != n_grid``.
            kind='OutOfDomain' — ``t < 0`` or non-finite.
        """
        ...

    def growth(self) -> "GrowthV3":
        """Return the growth bound of the underlying Chernoff function.

        Returns
        -------
        GrowthV3
            Named-field growth bound ``(multiplier, omega)``.
            For unit-diffusion: ``multiplier=1.0, omega=0.0``.
        """
        ...

    def size(self) -> int:
        """Return the number of grid nodes ``n_grid``."""
        ...

    def n_chernoff(self) -> int:
        """Return the Chernoff iteration count set at construction."""
        ...

    def values(self) -> NDArray[np.float64]:
        """Return the current grid values as a 1-D float64 numpy array (copy).

        Mutations to the returned array do not affect the internal state.
        """
        ...

    def __len__(self) -> int:
        """Return the number of grid nodes ``n_grid``."""
        ...


# ---------------------------------------------------------------------------
# v8.0.0 F1 — Dual-AD Greeks bindings (ADR-0133, ADR-0028 Amendment 2)
# ---------------------------------------------------------------------------

@final
class EvolverHeat1DGreeksV3:
    """Hyper-dual Greeks evolver for unit-diffusion heat (v8.0.0, ADR-0133 A3).

    Computes ``(value, delta, gamma)`` — the solution and its first/second
    derivatives w.r.t. the diffusion-scale parameter θ — via a single
    ``Dual<Dual<f64>>`` hyper-dual sweep (math §46.4).

    Parameters
    ----------
    domain_lo : float
        Left boundary; must be finite.
    domain_hi : float
        Right boundary; must be finite and ``> domain_lo``.
    n_grid : int
        Number of grid nodes; must be ``≥ 4``.
    u0 : array-like
        Initial state; 1-D float64, length ``n_grid``.
    n_chernoff : int
        Chernoff iteration count; must be ``≥ 1``.
    scale_theta : float, optional
        Diffusion-scale θ at which Δ/Γ are evaluated (default 0.5).

    Raises
    ------
    SemiflowError
        kind='GridMismatch' — invalid geometry or ``len(u0) != n_grid``.
        kind='NanInf' — ``u0`` contains NaN or Inf.
        kind='OutOfDomain' — ``n_chernoff == 0``.
    """

    def __init__(
        self,
        domain_lo: float,
        domain_hi: float,
        n_grid: int,
        u0: NDArray[np.float64],
        n_chernoff: int,
        scale_theta: float = 0.5,
    ) -> None: ...

    def greeks(
        self,
        t: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Advance by ``t`` and return ``(value, delta, gamma)`` as three arrays.

        Each array has length ``size()`` and dtype ``float64``.  The internal
        current state is updated to the primal-value result.  The GIL is
        released during the hyper-dual Chernoff sweep (ADR-0031).

        Parameters
        ----------
        t : float
            Time step; must be ``≥ 0`` and finite.

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
            ``(value, delta, gamma)`` — each float64 array of length ``n_grid``.
            - ``value[i]``  — primal solution u(t, x_i)
            - ``delta[i]``  — ∂u/∂θ (forward-mode Δ)
            - ``gamma[i]``  — ∂²u/∂θ² (hyper-dual Γ)

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' — ``t < 0`` or non-finite.
        """
        ...

    def size(self) -> int:
        """Return the number of grid nodes ``n_grid``."""
        ...

    def n_chernoff(self) -> int:
        """Return the Chernoff iteration count set at construction."""
        ...


@final
class KilledDirichlet1D:
    """PyO3 wrapper for ``KilledDirichletChernoff`` (TIER 2, §7 design doc).

    Crank–Nicolson Cayley map of the killed Dirichlet generator on
    ``[domain_lo, domain_hi]`` with absorbing endpoints (``u|∂R = 0``).
    Order-2 (math §44.ter, ADR-0135 Amendment 2).

    Parameters
    ----------
    domain_lo : float
        Left boundary.
    domain_hi : float
        Right boundary.
    n_grid : int
        Grid nodes; must be ``≥ 3``.
    u0 : array-like
        Initial condition; length ``n_grid``.
    n_chernoff : int
        Chernoff iteration count; must be ``≥ 1``.

    Raises
    ------
    SemiflowError
        kind='GridMismatch' — n_grid < 3 or len(u0) != n_grid.
        kind='OutOfDomain' — n_chernoff == 0.
        kind='NanInf' — non-finite value in u0.
    """

    def __init__(
        self,
        domain_lo: float,
        domain_hi: float,
        n_grid: int,
        u0: NDArray[np.float64],
        n_chernoff: int,
    ) -> None: ...

    def apply(self, t: float) -> NDArray[np.float64]:
        """Advance by ``t``; return evolved grid as float64 numpy array.

        Parameters
        ----------
        t : float
            Time step; must be ``≥ 0`` and finite.

        Returns
        -------
        NDArray[np.float64]
            Evolved state; length ``n_grid``.

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' — ``t < 0`` or non-finite.
        """
        ...

    def size(self) -> int:
        """Return the number of grid nodes."""
        ...


# ---------------------------------------------------------------------------
# v8.1.0 F2 — ResolventJumpV8 (TWS contour, TIER 1, ADR-0138, ADR-0134)
# ---------------------------------------------------------------------------

@final
class ResolventJumpV8:
    """Resolvent time-jump evaluator for 1D unit-diffusion heat (v8.1.0, ADR-0138).

    Evaluates ``e^{tA}g`` for a LARGE step ``t`` via the TWS parabolic-contour
    inverse Laplace quadrature (math.md §47, ADR-0134).  Suitable for large
    ``t`` where a many-step Chernoff product would be expensive.

    **NARROW scope**: self-adjoint / sectorial generators only (diffusion family).
    Non-self-adjoint / advection-dominated generators are OUT of scope
    (math.md §47.4 NORMATIVE).  ``m_nodes >= 6`` required.

    Parameters
    ----------
    domain_lo : float
        Left boundary (finite).
    domain_hi : float
        Right boundary (finite, ``> domain_lo``).
    n_grid : int
        Number of grid nodes (``>= 4``).
    m_nodes : int
        Number of TWS contour nodes (``>= 6``; M=16 recommended for ``|t| <= 1``).

    Raises
    ------
    SemiflowError
        kind='GridMismatch' — invalid grid geometry.
        kind='OutOfDomain'  — m_nodes < 6.
    """

    def __init__(
        self,
        domain_lo: float,
        domain_hi: float,
        n_grid: int,
        m_nodes: int,
    ) -> None: ...

    def jump(
        self,
        t: float,
        g: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate ``e^{tA}g`` and return the result as a numpy float64 array.

        The GIL is released during the M-node TWS contour solve (ADR-0031).
        The complex contour arithmetic stays sealed in core (ADR-0138).

        Parameters
        ----------
        t : float
            Time step (``> 0``, finite).
        g : NDArray[np.float64]
            Initial condition; 1-D float64, length ``n_grid``.

        Returns
        -------
        NDArray[np.float64]
            Result ``e^{tA}g``, float64, length ``n_grid``.

        Raises
        ------
        SemiflowError
            kind='GridMismatch' — ``len(g) != n_grid``.
            kind='OutOfDomain'  — ``t <= 0`` or non-finite.
        """
        ...

    def size(self) -> int:
        """Return the number of grid nodes ``n_grid``."""
        ...

    def m_nodes(self) -> int:
        """Return the number of TWS contour nodes set at construction."""
        ...


# ---------------------------------------------------------------------------
# v8.1.0 C2 — AdjointFokkerPlanckV8 (Lemma A.1 push, TIER 1, ADR-0138)
# ---------------------------------------------------------------------------

@final
class AdjointFokkerPlanckV8:
    """Adjoint Fokker-Planck Chernoff on M(ℝ) — flat two-buffer interface (v8.1.0).

    Applies adjoint (weak-*) Fokker-Planck Chernoff steps: each Dirac δ_x
    is pushed to four children (Lemma A.1, §38.3):

      S*(τ) δ_x = ¼δ_{x+h} + ¼δ_{x-h} + ½δ_{x+k} + τc·δ_x

    where ``h = 2√(aτ)`` and ``k = 2bτ``. The measure is passed as two
    parallel numpy float64 arrays; ``MeasureState`` never crosses the boundary.

    **NARROW scope**: D=1 constant-coefficient 4-Dirac pushforward (§38.3).
    Dirac count grows ×4 per step. Forward kernel = DiffusionChernoff (Brownian).

    Parameters
    ----------
    a : float
        Diffusion coefficient (``h = 2√(aτ)``). Must be finite.
    b : float
        Drift coefficient (``k = 2bτ``). Must be finite.
    c : float
        Reaction coefficient (mass factor ``1 + τc``). Must be finite.

    Raises
    ------
    SemiflowError
        kind='OutOfDomain' — non-finite coefficient.
    """

    def __init__(self, a: float, b: float, c: float) -> None: ...

    def step(
        self,
        tau: float,
        positions: NDArray[np.float64],
        weights: NDArray[np.float64],
        n_steps: int = 1,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Apply ``n_steps`` adjoint Fokker-Planck steps.

        The GIL is released during the multi-step Lemma A.1 push (ADR-0031).
        ``MeasureState`` never crosses the boundary (ADR-0138).

        Parameters
        ----------
        tau : float
            Step size (``> 0``, finite).
        positions : NDArray[np.float64]
            Dirac positions, 1-D float64, length ``n_diracs``.
        weights : NDArray[np.float64]
            Dirac weights, 1-D float64, same length as ``positions``.
        n_steps : int, optional
            Number of steps (default 1). Dirac count grows ×4 per step.

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.float64]]
            ``(positions, weights)`` after applying ``n_steps``.

        Raises
        ------
        SemiflowError
            kind='OutOfDomain'  — tau <= 0 or non-finite; n_steps == 0.
            kind='GridMismatch' — len(positions) != len(weights).
        """
        ...

    def total_variation(
        self,
        positions: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> float:
        """Return total variation ``‖ρ‖_TV = Σ|w_i|``."""
        ...

    def second_moment(
        self,
        positions: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> float:
        """Return second moment ``⟨x², ρ⟩ = Σ x_i² w_i``."""
        ...


# ---------------------------------------------------------------------------
# v8.1.0 C1 — SmolyakD6V8 (sparse-grid D=6, TIER 2, ADR-0138, ADR-0123 Amdt 1)
# ---------------------------------------------------------------------------

@final
class SmolyakD6V8:
    """Sparse-grid Smolyak Chernoff kernel, D=6, unit-diffusion (v8.1.0, ADR-0138).

    Applies one or more Chernoff steps of the unit-diffusion Smolyak operator
    to a flat 6-D numpy float64 array (length ``n_per_axis^6``).

    **NARROW scope**: unit a=I, b=0, c=0 only. Variable coefficients are NOT
    bound (TIER-3). Default level ℓ=9 (D+3=9 → 533 nodes). FFI/WASM deferred.

    Parameters
    ----------
    domain_lo : float
        Lower bound of each axis (finite, same for all 6 axes).
    domain_hi : float
        Upper bound (``> domain_lo``).
    n_per_axis : int
        Grid nodes per axis (``>= 4``).

    Raises
    ------
    SemiflowError
        kind='GridMismatch' — invalid domain or n_per_axis < 4.
        kind='OutOfDomain'  — non-finite domain bound.
    """

    def __init__(
        self,
        domain_lo: float,
        domain_hi: float,
        n_per_axis: int,
    ) -> None: ...

    def apply(
        self,
        tau: float,
        u0: NDArray[np.float64],
        n_steps: int = 1,
    ) -> NDArray[np.float64]:
        """Apply ``n_steps`` Smolyak Chernoff steps; return flat result.

        Parameters
        ----------
        tau : float
            Step size (``>= 0``, finite).
        u0 : NDArray[np.float64]
            Flat 6-D grid function, length ``n_per_axis^6``.
        n_steps : int, optional
            Number of Chernoff steps (default 1).

        Returns
        -------
        NDArray[np.float64]
            Flat output after applying ``n_steps``, same length as ``u0``.

        Raises
        ------
        SemiflowError
            kind='GridMismatch' — ``len(u0) != n_per_axis^6``.
            kind='OutOfDomain'  — ``tau < 0``, non-finite, or n_steps == 0.
        """
        ...

    def n_nodes(self) -> int:
        """Return the Smolyak sparse-grid node count."""
        ...

    def level(self) -> int:
        """Return the Smolyak level ℓ (default D+3=9)."""
        ...

    def size(self) -> int:
        """Return the total number of grid points (``n_per_axis^6``)."""
        ...


# ---------------------------------------------------------------------------
# v8.1.0 F4 — ComplexTripleJumpV8 (TIER 2, ADR-0138, ADR-0136 Amdt 2)
# ---------------------------------------------------------------------------

@final
class ComplexTripleJumpV8:
    """Order-4 complex triple-jump, filiform-N5 Carnot, D=5 only (v8.1.0, ADR-0138).

    Applies one step via ``Ψ(τ) = K(γ⋆τ) ∘ K((1−2γ⋆)τ) ∘ K(γ⋆τ)`` where
    K is the filiform-N5 palindromic Strang (math.md §28.bis.8, ADR-0136 Amdt 2).
    Complex substeps are internal; only the real projection ``Re(Ψ(τ)f)`` is exposed.

    **NARROW scope**: filiform-N5, D=5 ONLY. ``apply_complex``/``CplxGridFn5``
    are NOT exposed (ABI-safety invariant, ADR-0138). FFI/WASM deferred.

    Parameters
    ----------
    domain_lo : float
        Lower bound of each axis (finite, same for all 5 axes).
    domain_hi : float
        Upper bound (``> domain_lo``).
    n_per_axis : int
        Grid nodes per axis (``>= 4``).

    Raises
    ------
    SemiflowError
        kind='GridMismatch' — invalid domain or n_per_axis < 4.
        kind='OutOfDomain'  — non-finite domain bound.
    """

    def __init__(
        self,
        domain_lo: float,
        domain_hi: float,
        n_per_axis: int,
    ) -> None: ...

    def apply_real(
        self,
        tau: float,
        u0: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Apply one order-4 step; return real projection ``Re(Ψ(τ)f)``.

        The GIL is released during the triple complex Strang sweep (ADR-0031).
        ``Complex<f64>``/``CplxGridFn5`` never cross the boundary (ADR-0138).

        Parameters
        ----------
        tau : float
            Step size (``>= 0``, finite).
        u0 : NDArray[np.float64]
            Flat 5-D grid function, length ``n_per_axis^5``.

        Returns
        -------
        NDArray[np.float64]
            Real projection ``Re(Ψ(τ)f)``, flat float64, same length as ``u0``.

        Raises
        ------
        SemiflowError
            kind='GridMismatch' — ``len(u0) != n_per_axis^5``.
            kind='OutOfDomain'  — ``tau < 0`` or non-finite.
        """
        ...

    @staticmethod
    def verify_gamma_star() -> bool:
        """Return True iff γ⋆ satisfies ``2γ³+(1−2γ)³=0`` with Re>0.

        Independent self-check of the complex root (residual < 1e-12).
        """
        ...

    def size(self) -> int:
        """Return the total number of grid points (``n_per_axis^5``)."""
        ...


# ---------------------------------------------------------------------------
# v4.1 Phase D — Heisenberg / ζ⁴ / ζ⁶ Python parity (ADR-0028)
# ---------------------------------------------------------------------------

@final
class HypoellipticChernoffHeisenberg:
    """Heisenberg group H₁ Chernoff approximation (palindromic Strang-Hörmander).

    Implements ``exp(τ/4·X₁²) ∘ exp(τ/2·X₂²) ∘ exp(τ/4·X₁²)`` on ℝ³
    with ``X₁ = HeisenbergGroup::x1()``, ``X₂ = HeisenbergGroup::x2()``.

    Per ADR-0087 / math.md §28 AMENDMENT 2.  Step-2 Carnot bracket
    condition ``[X₁, X₂] ≈ (0, 0, 1)`` is verified at construction.

    Raises
    ------
    SemiflowError
        kind='OutOfDomain' if the bracket check fails.
    """

    def __init__(self) -> None:
        """Construct the Heisenberg group Chernoff approximation.

        Verifies the step-2 Carnot bracket ``[X₁, X₂] = ∂_t`` at the origin.

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' if bracket verification fails.
        """
        ...

    def order(self) -> int:
        """Return the approximation order (always 2 for Strang-Hörmander)."""
        ...

    def kernel(self, h: float, x: float, y: float, tc: float) -> float:
        """Evaluate the Heisenberg heat kernel ``p_h(x, y, tc)``.

        Convenience wrapper around :func:`heisenberg_heat_kernel`.

        Parameters
        ----------
        h : float
            Step parameter h > 0.
        x : float
            First horizontal coordinate.
        y : float
            Second horizontal coordinate.
        tc : float
            Vertical (centre) coordinate.

        Returns
        -------
        float
            Kernel value.
        """
        ...


def heisenberg_heat_kernel(h: float, x: float, y: float, tc: float) -> float:
    """Heisenberg group heat kernel oracle ``p_h(x, y, tc)`` (math.md §28 AMENDMENT 2).

    Computes the Gaveau-Hulanicki integral via 32-pt Gauss-Legendre quadrature.

    Parameters
    ----------
    h : float
        Step parameter h > 0.  Returns 0.0 for h ≤ 0.
    x : float
        First horizontal coordinate.
    y : float
        Second horizontal coordinate.
    tc : float
        Vertical (centre) coordinate.

    Returns
    -------
    float
        Kernel value.  At the origin (``x=y=tc=0``) equals ``2/(π²h²)``.
    """
    ...


@final
class Heat1DZeta4:
    """1-D heat equation with order-4-temporal ζ⁴ Chernoff kernel (v4.1).

    Solves ``∂_t u = ∂²u`` (unit diffusion ``a = 1``) using
    ``Diffusion4thZeta4Chernoff`` (order-4 temporal; ADR-0086 / Path ε).

    Default spatial sampling: CubicHermite. Use ``.with_chebyshev_sampling()``
    or ``.with_octonic_sampling()`` for higher spatial accuracy.
    Note: ``with_quintic_sampling`` / ``with_cubic_sampling`` removed at v7.0
    (ADR-0109 QuinticHermite removal clock). See ``docs/migration/v6-to-v7.md``.
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        n: int,
        u0: NDArray[np.float64],
        *,
        boundary: BoundaryLiteral = "reflect",
    ) -> None:
        """Create state on [xmin, xmax] with n grid nodes and initial datum u0.

        Raises
        ------
        SemiflowError
            kind='GridMismatch' if xmin >= xmax or n < 4 or len(u0) != n.
            kind='NanInf' if u0 contains NaN or Inf.
            kind='OutOfDomain' if boundary is not recognised.
        """
        ...

    def order(self) -> int:
        """Return the approximation order (always 4 for ζ⁴ kernel)."""
        ...


    def evolve(self, t: float, n_steps: int = 100) -> None:
        """Advance state by time t using n_steps Chernoff iterations.

        Mutates self in-place; GIL released during Rust compute (ADR-0031).

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' if t < 0, non-finite, or n_steps == 0.
        """
        ...

    def values(self) -> NDArray[np.float64]:
        """Return the current grid values as a 1-D float64 numpy array (copy)."""
        ...

    def __len__(self) -> int:
        """Return the number of grid nodes."""
        ...


@final
class Heat1DZeta6:
    """1-D heat equation with order-6-temporal ζ⁶ Chernoff kernel (v4.1).

    Solves ``∂_t u = ∂²u`` (unit diffusion ``a = 1``) using
    ``Diffusion6thZeta6Chernoff`` (order-6 temporal; ADR-0086 rung K=3).

    The inner ζ⁴ uses default CubicHermite spatial sampling.
    (QuinticHermite removed at v7.0; see ``docs/migration/v6-to-v7.md``.)
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        n: int,
        u0: NDArray[np.float64],
        *,
        boundary: BoundaryLiteral = "reflect",
    ) -> None:
        """Create state on [xmin, xmax] with n grid nodes and initial datum u0.

        Raises
        ------
        SemiflowError
            kind='GridMismatch' if xmin >= xmax or n < 4 or len(u0) != n.
            kind='NanInf' if u0 contains NaN or Inf.
            kind='OutOfDomain' if boundary is not recognised.
        """
        ...

    def order(self) -> int:
        """Return the approximation order (always 6 for ζ⁶ kernel)."""
        ...

    def evolve(self, t: float, n_steps: int = 100) -> None:
        """Advance state by time t using n_steps Chernoff iterations.

        Mutates self in-place; GIL released during Rust compute (ADR-0031).

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' if t < 0, non-finite, or n_steps == 0.
        """
        ...

    def values(self) -> NDArray[np.float64]:
        """Return the current grid values as a 1-D float64 numpy array (copy)."""
        ...

    def __len__(self) -> int:
        """Return the number of grid nodes."""
        ...


# ---------------------------------------------------------------------------
# ADR-0111 Wave P1 — 1-D diffusion completeness
# ---------------------------------------------------------------------------

@final
class Heat1DZeta8:
    """1-D heat equation with order-8-temporal zeta8 Chernoff kernel (v6.0.0).

    Solves du/dt = d^2u/dx^2 using Diffusion8thZeta8Chernoff
    (ADR-0088 Wave II, Chebyshev sampling ON by default).
    """
    def __init__(self, xmin: float, xmax: float, n: int,
                 u0: NDArray[np.float64], *, boundary: BoundaryLiteral = "reflect") -> None: ...
    def order(self) -> int: ...
    def evolve(self, t: float, n_steps: int = 100) -> None: ...
    def values(self) -> NDArray[np.float64]: ...
    def __len__(self) -> int: ...


@final
class TruncatedExp1D:
    """1-D diffusion with K=4 truncated-exp Chernoff kernel (CFL-conditional)."""
    def __init__(self, xmin: float, xmax: float, n: int,
                 u0: NDArray[np.float64], *, boundary: BoundaryLiteral = "reflect") -> None: ...
    def order(self) -> int: ...
    def evolve(self, t: float, n_steps: int = 100) -> None: ...
    def values(self) -> NDArray[np.float64]: ...
    def __len__(self) -> int: ...


@final
class TruncatedExp4th1D:
    """1-D diffusion with 4th-order truncated-exp Chernoff kernel (CFL-conditional)."""
    def __init__(self, xmin: float, xmax: float, n: int,
                 u0: NDArray[np.float64], *, boundary: BoundaryLiteral = "reflect") -> None: ...
    def order(self) -> int: ...
    def evolve(self, t: float, n_steps: int = 100) -> None: ...
    def values(self) -> NDArray[np.float64]: ...
    def __len__(self) -> int: ...


@final
class Strang1D:
    """1-D advection-diffusion via Strang operator splitting (global order 2).

    Solves du/dt = d^2u/dx^2 + b * du/dx. Default: b = 0.5.
    """
    def __init__(self, xmin: float, xmax: float, n: int,
                 u0: NDArray[np.float64], *, b: float = 0.5,
                 boundary: BoundaryLiteral = "reflect") -> None: ...
    def order(self) -> int: ...
    def evolve(self, t: float, n_steps: int = 100) -> None: ...
    def values(self) -> NDArray[np.float64]: ...
    def __len__(self) -> int: ...


# ---------------------------------------------------------------------------
# ADR-0111 Wave P2 — complex Schrödinger
# ---------------------------------------------------------------------------

@final
class SchrodingerComplex1D:
    """1-D Schrödinger equation with native complex state: ``iψ_t = (−½Δ + V)ψ``.

    Backed by ``SchrödingerChernoffComplex`` (ADR-0079 Option B, math.md §30.3):
    palindromic Strang splitting with Cayley–Crank-Nicolson kinetic step.
    Globally order 2; exactly unitary (‖ψ(t)‖₂ = ‖ψ(0)‖₂ to machine precision).

    Unlike ``Schrodinger1D`` (real-pair split), this class stores the wavefunction
    natively as a ``complex128`` array.
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        n: int,
        psi0: NDArray[np.complex128],
        *,
        boundary: BoundaryLiteral = "reflect",
    ) -> None:
        """Create a free-particle state (V = 0) from a complex128 initial array.

        Parameters
        ----------
        xmin : float
            Left boundary.
        xmax : float
            Right boundary (must be > xmin).
        n : int
            Number of grid nodes (must be >= 4).
        psi0 : NDArray[np.complex128]
            Initial wavefunction; length n; all finite.
        boundary : str, optional
            Boundary policy (keyword-only); one of ``"reflect"`` (default),
            ``"periodic"``, ``"zero"``, ``"linear"``.

        Raises
        ------
        SemiflowError
            kind='GridMismatch' if xmin >= xmax or n < 4 or len(psi0) != n.
            kind='NanInf' if psi0 contains NaN or Inf.
            kind='OutOfDomain' if boundary is not recognised.
        """
        ...

    @staticmethod
    def with_potential(
        xmin: float,
        xmax: float,
        n: int,
        v: NDArray[np.float64],
        psi0: NDArray[np.complex128],
        *,
        boundary: BoundaryLiteral = "reflect",
    ) -> "SchrodingerComplex1D":
        """Create state with a pre-sampled real potential V(x) and complex128 psi0.

        Parameters
        ----------
        v : NDArray[np.float64]
            Pre-sampled ``V(x_i)`` values; length n; all finite.
        psi0 : NDArray[np.complex128]
            Initial wavefunction; length n; all finite.
        boundary : str, optional
            Boundary policy (keyword-only); default ``"reflect"``.

        Raises
        ------
        SemiflowError
            kind='GridMismatch' / 'NanInf' / 'OutOfDomain' on invalid inputs.
        """
        ...

    def evolve(self, t: float, n_steps: int = 200) -> None:
        """Advance the wavefunction by time t using n_steps Chernoff steps.

        Mutates self in-place; returns None.  The GIL is released during the
        inner Rust compute loop (ADR-0031 three-phase pattern).

        **Negative t (D2 — ADR-0113)**: t may be negative for backward
        (time-reversed) unitary evolution.  The palindromic Strang kernel
        satisfies S(−τ) = S(τ)⁻¹ exactly (verified round-trip residual
        1.19e-13 for Schrodinger1D).  Norm is preserved to machine precision.

        Parameters
        ----------
        t : float
            Time to advance.  Must be finite; may be negative for backward
            unitary evolution.
        n_steps : int, optional
            Number of Chernoff steps (default 200).  Must be >= 1.

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' if t is non-finite or n_steps == 0.
        """
        ...

    def values(self) -> NDArray[np.complex128]:
        """Return current wavefunction as a complex128 numpy array of length n.

        Returns a copy of the internal state; mutations do not affect this object.
        The returned dtype is ``numpy.complex128`` (= ``numpy.cdouble``).
        """
        ...

    def __len__(self) -> int:
        """Return the number of grid nodes n."""
        ...

    def order(self) -> int:
        """Return the approximation order (always 2 for palindromic Strang)."""
        ...

    def norm_squared(self) -> float:
        """Return sum |psi_i|^2 * dx — grid-spacing-weighted squared L2 norm.

        Equals 1.0 for a normalised wavefunction.  Used to verify unitarity:
        ``abs(sch.norm_squared() / norm0 - 1) < 1e-12`` after evolution.
        """
        ...

# ---------------------------------------------------------------------------
# ADR-0111 Wave P3 — boundary-condition kernels
# ---------------------------------------------------------------------------

@final
class Resolvent1D:
    """1-D Laplace-Chernoff resolvent ``(λI − ∂²)⁻¹ g`` (M7).

    Computes ``R̃(λ) g = ∫₀^∞ exp(−λt) S(t)g dt`` via 32-point
    Gauss-Laguerre quadrature (Remizov 2025, Vladikavkaz Thm 3).
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        n: int,
        *,
        n_chernoff: int = 32,
    ) -> None:
        """Construct Resolvent1D (unit diffusion, Gauss-Laguerre-32).

        Parameters
        ----------
        xmin : float
            Left boundary.
        xmax : float
            Right boundary (must be > xmin).
        n : int
            Number of grid nodes (must be >= 4).
        n_chernoff : int, optional
            Chernoff truncation level for the inner heat semigroup (default 32).

        Raises
        ------
        SemiflowError
            kind='GridMismatch' if xmin >= xmax or n < 4.
            kind='OutOfDomain' if n_chernoff == 0.
        """
        ...

    def eval(
        self,
        lambda_: float,
        g: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate ``R̃(lambda) g`` and return the result.

        GIL released during inner Rust compute (ADR-0031).

        Parameters
        ----------
        lambda_ : float
            Resolvent parameter; must be > 0 and finite.
        g : NDArray[np.float64]
            Right-hand side; float64 array of length n.

        Returns
        -------
        NDArray[np.float64]
            ``R̃(lambda) g``; float64 array of length n.

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' if lambda <= 0 or non-finite.
            kind='GridMismatch' if len(g) != n.
            kind='NanInf' if g contains NaN or Inf.
        """
        ...

    def residual(
        self,
        lambda_: float,
        g: NDArray[np.float64],
    ) -> float:
        """Compute the residual ``‖(λI − ∂²) R̃(λ) g − g‖_∞``.

        Uses 3-point FD Laplacian on interior nodes.
        GIL released during compute (ADR-0031).

        Returns
        -------
        float
            Supremum residual over interior nodes.

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' if lambda <= 0 or n < 3.
        """
        ...

    def __len__(self) -> int:
        """Return the number of grid nodes."""
        ...


@final
class Killing1D:
    """1-D heat equation with absorbing (Dirichlet) BC via Feynman-Kac killing (M8).

    Solves ``∂_t u = ∂²u`` inside the box ``[lo, hi)``; values outside
    are zeroed at each Chernoff step (Butko 2018, order 1; math.md §21).
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        n: int,
        u0: NDArray[np.float64],
        *,
        lo: float = float("nan"),
        hi: float = float("nan"),
        boundary: BoundaryLiteral = "reflect",
    ) -> None:
        """Construct Killing1D.

        Parameters
        ----------
        xmin : float
            Left domain boundary.
        xmax : float
            Right domain boundary (must be > xmin).
        n : int
            Number of grid nodes (must be >= 4).
        u0 : NDArray[np.float64]
            Initial condition; length n.
        lo : float, optional
            Lower bound of the killing box (default = xmin + range/4).
        hi : float, optional
            Upper bound of the killing box, exclusive (default = xmax - range/4).
        boundary : str, optional
            Boundary policy; default ``"reflect"``.

        Raises
        ------
        SemiflowError
            kind='GridMismatch' if xmin >= xmax or n < 4 or len(u0) != n.
            kind='NanInf' if u0 contains NaN or Inf.
            kind='OutOfDomain' if lo >= hi or boundary unrecognised.
        """
        ...

    def order(self) -> int:
        """Return the approximation order (always 1 for Killing)."""
        ...

    def evolve(self, t: float, n_steps: int = 100) -> None:
        """Advance state by time t using n_steps Chernoff iterations.

        Mutates self in-place; GIL released during compute (ADR-0031).

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' if t < 0, non-finite, or n_steps == 0.
        """
        ...

    def values(self) -> NDArray[np.float64]:
        """Return current grid values as float64 numpy array (copy)."""
        ...

    def __len__(self) -> int:
        """Return the number of grid nodes."""
        ...


@final
class Reflected1D:
    """1-D heat equation with Neumann (reflecting) BC via the image method (M9).

    Solves ``∂_t u = ∂²u`` on ``[xmin, xmax]`` with zero-flux condition
    at ``x = origin``.  Backed by image-method Chernoff (Walsh 1986, order 2;
    math.md §25).
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        n: int,
        u0: NDArray[np.float64],
        *,
        origin: float = float("nan"),
        boundary: BoundaryLiteral = "reflect",
    ) -> None:
        """Construct Reflected1D.

        Parameters
        ----------
        xmin : float
            Left domain boundary (>= 0 for half-line idiom).
        xmax : float
            Right domain boundary.
        n : int
            Number of grid nodes (must be >= 4).
        u0 : NDArray[np.float64]
            Initial condition; length n.
        origin : float, optional
            Point on the reflecting boundary (default = xmin).
        boundary : str, optional
            Boundary policy; default ``"reflect"``.

        Raises
        ------
        SemiflowError
            kind='GridMismatch' if xmin >= xmax or n < 4 or len(u0) != n.
            kind='NanInf' if u0 contains NaN or Inf.
            kind='OutOfDomain' if boundary unrecognised.
        """
        ...

    def order(self) -> int:
        """Return the approximation order (2 — image method preserves order)."""
        ...

    def evolve(self, t: float, n_steps: int = 100) -> None:
        """Advance state by time t using n_steps Chernoff iterations.

        Mutates self in-place; GIL released during compute (ADR-0031).

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' if t < 0, non-finite, or n_steps == 0.
        """
        ...

    def values(self) -> NDArray[np.float64]:
        """Return current grid values as float64 numpy array (copy)."""
        ...

    def __len__(self) -> int:
        """Return the number of grid nodes."""
        ...


@final
class DirichletHeat2nd1D:
    """1-D heat equation with Dirichlet BC via the odd-image method (M11, §21.9, ADR-0176).

    Order-2 absorbing Dirichlet: ``u = 0`` at ``origin``.
    Backed by ``DirichletHeat2ndChernoff<DiffusionChernoff, HalfSpaceRegion>``.
    Sibling of ``Reflected1D`` (Neumann) and higher-order than ``Killing1D`` (order 1).
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        n: int,
        u0: NDArray[np.float64],
        *,
        origin: float = float("nan"),
        boundary: BoundaryLiteral = "reflect",
    ) -> None:
        """Construct DirichletHeat2nd1D.

        Parameters
        ----------
        xmin : float
            Left domain boundary.
        xmax : float
            Right domain boundary (must be > xmin).
        n : int
            Number of grid nodes (must be >= 4).
        u0 : NDArray[np.float64]
            Initial condition; length n, all finite.
        origin : float, optional
            Location of the absorbing wall (default = xmin).
        boundary : str, optional
            Background boundary policy; default ``"reflect"``.

        Raises
        ------
        SemiflowError
            kind='GridMismatch' if xmin >= xmax or n < 4 or len(u0) != n.
            kind='NanInf' if u0 contains NaN or Inf.
            kind='OutOfDomain' if boundary unrecognised.
        """
        ...

    def order(self) -> int:
        """Return the approximation order (2)."""
        ...

    def evolve(self, t: float, n_steps: int = 100) -> None:
        """Advance state by time t using n_steps Chernoff iterations.

        Mutates self in-place; GIL released during compute.

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' if t < 0, non-finite, or n_steps == 0.
        """
        ...

    def values(self) -> NDArray[np.float64]:
        """Return current grid values as float64 numpy array (copy)."""
        ...

    def __len__(self) -> int:
        """Return the number of grid nodes."""
        ...


@final
class Robin1D:
    """1-D heat equation with Robin BC via the skew image method (M10).

    Solves ``∂_t u = ∂²u`` with Robin condition ``α·u(origin) − β·∂_n u(origin) = 0``.
    Backed by ``RobinHeatChernoff`` (Carslaw-Jaeger 1959, order 1; math.md §3.5.tris).
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        n: int,
        u0: NDArray[np.float64],
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
        origin: float = float("nan"),
        boundary: BoundaryLiteral = "reflect",
    ) -> None:
        """Construct Robin1D.

        Parameters
        ----------
        xmin : float
            Left domain boundary.
        xmax : float
            Right domain boundary.
        n : int
            Number of grid nodes (must be >= 4).
        u0 : NDArray[np.float64]
            Initial condition; length n.
        alpha : float, optional
            Robin coefficient on u (default 1.0); must be >= 0.
        beta : float, optional
            Robin coefficient on du/dn (default 1.0); must be > 0.
        origin : float, optional
            Point on the Robin boundary (default = xmin).
        boundary : str, optional
            Boundary policy; default ``"reflect"``.

        Raises
        ------
        SemiflowError
            kind='GridMismatch' if xmin >= xmax or n < 4 or len(u0) != n.
            kind='NanInf' if u0 contains NaN or Inf.
            kind='OutOfDomain' if alpha < 0, beta <= 0, or boundary unrecognised.
        """
        ...

    def order(self) -> int:
        """Return the approximation order (always 1 for Robin skew-image)."""
        ...

    def evolve(self, t: float, n_steps: int = 100) -> None:
        """Advance state by time t using n_steps Chernoff iterations.

        Mutates self in-place; GIL released during compute (ADR-0031).

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' if t < 0, non-finite, or n_steps == 0.
        """
        ...

    def values(self) -> NDArray[np.float64]:
        """Return current grid values as float64 numpy array (copy)."""
        ...

    def __len__(self) -> int:
        """Return the number of grid nodes."""
        ...


# ---------------------------------------------------------------------------
# v6.3.0 — obstacle / variational-inequality Chernoff (math §44)
# ---------------------------------------------------------------------------

@final
class ObstacleChernoff:
    """1-D obstacle / variational-inequality Chernoff evolver (math §44).

    Implements ``V^{n+1} = Π_g( S(Δτ) Vⁿ )`` where ``S(Δτ)`` is a
    diffusion/convection-diffusion-reaction Chernoff step and
    ``Π_g(W) = max(W, g)`` is the metric projection onto the cone ``{V ≥ g}``
    (Theorem 44.1). Order 1 globally (§44.4).

    Generator: ``L = a u_xx + b u_x + c u`` (constant coefficients).

    When ``b == 0`` and ``c == 0`` (default) the fast-path
    ``DiffusionChernoff`` kernel is used.  When ``b ≠ 0`` or ``c ≠ 0`` a
    palindromic Strang split is used.  The obstacle projection caps the global
    order to 1 regardless.

    Exactly one of ``level`` or ``obstacle_array`` must be provided.
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        n: int,
        u0: NDArray[np.float64],
        *,
        a: float = 1.0,
        b: float = 0.0,
        c: float = 0.0,
        level: float = float("nan"),
        obstacle_array: NDArray[np.float64] | None = None,
    ) -> None:
        """Construct ObstacleChernoff.

        Parameters
        ----------
        xmin : float
            Left boundary.
        xmax : float
            Right boundary (must be > xmin).
        n : int
            Number of grid nodes (must be >= 4).
        u0 : NDArray[np.float64]
            Initial condition; float64 array of length n.
        a : float, optional
            Constant diffusion coefficient (> 0, default 1.0).
        b : float, optional
            Constant drift coefficient (default 0.0, finite).
        c : float, optional
            Constant reaction coefficient (default 0.0, finite).
            Positive c → growth; negative c → decay (``L u = c·u``).
        level : float, optional
            Constant obstacle floor ``g ≡ level`` (ConstantObstacle).
            Mutually exclusive with ``obstacle_array``.
        obstacle_array : NDArray[np.float64], optional
            Per-node obstacle values ``g(x_i)``, length n (ArrayObstacle).
            Mutually exclusive with ``level``.

        Raises
        ------
        SemiflowError
            kind='GridMismatch' if xmin >= xmax or n < 4 or len(u0) != n.
            kind='NanInf' if u0 or obstacle_array contains NaN or Inf.
            kind='OutOfDomain' if a <= 0 or b/c is non-finite or level is non-finite.
        """
        ...

    def order(self) -> int:
        """Return the approximation order (always 1; §44.4 projection cap)."""
        ...

    def evolve(self, t: float, n_steps: int = 100) -> NDArray[np.float64]:
        """Advance state by time t using n_steps Chernoff iterations.

        Returns the new grid values as float64 array of length n.
        Mutates self in-place; GIL released during compute (ADR-0031).

        The post-projection invariant ``V(t) >= g`` holds elementwise.

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' if t < 0, non-finite, or n_steps == 0.
        """
        ...

    def values(self) -> NDArray[np.float64]:
        """Return current grid values as float64 numpy array (copy)."""
        ...

    def evolve_active_set_adjoint(
        self,
        w_fwd: NDArray[np.float64],
        lam: NDArray[np.float64],
        tau: float,
    ) -> NDArray[np.float64]:
        """One active-set adjoint step (math §44.5 Theorem 44.3).

        NOTE: Raises SemiflowError(kind='OutOfDomain') always when the inner is
        DiffusionChernoff (no transpose-apply primitive, ADR-0114). Use the
        Adjoint wrapper for the self-adjoint forward-adjoint path.

        Parameters
        ----------
        w_fwd : NDArray[np.float64]
            Pre-projection forward state ``S(Δτ)Vⁿ``; length n.
        lam : NDArray[np.float64]
            Incoming costate ``∂/∂V^{n+1}``; length n.
        tau : float
            Step size Δτ (must be > 0).

        Returns
        -------
        NDArray[np.float64]
            Outgoing costate ``∂/∂Vⁿ``; length n.

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' when DiffusionChernoff inner used (no adjoint).
            kind='GridMismatch' if w_fwd or lam length != n.
        """
        ...

    def __len__(self) -> int:
        """Return the number of grid nodes."""
        ...


# ---------------------------------------------------------------------------
# ADR-0111 Wave P4 — nonautonomous + subordinated
# ---------------------------------------------------------------------------

@final
class Howland1D:
    """1-D nonautonomous heat via Howland lift (M11).

    Wraps ``HowlandLift<DiffusionChernoff<f64>>`` — the autonomous unit-diffusion
    generator lifted to ``L^2([0, t_horizon], L^2([xmin, xmax]))``.

    For an **autonomous** base generator, the Howland-lifted evolution at
    ``t = t_horizon`` equals the regular heat semigroup ``S(t_horizon)``.

    Parameters
    ----------
    xmin : float
        Left boundary of the spatial domain.
    xmax : float
        Right boundary (must be > xmin).
    n : int
        Number of spatial grid nodes (must be >= 4).
    u0 : NDArray[np.float64]
        Initial condition; length n.  Replicated across all n_t time slices.
    n_t : int, optional
        Number of temporal grid points (default 11).  Must be >= 2.
    t_horizon : float, optional
        Time horizon T (default 0.1).  Must be finite and > 0.
    boundary : BoundaryLiteral, optional
        Spatial boundary policy; default ``"reflect"``.

    Raises
    ------
    SemiflowError
        kind='GridMismatch' if xmin >= xmax or n < 4 or len(u0) != n.
        kind='NanInf' if u0 contains NaN or Inf.
        kind='OutOfDomain' if n_t < 2, t_horizon <= 0, or boundary unrecognised.
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        n: int,
        u0: NDArray[np.float64],
        *,
        n_t: int = 11,
        t_horizon: float = 0.1,
        boundary: BoundaryLiteral = "reflect",
    ) -> None: ...

    def order(self) -> int:
        """Return the approximation order (always 1 — left-endpoint shift)."""
        ...

    def delta_s(self) -> float:
        """Return the time-grid spacing ``delta_s = t_horizon / (n_t - 1)``."""
        ...

    def n_t(self) -> int:
        """Return the number of temporal grid points ``n_t``."""
        ...

    def t_horizon(self) -> float:
        """Return the time horizon ``T`` set at construction."""
        ...

    def evolve(self) -> None:
        """Advance the Howland state by one full ``t_horizon`` evolution.

        Uses ``n_steps = n_t - 1`` Chernoff iterations with ``tau = delta_s``
        (matched-step requirement of ``HowlandLift``).  Mutates self in-place;
        returns None.  GIL released during compute (ADR-0031).

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' if the internal matched-step constraint fires.
        """
        ...

    def values(self) -> NDArray[np.float64]:
        """Return the last time slice ``u(t_horizon, .)`` as float64 array (copy)."""
        ...

    def __len__(self) -> int:
        """Return the number of spatial grid nodes ``n``."""
        ...


@final
class Subordinated1D:
    """1-D subordinated heat semigroup via Bochner-Phillips calculus (M12).

    Wraps ``SubordinatedChernoff<DiffusionChernoff<f64>, Subordinator, f64>``
    (Butko 2018 Thm 2.1, math.md §37, ADR-0103).

    Backends (``subordinator=`` kwarg):

    - ``"stable"`` — alpha-stable: phi(lambda) = lambda^alpha (alpha in (0,1)).
    - ``"gamma"`` — Gamma: phi(lambda) = log(1 + lambda/c) (c > 0).
    - ``"inverse_gaussian"`` — IG: phi(lambda) = sqrt(c^2+2*lambda) - c (c > 0).

    Parameters
    ----------
    xmin : float
        Left boundary.
    xmax : float
        Right boundary (must be > xmin).
    n : int
        Number of grid nodes (must be >= 4).
    u0 : NDArray[np.float64]
        Initial condition; length n.
    subordinator : str, optional
        Backend selector: ``"stable"`` (default), ``"gamma"``, or
        ``"inverse_gaussian"``.
    alpha : float, optional
        Stability index for ``"stable"`` backend; must be in ``(0, 1)``.
        Default 0.5.
    c : float, optional
        Rate/drift parameter for ``"gamma"``/``"inverse_gaussian"``; must be > 0.
        Default 1.0.
    n_nodes : int, optional
        Number of GL-32 quadrature nodes (1–32); default 32.
    boundary : BoundaryLiteral, optional
        Boundary policy; default ``"reflect"``.

    Raises
    ------
    SemiflowError
        kind='GridMismatch' if grid or u0 are invalid.
        kind='NanInf' if u0 contains NaN or Inf.
        kind='OutOfDomain' if alpha/c/n_nodes out of valid range.
        kind='Unsupported' if ``subordinator`` string is not recognised.
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        n: int,
        u0: NDArray[np.float64],
        *,
        subordinator: str = "stable",
        alpha: float = 0.5,
        c: float = 1.0,
        n_nodes: int = 32,
        boundary: BoundaryLiteral = "reflect",
    ) -> None: ...

    def order(self) -> int:
        """Return the approximation order (always 1 — Butko 2018 Theorem 2.1)."""
        ...

    def evolve(self, t: float, n_steps: int = 100) -> None:
        """Advance state by time t using n_steps Chernoff iterations.

        Mutates self in-place; returns None.  GIL released during compute (ADR-0031).

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' if t < 0, non-finite, or n_steps == 0.
        """
        ...

    def values(self) -> NDArray[np.float64]:
        """Return current grid values as float64 numpy array (copy)."""
        ...

    def __len__(self) -> int:
        """Return the number of grid nodes (Subordinated1D)."""
        ...

# ---------------------------------------------------------------------------
# ADR-0111 Wave P5 — geometry: manifold + hypoelliptic backends
# ---------------------------------------------------------------------------

ManifoldLiteral = Literal["torus", "sphere2", "hyperbolic2"]

@final
class Manifold2D:
    """2-D Riemannian manifold Chernoff approximation (M13).

    Wraps ManifoldChernoff<M, f64> (MMRS 2023 *Math. Nachr.* Thm 1,
    math.md §24, ADR-0071).  Backend manifold selected via manifold= kwarg.

    Backends
    --------
    - ``"torus"``       — flat 2-torus T² (R ≡ 0; correction is identity).
    - ``"sphere2"``     — 2-sphere S²(r) (R ≡ 2/r²; radius sets r).
    - ``"hyperbolic2"`` — Poincaré disk H²(s) (R ≡ -2/s²; radius sets s).

    State type: GridFn2D<f64> — flat float64 array of length nx*ny.

    Parameters
    ----------
    x0min, x0max : float
        Chart-axis-0 boundary (must be finite and x0min < x0max).
    nx : int
        Number of nodes on axis 0 (must be >= 4).
    x1min, x1max : float
        Chart-axis-1 boundary (must be finite and x1min < x1max).
    ny : int
        Number of nodes on axis 1 (must be >= 4).
    u0 : NDArray[np.float64]
        Initial condition; flat float64 array of length nx*ny (row-major).
    manifold : ManifoldLiteral, optional
        Backend: ``"torus"`` (default), ``"sphere2"``, or ``"hyperbolic2"``.
    radius : float, optional
        Sphere radius (sphere2) or scale (hyperbolic2); default 1.0.
    curvature_correction : bool, optional
        If True, apply R/12 correction (MMRS 2023); default True.

    Raises
    ------
    SemiflowError
        kind='GridMismatch' if grid params or u0 length are invalid.
        kind='NanInf' if u0 contains NaN or Inf.
        kind='OutOfDomain' if radius <= 0.
        kind='Unsupported' if manifold string is not recognised.
    """

    def __init__(
        self,
        x0min: float,
        x0max: float,
        nx: int,
        x1min: float,
        x1max: float,
        ny: int,
        u0: NDArray[np.float64],
        *,
        manifold: ManifoldLiteral = "torus",
        radius: float = 1.0,
        curvature_correction: bool = True,
    ) -> None: ...

    def order(self) -> int:
        """Return the approximation order (1 or 2; 2 when curvature_correction=True and R≠0)."""
        ...

    def evolve(self, t: float, n_steps: int = 100) -> None:
        """Advance state by time t using n_steps Chernoff iterations.

        GIL released during compute (ADR-0031).

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' if t < 0, non-finite, or n_steps == 0.
        """
        ...

    def values(self) -> NDArray[np.float64]:
        """Return current chart values as float64 numpy array (copy, length nx*ny)."""
        ...

    def __len__(self) -> int:
        """Return nx * ny (total number of chart nodes)."""
        ...

@final
class HypoellipticChernoffKolmogorov:
    """Kolmogorov phase-space hypoelliptic Chernoff approximation (M14).

    Wraps KolmogorovHypoelliptic<f64> (= HypoellipticChernoff<f64, 2, 1>)
    with the palindromic Strang-Hörmander decomposition.

    Models ∂_t p = v ∂_x p + ½ ∂²_v p (Kolmogorov 1934).

    State type: GridFn2D<f64> — flat float64 array of length nx*nv.

    Parameters
    ----------
    xmin, xmax : float
        x-axis boundary (must be finite and xmin < xmax).
    nx : int
        Number of x-axis nodes (must be >= 4).
    vmin, vmax : float
        v-axis boundary (must be finite and vmin < vmax).
    nv : int
        Number of v-axis nodes (must be >= 4).
    u0 : NDArray[np.float64]
        Initial condition; flat float64 array of length nx*nv (row-major).

    Raises
    ------
    SemiflowError
        kind='GridMismatch' if grid params or u0 are invalid.
        kind='NanInf' if u0 contains NaN or Inf.
        kind='OutOfDomain' if Hörmander bracket check fails at origin.
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        nx: int,
        vmin: float,
        vmax: float,
        nv: int,
        u0: NDArray[np.float64],
    ) -> None: ...

    def order(self) -> int:
        """Return the approximation order (always 2 — palindromic Strang-Hörmander)."""
        ...

    def evolve(self, t: float, n_steps: int = 100) -> None:
        """Advance state by time t using n_steps Chernoff iterations.

        GIL released during compute (ADR-0031).

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' if t < 0, non-finite, or n_steps == 0.
        """
        ...

    def values(self) -> NDArray[np.float64]:
        """Return current phase-space values as float64 numpy array (copy, length nx*nv)."""
        ...

    def __len__(self) -> int:
        """Return nx * nv (total phase-space grid nodes)."""
        ...

@final
class HypoellipticChernoffEngel:
    """Engel step-3 Carnot group hypoelliptic Chernoff approximation (M15).

    Wraps HypoellipticChernoff<f64, 4, 2> with the palindromic
    Strang-Hörmander decomposition on ℝ⁴ (math.md §28.bis.2, ADR-0095).

    State type: GridFnND<f64, 4> — flat float64 array of length n**4.

    Parameters
    ----------
    xmin, xmax : float
        Common axis boundary for all 4 axes (must be finite and xmin < xmax).
    n : int
        Per-axis node count (must be >= 4).  All 4 axes share n.
    u0 : NDArray[np.float64]
        Initial condition; flat float64 array of length n**4.

    Raises
    ------
    SemiflowError
        kind='GridMismatch' if grid params or u0 length are invalid.
        kind='NanInf' if u0 contains NaN or Inf.
        kind='OutOfDomain' if Engel bracket check fails at the origin.
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        n: int,
        u0: NDArray[np.float64],
    ) -> None: ...

    def order(self) -> int:
        """Return the approximation order (always 2 — palindromic Strang-Hörmander)."""
        ...

    def evolve(self, t: float, n_steps: int = 10) -> None:
        """Advance state by time t using n_steps Chernoff iterations.

        GIL released during compute (ADR-0031).

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' if t < 0, non-finite, or n_steps == 0.
        """
        ...

    def values(self) -> NDArray[np.float64]:
        """Return current 4D-grid values as float64 numpy array (copy, length n**4)."""
        ...

    def __len__(self) -> int:
        """Return n**4 (total grid nodes)."""
        ...

# ---------------------------------------------------------------------------
# ADR-0111 Wave P6 — quantum graphs, matrix diffusion, point-eval, graph traj
# ---------------------------------------------------------------------------

@final
class QuantumGraph:
    """Metric graph for quantum graph heat equations (M16, ADR-0078, math §29).

    Factories: path, star, from_edges.
    """

    @staticmethod
    def path(n_edges: int, edge_length: float = 1.0, n_grid: int = 32) -> "QuantumGraph": ...
    @staticmethod
    def star(n_arms: int, edge_length: float = 1.0, n_grid: int = 32) -> "QuantumGraph": ...
    @staticmethod
    def from_edges(edges: NDArray[np.float64], n_grid: int = 32) -> "QuantumGraph": ...

    @property
    def n_vertices(self) -> int: ...
    @property
    def n_edges(self) -> int: ...
    @property
    def total_arc_length(self) -> float: ...

@final
class QuantumGraphHeat:
    """Quantum graph heat Chernoff approximation (M16).

    Solves ∂_t u = ½∂²_x u per edge with Kirchhoff vertex conditions.
    State: flat float64 array (edge-concatenated, length = n_edges * n_per_edge).
    """

    def __init__(self, qgraph: QuantumGraph) -> None: ...
    def set_state(self, u0: NDArray[np.float64]) -> None: ...
    def evolve(self, t: float, n_steps: int = 100) -> None: ...
    def values(self) -> NDArray[np.float64]: ...
    def __len__(self) -> int: ...

@final
class MatrixDiffusion1D:
    """Coupled 2-component 1D diffusion (M17, ADR-0082, math §33).

    State: flat float64 array of length 2*n (component-inner row-major).
    Specialised to M=2 for PyO3 const-generic boundary crossing.
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        n: int,
        u0: NDArray[np.float64],
        *,
        a_diag: float = 1.0,
        c_coupling: float = 0.0,
    ) -> None: ...
    def evolve(self, t: float, n_steps: int = 100) -> None: ...
    def values(self) -> NDArray[np.float64]: ...
    def order(self) -> int: ...
    def __len__(self) -> int: ...

@final
class PointEval:
    """Pointwise evaluation via DiffusionChernoff Backend A (M18, ADR-0080).

    eval_at(tau, u0, x, n_steps) → float (byte-identical to full-grid path).
    """

    def __init__(self, xmin: float, xmax: float, n: int) -> None: ...
    def eval_at(
        self,
        tau: float,
        u0: NDArray[np.float64],
        x: float,
        n_steps: int = 1,
    ) -> float: ...

def sample_gridfn2d(
    values: NDArray[np.float64],
    x0min: float,
    x0max: float,
    nx: int,
    x1min: float,
    x1max: float,
    ny: int,
    cx: float,
    cy: float,
) -> float:
    """Bilinear interpolation of a 2D grid function at chart position (cx, cy) (M18)."""
    ...

@final
class GraphTraj:
    """Fixed-topology graph trajectory (M22, ADR-0052, math §14.1).

    Single-segment, constant combinatorial Laplacian over [0, t_horizon].
    """

    def __init__(self, graph: Graph, t_horizon: float) -> None: ...

    @property
    def n_nodes(self) -> int: ...
    @property
    def t_horizon(self) -> float: ...
    @property
    def n_segments(self) -> int: ...

@final
class StrangGraph:
    """Palindromic Strang split for graph heat Chernoff kernels (M22, math §12.8).

    Order-2 via bipartite edge-parity coloring (commutativity guaranteed).

    Factories: from_path, from_cycle.
    """

    @staticmethod
    def from_path(graph: Graph) -> "StrangGraph": ...
    @staticmethod
    def from_cycle(graph: Graph) -> "StrangGraph": ...

    def evolve(
        self,
        t_final: float,
        n_steps: int,
        f0: NDArray[np.float64],
    ) -> NDArray[np.float64]: ...
    def order(self) -> int: ...

    @property
    def n_nodes(self) -> int: ...


# ---------------------------------------------------------------------------
# Wave P7 — multi-D anisotropic + 2D/3D variable-coefficient (ADR-0111 M19–M21)
# ---------------------------------------------------------------------------

@final
class AnisotropicShiftND2:
    """Anisotropic shift Chernoff on 2-D tensor-product grid (M19, order 1).

    Solves du/dt = A(x)∇²u + b(x)·∇u + c(x)u where A(x) is a 2×2 SPD tensor.
    Order 1 (honest, ADR-0112). F(0)=I guaranteed by π^{-D/2} normalization.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        a_values: NDArray[np.float64],
        *,
        b_values: Union[NDArray[np.float64], None] = None,
        c_values: Union[NDArray[np.float64], None] = None,
    ) -> None: ...

    def set_state(self, u0: NDArray[np.float64]) -> None: ...
    def evolve(self, t: float, n_steps: int = 100) -> None: ...
    def values(self) -> NDArray[np.float64]: ...
    def order(self) -> int: ...
    def __len__(self) -> int: ...


@final
class AnisotropicShiftND3:
    """Anisotropic shift Chernoff on 3-D tensor-product grid (M19, D=3, order 1).

    Same contract as AnisotropicShiftND2 but for 3 spatial dimensions.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        zmin: float,
        zmax: float,
        a_values: NDArray[np.float64],
        *,
        b_values: Union[NDArray[np.float64], None] = None,
        c_values: Union[NDArray[np.float64], None] = None,
    ) -> None: ...

    def set_state(self, u0: NDArray[np.float64]) -> None: ...
    def evolve(self, t: float, n_steps: int = 100) -> None: ...
    def values(self) -> NDArray[np.float64]: ...
    def order(self) -> int: ...
    def __len__(self) -> int: ...


@final
class NonSeparable2DAniso:
    """Non-separable 2D diffusion with position-dependent coupling β(x,y) (M20).

    Solves du/dt = u_xx + u_yy + β(x,y)·u_xy using pre-sampled β array.
    Anisotropic variant of NonSeparable2D; uses the performant array-based
    constructor (ADR-0034 / ADR-0111 §4).
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        nx: int,
        ymin: float,
        ymax: float,
        ny: int,
        beta_values: NDArray[np.float64],
        u0: NDArray[np.float64],
        *,
        beta_norm_bound: Union[float, None] = None,
        boundary: BoundaryLiteral = "reflect",
    ) -> None: ...

    def evolve(
        self, t: float, n_steps: int = 100
    ) -> NDArray[np.float64]: ...
    def __len__(self) -> int: ...


@final
class Heat2DVarA:
    """Variable-coefficient 2D heat Chernoff (M21, order 2).

    Solves du/dt = a_x(x)·u_xx + a_y(y)·u_yy via palindromic Strang splitting.
    Additive sibling of Heat2D for non-unit diffusion.
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        nx: int,
        ymin: float,
        ymax: float,
        ny: int,
        a_x: NDArray[np.float64],
        a_y: NDArray[np.float64],
        *,
        boundary: BoundaryLiteral = "reflect",
    ) -> None: ...

    def evolve(
        self,
        u0: NDArray[np.float64],
        tau: float,
        n_steps: int,
    ) -> NDArray[np.float64]: ...
    def order(self) -> int: ...
    def __len__(self) -> int: ...

    @property
    def nx(self) -> int: ...
    @property
    def ny(self) -> int: ...


@final
class Heat3DVarA:
    """Variable-coefficient 3D heat Chernoff (M21, order 2).

    Solves du/dt = a_x(x)·u_xx + a_y(y)·u_yy + a_z(z)·u_zz via Strang3D.
    Additive sibling of Heat3D for non-unit diffusion.
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        nx: int,
        ymin: float,
        ymax: float,
        ny: int,
        zmin: float,
        zmax: float,
        nz: int,
        a_x: NDArray[np.float64],
        a_y: NDArray[np.float64],
        a_z: NDArray[np.float64],
        *,
        boundary: BoundaryLiteral = "reflect",
    ) -> None: ...

    def evolve(
        self,
        u0: NDArray[np.float64],
        tau: float,
        n_steps: int,
    ) -> NDArray[np.float64]: ...
    def order(self) -> int: ...
    def __len__(self) -> int: ...

# ---------------------------------------------------------------------------
# Issue #1 — Adjoint-state parameter-sensitivity (ADR-0115)
# ---------------------------------------------------------------------------

def edge_weight_grad(
    graph: "Graph | GraphPath | None" = None,
    a: None = None,
    *,
    u0: NDArray[np.float64],
    dj_du_n: NDArray[np.float64],
    t: float,
    n_steps: int,
    rho_bar: float,
    params: "list[tuple[int, int]] | Literal['all_edges']",
) -> NDArray[np.float64]:
    """Adjoint-state gradient ``∂J/∂w`` for each requested edge weight.

    Computes the discrete adjoint-state gradient of a scalar functional
    ``J`` over an ``n_steps``-step Magnus K=4 graph-heat trajectory using the
    §42 state-adjoint backward sweep (math.md §43.4, NORMATIVE).

    IN-CORE MATH only; autograd / ML plumbing stays in ``revssm`` (ADR-0115).

    Parameters
    ----------
    graph : Graph or GraphPath, optional
        Fixed-topology graph.
    a : None
        Reserved; must be ``None`` (varcoef path deferred).
    u0 : NDArray[np.float64]
        Initial condition (shape ``(n_nodes,)``).
    dj_du_n : NDArray[np.float64]
        Terminal sensitivity ``∂J/∂u_n`` (shape ``(n_nodes,)``).
    t : float
        Total evolution time.
    n_steps : int
        Number of Magnus K=4 steps (``tau = t / n_steps``).
    rho_bar : float
        Upper bound on the Gershgorin radius ``ρ̄(L_G)``.
    params : list[tuple[int, int]] or "all_edges"
        Which edge weights to differentiate.  ``"all_edges"`` covers every
        undirected edge once (CSR row-major, ``i < j``).

    Returns
    -------
    NDArray[np.float64]
        ``∂J/∂w`` for each requested edge (same order as ``params``).

    Notes
    -----
    GIL released during the Rust compute loop (ADR-0031).
    """
    ...

# ---------------------------------------------------------------------------
# v8.3.0 C-9 — WentzellV8 + GammaFamily (ADR-0153, ADR-0151)
# ---------------------------------------------------------------------------

@final
class GammaFamily:
    """Ergonomic γ-schedule family for WentzellV8 (v8.3.0, ADR-0153).

    Expands ``Constant / Linear(a,b) / Exponential(rate)`` to a pre-sampled
    schedule of length ``n_steps`` at left-endpoint freeze points
    ``t_k = t_offset + k·τ``.
    "Covers 90% ergonomically; use ``WentzellV8(gamma_schedule=...)`` for
    arbitrary γ."

    **NARROW**: 1D half-line only; multi-D Wentzell deferred (math §49.7).
    """

    @staticmethod
    def constant(c: float) -> "GammaFamily":
        """Constant γ(t) = c.  ``c >= 0`` and finite."""
        ...

    @staticmethod
    def linear(a: float, b: float) -> "GammaFamily":
        """Linear γ(t) = a + b·t.  ``a >= 0`` and finite; ``b`` finite."""
        ...

    @staticmethod
    def exponential(rate: float) -> "GammaFamily":
        """Exponential γ(t) = exp(rate·t).  ``rate`` finite."""
        ...


@final
class WentzellV8:
    """Dynamic Wentzell/Robin BC evolver for 1D unit-diffusion heat (v8.3.0).

    Advances ``∂_t u = ∂_xx u`` on ``[domain_lo, domain_hi]`` (half-line) with
    the dynamic Wentzell BC ``∂_t u + γ(t)·∂_ν u + c·u = 0`` at ``domain_lo``,
    implemented via bulk–boundary Cayley Lie split (math §49, ADR-0151).

    **γ-schedule (primary API)**: ``gamma_schedule`` is a ``np.ndarray[float64]``
    of length ``n_steps``.  The host pre-samples its arbitrary γ at
    ``t_k = t_offset + k·τ`` (``τ = t / n_steps``) BEFORE evolving.
    **NORMATIVE**: sampling MUST match the left-endpoint freeze point exactly
    (math §49.2, Howland freeze) or a silent order-1 error results.

    **NARROW scope**: 1D half-line only (``dst.values[0]`` = boundary trace DOF).
    Multi-D true-product Wentzell state is deferred (math §49.7 NORMATIVE).
    Order = 1 (bulk↔boundary Lie split commutator nonzero, §49.8).

    Parameters
    ----------
    domain_lo : float
        Left boundary (half-line origin).
    domain_hi : float
        Right boundary (finite, ``> domain_lo``).
    n_grid : int
        Number of grid nodes (``>= 4``).
    u0 : NDArray[np.float64]
        Initial condition, 1-D float64, length ``n_grid``, all finite.
    n_steps : int
        Number of Chernoff steps per ``evolve`` call (``>= 1``).
    c_reaction : float
        Boundary reaction coefficient ``c >= 0`` (finite).
    gamma_schedule : NDArray[np.float64]
        Pre-sampled γ-schedule, float64, length ``n_steps``.
        Each entry must be ``>= 0`` and finite.

    Raises
    ------
    SemiflowError
        kind='GridMismatch' — geometry invalid or length mismatch.
        kind='NanInf'       — non-finite value in u0 or schedule.
        kind='OutOfDomain'  — c < 0, γ < 0, or n_steps == 0.
    """

    def __init__(
        self,
        domain_lo: float,
        domain_hi: float,
        n_grid: int,
        u0: NDArray[np.float64],
        n_steps: int,
        c_reaction: float,
        gamma_schedule: NDArray[np.float64],
    ) -> None: ...

    @classmethod
    def from_family(
        cls,
        domain_lo: float,
        domain_hi: float,
        n_grid: int,
        u0: NDArray[np.float64],
        n_steps: int,
        c_reaction: float,
        family: GammaFamily,
    ) -> "WentzellV8":
        """Construct from a GammaFamily (ergonomic sugar; expands to schedule).

        The schedule is built for ``t=1.0, t_offset=0.0`` as a template;
        call ``evolve(t, t_offset)`` with the actual time parameters.
        For time-varying γ, use the primary ``__init__`` with an explicit schedule.
        """
        ...

    def evolve(
        self,
        t: float,
        t_offset: float = 0.0,
    ) -> NDArray[np.float64]:
        """Advance by ``t`` and return evolved grid as numpy float64 array.

        Sweeps the γ-schedule once (``n_steps`` Chernoff steps), sampling
        ``schedule[k]`` at step k (left-endpoint freeze: ``t_k = t_offset + k·τ``).
        The GIL is released during the sweep (ADR-0031).
        Internal state is updated in-place (chainable).

        Parameters
        ----------
        t : float
            Time step (``> 0``, finite).
        t_offset : float, optional
            Absolute start time for γ sampling (default 0.0).

        Returns
        -------
        NDArray[np.float64]
            Evolved state, float64, length ``n_grid``.

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' — ``t <= 0`` or non-finite.
        """
        ...

    def size(self) -> int:
        """Return the number of grid nodes."""
        ...

    def n_steps(self) -> int:
        """Return the number of Chernoff steps."""
        ...

# ---------------------------------------------------------------------------
# v8.3.0 F2-ND — ResolventJump2DV8 + ResolventJump3DV8 (ADR-0153, ADR-0148)
# ---------------------------------------------------------------------------

@final
class ResolventJump2DV8:
    """Resolvent time-jump evaluator for 2D unit-diffusion heat (v8.3.0, ADR-0153).

    Evaluates ``e^{tA}g`` for a 2D LARGE step ``t`` via the TWS parabolic-contour
    inverse Laplace quadrature (math.md §47.8, ADR-0148).

    **NARROW scope**: self-adjoint / sectorial parabolic generators only (§47.8
    NORMATIVE). Non-sectorial generators are OUT of scope.  ``m_nodes >= 6``.

    ND layout (NORMATIVE, §3.1 V8_3_TIER3_BINDING_DESIGN.md)
    ---------------------------------------------------------
    Pass ``g`` as shape ``(nx, ny)`` — the binding calls ``g.ravel(order="F")``
    internally to match the Rust axis-0-fastest layout
    (``idx(i,j) = j·nx + i``).  A pre-raveled flat array of length ``nx·ny``
    is also accepted.  The returned array has shape ``(nx, ny)`` reshaped with
    ``order="F"``.  This is the NORMATIVE fix for the v8.1.0 C-vs-F-order bug.

    Parameters
    ----------
    xmin, xmax : float
        x-axis bounds (finite, xmin < xmax).
    nx : int
        Number of x-axis grid nodes (>= 4).
    ymin, ymax : float
        y-axis bounds (finite, ymin < ymax).
    ny : int
        Number of y-axis grid nodes (>= 4).
    m_nodes : int
        TWS contour node count (>= 6; M=8 recommended for |t|<=1).

    Raises
    ------
    SemiflowError
        ``kind='GridMismatch'`` — invalid grid geometry.
        ``kind='OutOfDomain'`` — m_nodes < 6.
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        nx: int,
        ymin: float,
        ymax: float,
        ny: int,
        m_nodes: int,
    ) -> None: ...

    def jump(
        self,
        t: float,
        g: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate ``e^{tA}g``.

        Parameters
        ----------
        t : float
            Time step (> 0, finite).
        g : NDArray[np.float64]
            Shape ``(nx, ny)`` (raveled ``order="F"`` internally) or flat
            ``float64`` array of length ``nx·ny``.

        Returns
        -------
        NDArray[np.float64]
            Result, shape ``(nx, ny)``, ``float64``, Fortran-order layout.

        Raises
        ------
        SemiflowError
            ``kind='GridMismatch'`` — g length != nx·ny.
            ``kind='OutOfDomain'`` — t <= 0 or non-finite.
        """
        ...

    def shape(self) -> tuple[int, int]:
        """Return ``(nx, ny)``."""
        ...

    def m_nodes(self) -> int:
        """Return the number of TWS contour nodes."""
        ...

@final
class ResolventJump3DV8:
    """Resolvent time-jump evaluator for 3D unit-diffusion heat (v8.3.0, ADR-0153).

    Evaluates ``e^{tA}g`` for a 3D LARGE step ``t`` via the TWS parabolic-contour
    inverse Laplace quadrature (math.md §47.8, ADR-0148).

    **NARROW scope**: self-adjoint / sectorial parabolic generators only (§47.8
    NORMATIVE).  ``m_nodes >= 6``.

    ND layout (NORMATIVE, §3.1 V8_3_TIER3_BINDING_DESIGN.md)
    ---------------------------------------------------------
    Pass ``g`` as shape ``(nx, ny, nz)`` — raveled ``order="F"`` internally to
    match ``idx(i,j,k) = k·nx·ny + j·nx + i``.  Returns shape ``(nx, ny, nz)``
    with ``order="F"``.

    Parameters
    ----------
    xmin, xmax : float  — x-axis bounds.
    nx : int            — x-axis grid nodes (>= 4).
    ymin, ymax : float  — y-axis bounds.
    ny : int            — y-axis grid nodes (>= 4).
    zmin, zmax : float  — z-axis bounds.
    nz : int            — z-axis grid nodes (>= 4).
    m_nodes : int       — TWS contour nodes (>= 6).

    Raises
    ------
    SemiflowError
        ``kind='GridMismatch'`` — invalid grid geometry.
        ``kind='OutOfDomain'`` — m_nodes < 6.
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        nx: int,
        ymin: float,
        ymax: float,
        ny: int,
        zmin: float,
        zmax: float,
        nz: int,
        m_nodes: int,
    ) -> None: ...

    def jump(
        self,
        t: float,
        g: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate ``e^{tA}g``.

        Parameters
        ----------
        t : float
            Time step (> 0, finite).
        g : NDArray[np.float64]
            Shape ``(nx, ny, nz)`` (raveled ``order="F"`` internally) or flat
            ``float64`` array of length ``nx·ny·nz``.

        Returns
        -------
        NDArray[np.float64]
            Result, shape ``(nx, ny, nz)``, ``float64``, Fortran-order layout.

        Raises
        ------
        SemiflowError
            ``kind='GridMismatch'`` — g length != nx·ny·nz.
            ``kind='OutOfDomain'`` — t <= 0 or non-finite.
        """
        ...

    def shape(self) -> tuple[int, int, int]:
        """Return ``(nx, ny, nz)``."""
        ...

    def m_nodes(self) -> int:
        """Return the number of TWS contour nodes."""
        ...


# ---------------------------------------------------------------------------
# v8.3.0 B-7 — ObstacleGammaV8 (TIER 2, ADR-0153)
# ---------------------------------------------------------------------------

@final
class ObstacleGammaV8:
    """Inactive-set Γ = V″ primitive for obstacle problems (v8.3.0, ADR-0153, §4.1).

    .. warning::

        **Honesty (NORMATIVE, math §44.5.bis)**: ``defined[i] == False`` means
        Γ is **REFUSED** (active set / contact line / one-node guard band) —
        it does **NOT** mean ``gamma[i] == 0``.  Callers MUST consult
        ``defined`` before reading ``gamma``.  Γ JUMPS across the free
        boundary ``x*`` (perpetual-put witness Γ(S*⁺)≈4.90, Γ(S*⁻)=0);
        no classical global Γ exists.  **D = 1 only** (multi-asset Γ deferred,
        §44.5.ter, ADR-0153).

    FFI/WASM obstacle bindings are DEFERRED (ADR-0153 opportunistic, §4).

    Parameters
    ----------
    domain_lo : float
        Left boundary (finite, < domain_hi).
    domain_hi : float
        Right boundary (finite, > domain_lo).
    n_grid : int
        Number of grid nodes (>= 4).
    level : float, optional
        Constant obstacle floor ``g ≡ level`` (keyword-only).
    obstacle_array : array-like, optional
        Per-node obstacle floor, length ``n_grid`` (keyword-only).

    Raises
    ------
    SemiflowError
        ``kind='GridMismatch'`` — invalid grid or length mismatch.
        ``kind='OutOfDomain'`` — invalid domain or ``n_grid < 4``.
        ``kind='NanInf'`` — obstacle contains NaN / Inf.
    """

    def __init__(
        self,
        domain_lo: float,
        domain_hi: float,
        n_grid: int,
        *,
        level: float = ...,
        obstacle_array: NDArray[np.float64] | None = None,
    ) -> None: ...

    def inactive_gamma(
        self,
        v: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_], int]:
        """Compute inactive-set Γ = V″ on the OPEN continuation set.

        Returns ``(gamma, defined, count)``:

        - ``gamma`` — float64 array length ``n_grid``.  Valid at ``defined==True``.
        - ``defined`` — **bool** array length ``n_grid``.  ``False`` means Γ is
          REFUSED (active set / contact / guard band).  Never interpret as "Γ=0".
        - ``count`` — number of defined nodes (= ``defined.sum()``).

        Parameters
        ----------
        v : NDArray[np.float64]
            Value field, length ``n_grid``.

        Raises
        ------
        SemiflowError
            ``kind='GridMismatch'`` — ``len(v) != n_grid``.
        """
        ...

    def size(self) -> int:
        """Return the number of grid nodes."""
        ...


# ---------------------------------------------------------------------------
# v8.3.0 B-7 — ObstacleNDV8 (D=2, TIER 2, ADR-0153)
# ---------------------------------------------------------------------------

@final
class ObstacleNDV8:
    """D=2 projective-splitting obstacle evolver (v8.3.0, ADR-0153, §4.2).

    Wraps ``ObstacleChernoffND<Strang2D, ConstantObstacle, f64, 2>`` —
    forward-only ``Π_g ∘ S(Δτ)`` on a 2D grid.

    .. note::

        **ND layout (NORMATIVE, §3.1 V8_3_TIER3_BINDING_DESIGN.md)**
        Input ``v`` should have shape ``(nx, ny)`` (raveled ``order="F"``
        internally) or flat length ``nx*ny``.  Output is a flat float64 array
        of length ``nx*ny``; use ``out.reshape((nx, ny), order="F")`` to
        recover 2D layout.

    .. note::

        **Scope (§44.5.ter / ADR-0153)**: D=2 forward evolution only.
        D=3, active-set adjoint, and inactive-set Γ remain D=1.
        FFI/WASM ND deferred.

    Parameters
    ----------
    xmin, xmax : float
        X-axis bounds.
    nx : int
        Number of x-axis nodes (>= 4).
    ymin, ymax : float
        Y-axis bounds.
    ny : int
        Number of y-axis nodes (>= 4).
    level : float
        Constant obstacle floor ``g ≡ level`` (finite).

    Raises
    ------
    SemiflowError
        ``kind='GridMismatch'`` — invalid grid geometry.
        ``kind='NanInf'`` — non-finite level.
        ``kind='OutOfDomain'`` — invalid bounds.
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        nx: int,
        ymin: float,
        ymax: float,
        ny: int,
        level: float,
    ) -> None: ...

    def apply(
        self,
        tau: float,
        v: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Apply one Chernoff step ``Π_g ∘ S(Δτ)`` to ``v``.

        Parameters
        ----------
        tau : float
            Time step (> 0, finite).
        v : NDArray[np.float64]
            Shape ``(nx, ny)`` or flat length ``nx*ny`` (Fortran-order layout).

        Returns
        -------
        NDArray[np.float64]
            Flat float64 array of length ``nx*ny`` (axis-0-fastest).
            ``out.reshape((nx, ny), order="F")`` recovers the 2D layout.

        Raises
        ------
        SemiflowError
            ``kind='GridMismatch'`` — ``v`` length mismatch.
            ``kind='OutOfDomain'`` — ``tau <= 0`` or non-finite.
        """
        ...

    def shape(self) -> tuple[int, int]:
        """Return ``(nx, ny)``."""
        ...


# ---------------------------------------------------------------------------
# v9 S³ carrier classes (ADR-0162): TtState, TtEvolver, TtCoupledEvolver,
# MeasureState, GridlessEvolver
# ---------------------------------------------------------------------------

@final
class TtState:
    """Tensor-train state built from rank-1 separable per-axis slices (v9, §52).

    Storage: O(d·n·r²) — curse-escaped for diagonal-A Gaussian diffusion.

    Parameters
    ----------
    slices : list[NDArray[np.float64]]
        Per-axis 1-D float64 arrays (at least one element each).

    Raises
    ------
    SemiflowError
        kind='GridMismatch' — empty list or any slice is empty.
        kind='NanInf' — any slice contains NaN or Inf.
    """

    def __init__(self, slices: list[NDArray[np.float64]]) -> None: ...

    def ndim(self) -> int:
        """Number of modes (dimensions d)."""
        ...

    def n_j(self, j: int) -> int:
        """Mode size ``n_j`` for axis ``j``.

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' if ``j >= ndim()``.
        """
        ...

    def peak_rank(self) -> int:
        """Peak bond rank (max over internal bonds)."""
        ...

    def storage_size(self) -> int:
        """Total number of stored scalars (working-set size)."""
        ...

    def inner_separable(self, functionals: list[NDArray[np.float64]]) -> float:
        """Separable inner product ``⟨f, u⟩`` for a list of per-axis numpy vectors.

        Parameters
        ----------
        functionals : list[NDArray[np.float64]]
            One 1-D float64 array per axis; length of each must match ``n_j(axis)``.

        Returns
        -------
        float
            The scalar projection value.

        Raises
        ------
        SemiflowError
            kind='GridMismatch' — list length != ndim or slice lengths mismatch.
            kind='NanInf' — NaN/Inf in any functional.
        """
        ...


@final
class TtEvolver:
    """Tensor-train Chernoff evolver for separable diagonal-A diffusion (v9, §52).

    Parameters
    ----------
    a : list[float]
        Per-axis diffusion coefficients (all >= 0, finite).
    b : list[float]
        Per-axis drift coefficients (finite).
    c : float
        Scalar reaction coefficient (finite).
    dom_min : list[float]
        Per-axis domain lower bounds.
    dom_max : list[float]
        Per-axis domain upper bounds (each > corresponding ``dom_min``).
    eps_round : float
        TT-rounding tolerance (finite, >= 0).

    Raises
    ------
    SemiflowError
        kind='GridMismatch' — empty axis list.
        kind='NanInf' — non-finite or negative ``a[j]``, non-finite ``b[j]``/``c``/domain.
    """

    def __init__(
        self,
        a: list[float],
        b: list[float],
        c: float,
        dom_min: list[float],
        dom_max: list[float],
        eps_round: float,
    ) -> None: ...

    def ndim(self) -> int:
        """Number of axes this evolver was built for."""
        ...

    def evolve(self, state: TtState, t_final: float, n_steps: int) -> None:
        """Evolve ``state`` in-place for time ``t_final`` using ``n_steps`` Chernoff steps.

        Parameters
        ----------
        state : TtState
            Mutable carrier state (rank-1 IC or previous result).
        t_final : float
            Total evolution time (>= 0, finite).
        n_steps : int
            Number of Chernoff time steps (>= 1).

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' — ``n_steps`` == 0, ``t_final`` non-finite/negative,
            or ``evolver.ndim() != state.ndim()``.
        """
        ...


@final
class VarCoefTtEvolver:
    """Additive-separable variable-coefficient TT evolver (ADR-0178, math §52.10, issue #2).

    Evolves a ``TtState`` by ``exp(τ·L)`` where ``L = Σⱼ Lⱼ``,
    ``Lⱼ = ∂_{xⱼ}(aⱼ·∂_{xⱼ}) + bⱼ·∂_{xⱼ} + vⱼ``.
    Operates on the SAME ``TtState`` carrier as ``TtEvolver``.
    Rank-1 IC → rank-1 output (bond-preserving, §52.10d).
    """

    def __init__(
        self,
        a_axis: list[list[float]],
        b_axis: list[list[float]],
        v_axis: list[list[float]],
        domain: list[tuple[float, float]],
        eps_round: float,
    ) -> None:
        """Construct a VarCoefTtEvolver.

        Parameters
        ----------
        a_axis : list[list[float]]
            Per-axis diffusion ``aⱼ(xⱼ)``; each inner list has length nⱼ,
            all entries strictly positive.
        b_axis : list[list[float]]
            Per-axis drift ``bⱼ(xⱼ)``; each inner list length nⱼ.
        v_axis : list[list[float]]
            Per-axis reaction ``vⱼ(xⱼ)``; empty list means zero on that axis.
        domain : list[tuple[float, float]]
            Per-axis ``(lo, hi)`` bounds.
        eps_round : float
            TT-rounding tolerance (finite, >= 0).

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' — ``d == 0``, shape mismatch, ``nⱼ < 2``,
            or any ``a_axis[j][i] <= 0``.
            kind='NanInf' — non-finite coefficient or domain bound.
        """
        ...

    def ndim(self) -> int:
        """Number of axes this evolver was built for."""
        ...

    def evolve(self, state: TtState, t_final: float, n_steps: int) -> None:
        """Evolve ``state`` in-place for time ``t_final`` using ``n_steps`` steps.

        Parameters
        ----------
        state : TtState
            Mutable carrier (same ``TtState`` as ``TtEvolver``).
        t_final : float
            Total evolution time (>= 0, finite).
        n_steps : int
            Number of time steps (>= 1).

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' — ``n_steps == 0``, ``t_final`` non-finite/negative,
            or ``ev.ndim() != state.ndim()``.
        """
        ...


@final
class TtCoupledEvolver:
    """Coupled tensor-train Chernoff evolver (v9, §52.9, ADR-0162).

    Extends ``TtEvolver`` with stable pair-factor coupling ``exp(τ·L_pair)``.
    With ``coupling=("None",)``, behaviour is bit-identical to ``TtEvolver``.

    Parameters
    ----------
    a : list[float]
        Per-axis diffusion coefficients (finite, >= 0).
    b : list[float]
        Per-axis drift (must all be 0.0 — drift deferred to v9.2.0).
    c : float
        Scalar reaction coefficient (finite).
    coupling : tuple
        One of:

        - ``("None",)`` — no coupling.
        - ``("Tridiagonal", rho)`` — nearest-neighbour chain.
        - ``("Pairs", [(j, k, rho), ...])`` — explicit adjacent pairs.
    dom_min : list[float]
        Per-axis domain lower bounds.
    dom_max : list[float]
        Per-axis domain upper bounds.
    eps_round : float
        TT-rounding tolerance (finite, >= 0).

    Raises
    ------
    SemiflowError
        kind='OutOfDomain' — any ``b[j]`` != 0, non-adjacent pair, non-SPD block.
        kind='GridMismatch' — empty axis list or invalid domain.
        kind='NanInf' — non-finite inputs.
    """

    def __init__(
        self,
        a: list[float],
        b: list[float],
        c: float,
        coupling: tuple[Any, ...],
        dom_min: list[float],
        dom_max: list[float],
        eps_round: float,
    ) -> None: ...

    def ndim(self) -> int:
        """Number of axes this evolver was built for."""
        ...

    def evolve(self, state: TtState, t_final: float, n_steps: int) -> None:
        """Evolve ``state`` in-place for time ``t_final`` using ``n_steps`` Chernoff steps.

        Same carrier (``TtState``) as ``TtEvolver.evolve``.

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' — ``n_steps`` == 0, ``t_final`` non-finite/negative,
            or ``evolver.ndim() != state.ndim()``.
        """
        ...


@final
class MeasureState:
    """Sparse weighted-Dirac particle ensemble on R (D=1, v9, §50).

    Represents a signed measure ``rho = sum_i w_i delta_{x_i}`` as a particle set.
    Curse-escape: the ``3^D`` dense tree is never materialised; only sparse
    marginals and scalar observables cross the Python boundary.

    Parameters
    ----------
    positions : NDArray[np.float64]
        Flat array of particle positions, length ``n_part`` (D=1).
    weights : NDArray[np.float64]
        Signed weights, length ``n_part``.
    dim : int
        Must equal 1 (compiled D); any other value raises ``kind='Unsupported'``.

    Raises
    ------
    SemiflowError
        kind='Unsupported' — ``dim`` != 1.
        kind='GridMismatch' — ``n_part`` == 0 or lengths mismatch.
        kind='NanInf' — NaN/Inf in positions or weights.
    """

    def __init__(
        self,
        positions: NDArray[np.float64],
        weights: NDArray[np.float64],
        dim: int,
    ) -> None: ...

    def n_diracs(self) -> int:
        """Number of Dirac atoms."""
        ...

    def total_variation(self) -> float:
        """Total-variation norm ``TV(rho)``."""
        ...

    def second_moment(self) -> float:
        """Second moment ``<x^2, rho>`` — tightness monitor (§38.5)."""
        ...

    def marginal(self, axis: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the marginal projection onto ``axis`` as (positions, weights) arrays.

        Parameters
        ----------
        axis : int
            Axis to project onto.  Must be 0 for D=1.

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.float64]]
            Both arrays have length ``n_diracs()``.

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' — ``axis >= 1`` (COMPILED_D).
        """
        ...


@final
class GridlessEvolver:
    """Gridless particle-ensemble Chernoff evolver (D=1, v9, §50, ADR-0155).

    Advances ``MeasureState`` via the 1-D 3-branch Chernoff kernel.

    Parameters
    ----------
    a : float
        Diffusion coefficient (>= 0, finite). D=1 scalar.
    b : float
        Drift coefficient (finite). D=1 scalar.
    c : float
        Reaction coefficient (finite).
    voronoi_cap : int, optional
        Particle cap for ``WeightedVoronoi`` reduction (default 64, must be >= 1).
    gaussian_background : bool, optional
        If ``True``, use the ``GaussianBackground`` stub instead of Voronoi cap
        (default ``False``).

    Raises
    ------
    SemiflowError
        kind='NanInf' — non-finite or negative ``a``, non-finite ``b``/``c``.
        kind='OutOfDomain' — ``voronoi_cap == 0`` with ``WeightedVoronoi`` selected.
    """

    def __init__(
        self,
        a: float,
        b: float,
        c: float,
        voronoi_cap: int = 64,
        gaussian_background: bool = False,
    ) -> None: ...

    def apply(self, tau: float, src: MeasureState, dst: MeasureState) -> None:
        """Apply one Chernoff step of size ``tau`` to ``src``, writing result into ``dst``.

        Parameters
        ----------
        tau : float
            Step size (>= 0, finite).
        src : MeasureState
            Read-only source.
        dst : MeasureState
            Overwritten with push-forward.

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' — ``tau`` < 0 or non-finite.
        """
        ...

    def evolve(self, state: MeasureState, t_final: float, n_steps: int) -> None:
        """Evolve ``state`` in-place for time ``t_final`` using ``n_steps`` Chernoff steps.

        Parameters
        ----------
        state : MeasureState
            Modified in-place.
        t_final : float
            Total time (>= 0, finite).
        n_steps : int
            Number of steps (>= 1).

        Raises
        ------
        SemiflowError
            kind='OutOfDomain' — ``n_steps`` == 0 or ``t_final`` non-finite/negative.
        """
        ...
