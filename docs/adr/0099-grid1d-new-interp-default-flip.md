# ADR-0099 — `Grid1D::new` sub-grid interpolation default flip

- **Status**: SUPERSEDED by ADR-0109 — executed there, never independently implemented.
- **Date**: number allocated in the v4.6/v5.0 wave; **this file written retroactively 2026-08-17**.
- **Authors**: reconstructed — see "Provenance" below.
- **Superseded by**: ADR-0109 (SepticHermite virtual-node sampler, v6.0.0 BREAKING Window #3),
  which states in its own header: *"**Bundles**: ADR-0099 (`Grid1D::new` default flip) +
  ADR-0104 12-month deprecation clock ... into the v6.0.0 BREAKING window"*.
- **Later amended by**: ADR-0133 §46.5.bis (the same default applied to the `Dual<f64>` grid
  constructor so it composes with the f64 default).

## Why this file exists

**This is a reconstructed placeholder, not an original decision record.** No ADR-0099 was ever
written. The number was allocated and then cited 15 times across `ROADMAP.md`,
`contracts/semiflow-core.math.md` and `contracts/semiflow-core.properties.yaml` as a real,
load-bearing decision, leaving every one of those citations dangling — the single dangling ADR
reference in the repository (audit 2026-08-17: 194 ADR files on disk, 1 cited number missing).

Nothing here is invented. Every claim below is quoted or derived from surviving records, listed
under Provenance. Where the original rationale is simply not recoverable, this file says so
rather than filling the gap.

## Decision (as recorded by the citations)

`Grid1D::new` should stop defaulting to the historical interpolant and adopt the
then-current best sampler as its default — a BREAKING change to sampling behaviour for every
caller that had not set `InterpKind` explicitly.

Originally scheduled for **v5.0**, then **rescheduled to v6.0**. ROADMAP records the reason for
the reschedule: it *"frees v5.0 BREAKING budget for Chebyshev"* — i.e. the v5.0 breaking-change
budget was spent on the Chebyshev sampler fix instead, and the default flip was deferred one
window rather than dropped.

## What actually shipped

The flip landed inside ADR-0109's v6.0.0 window, and the target changed on the way. The
citations describe a **Chebyshev** default flip; what ADR-0109 actually made the default was
**`SepticHermite`**, after ADR-0097 AMENDMENT 1's RED verdict removed the Chebyshev default from
the table. Current state in `crates/semiflow/src/grid.rs`:

```rust
// v8.0 ADR-0133 §46.5.bis: default changed CubicHermite → SepticHermite
interp: InterpKind::SepticHermite,
```

So the decision was **executed but redirected**: the *flip* happened, the *destination* did not
match the ADR-0099-era citations. Readers following an ADR-0099 citation expecting a Chebyshev
default will be looking at `SepticHermite` in the code, and that is not a defect.

`InterpKind::ChebyshevSpectral { m }` was removed on the ADR-0104 12-month deprecation clock,
bundled into the same window.

## Consequences

- The dangling citations now resolve. They remain accurate about the *decision*, and this file
  is the place that records the destination change.
- Downstream ADRs, gates and calibrations keyed to the `SepticHermite` floor (notably ADR-0163's
  G3⁶-2D recalibration, whose whole cause was `SepticHermite` floor saturation) trace back to
  ADR-0109, not here.
- No gate is added or altered by this file. It is documentation only.

## Provenance

Reconstructed from, and verifiable against:

- `docs/adr/0109-septichermite-v6-0-0-breaking-window-3.md` header, **Bundles** line — the
  primary evidence that ADR-0099 existed as an allocated decision and was absorbed.
- `ROADMAP.md` — six citations, including *"ADR-0099 `Grid1D::new` Chebyshev DEFAULT flip
  RE-SCHEDULED to v6.0 (separate BREAKING decision; freed v5.0 BREAKING budget for Chebyshev
  sampler fix)"*.
- `contracts/semiflow-core.properties.yaml` — *"Bundles ADR-0099 default flip + ADR-0104
  12-month ..."*.
- `contracts/semiflow-core.math.md` — records the decision as PENDING at the time of writing.
- `crates/semiflow/src/grid.rs` — the shipped default, with its ADR-0133 comment.

**Not recoverable**: the original decision-maker, the exact date the number was allocated, and
any rationale beyond the reschedule note. Those fields are deliberately left unfilled rather
than guessed.
