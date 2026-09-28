# nostd-qemu: semiflow on bare-metal Cortex-M

This crate checks that `semiflow` runs as `#![no_std]` + `alloc` on a
microcontroller with no operating system, and that the results are correct
there. Compiling for a bare-metal target only shows that the code builds. This
harness runs it and compares the output with known answers.

With default features off, `semiflow` does all `f64` math through the `libm`
crate. That path is never exercised by `semiflow`'s own test suite, because the
suite's dev-dependencies turn on `num-traits/std`. The scenarios live in
[`crates/semiflow-nostd-check`](../crates/semiflow-nostd-check). That crate
depends on `semiflow` alone, with default features off. Each scenario is a
small problem with a closed-form or independent answer: heat flow checked
against the exact Gaussian, Chebyshev boundary folding, Schrödinger norm
conservation, `expmv`, graph Krylov/Chebyshev and implicit-Euler PCG against
dense Padé, the Gamma-subordinated semigroup, gridless particle moments,
Smolyak quadrature, and reverse-mode AD with `√n` checkpoints. The same
scenarios run on the host with `cargo test -p semiflow-nostd-check --release`.

The binary boots on QEMU's Arm MPS2 boards:

| Target | Board | CPU | Floating point |
|---|---|---|---|
| `thumbv7m-none-eabi` | `mps2-an385` | Cortex-M3 | all software |
| `thumbv7em-none-eabihf` | `mps2-an386` | Cortex-M4F | hardware `f32`, software `f64` |

It uses `cortex-m-rt` for startup and `embedded-alloc` for a 1 MiB heap. It
prints one line per scenario over semihosting and exits with status 0 only if
every scenario passes. A failed tolerance, a panic, an out-of-memory error or a
HardFault all exit with a non-zero status.

## Running

You need QEMU (`apt-get install qemu-system-arm`) and the Rust targets:

```sh
rustup target add thumbv7m-none-eabi thumbv7em-none-eabihf
cd nostd-qemu
cargo run --release --target thumbv7m-none-eabi     # Cortex-M3
cargo run --release --target thumbv7em-none-eabihf  # Cortex-M4F
```

The QEMU command lines are set as runners in `.cargo/config.toml`. The memory
layout is in `memory.x`: 4 MiB of code at `0x0000_0000` and 4 MiB of RAM at
`0x2000_0000`.

Each line of output reports the measured error, its tolerance, the scenario's
peak heap use, the heap still allocated afterwards (`leaked`, which should be
0) and the elapsed host time:

```text
PASS heat_shift1d err=3.2113543939227274e-4 tol=5e-4 heap_peak=4824 leaked=0 time=1.41s
...
summary: 14 passed, 0 failed; heap peak 525552 of 1048576 bytes; 20.65s
ALL PASS
```

`libm` is plain software built on IEEE-754 arithmetic. In practice the errors
printed here have matched the host run of `cargo test -p semiflow-nostd-check`
bit for bit on both boards.

This crate has its own `[workspace]` and `Cargo.lock`. The repository
workspace lists it under `exclude`, because it builds only for `thumbv7*`
targets.
