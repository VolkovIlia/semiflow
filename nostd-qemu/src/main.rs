//! Bare-metal run of the `semiflow-nostd-check` scenarios on Cortex-M (QEMU).
//!
//! `semiflow` is built `#![no_std]` + `alloc` with default features off, so all
//! `f64` math goes through `libm` — soft-float on the Cortex-M3 (`thumbv7m`),
//! the single-precision FPU plus soft `f64` on the Cortex-M4F (`thumbv7em`).
//! Prints one line per scenario over semihosting and exits with status 0 only
//! if every scenario met its tolerance.

#![no_std]
#![no_main]

use core::{
    alloc::{GlobalAlloc, Layout},
    mem::MaybeUninit,
    sync::atomic::{AtomicUsize, Ordering},
};

use cortex_m_rt::{entry, exception, ExceptionFrame};
use cortex_m_semihosting::{debug, hprintln};
use embedded_alloc::LlffHeap;

/// Heap size. Peak use of the scenarios is reported at the end of the run.
const HEAP_SIZE: usize = 1024 * 1024;

/// Heap allocator that records the high-water mark of bytes in use.
struct PeakHeap {
    heap: LlffHeap,
    peak: AtomicUsize,
}

impl PeakHeap {
    fn used(&self) -> usize {
        self.heap.used()
    }

    /// High-water mark since the last call; restarts from current use.
    fn take_peak(&self) -> usize {
        self.peak.swap(self.used(), Ordering::Relaxed)
    }
}

/// Host wall-clock time in centiseconds (semihosting `SYS_CLOCK`).
fn clock_cs() -> usize {
    // SAFETY: SYS_CLOCK takes no argument and only reads the host clock.
    unsafe { cortex_m_semihosting::syscall!(CLOCK) }
}

// SAFETY: forwards to `LlffHeap`, which is a valid `GlobalAlloc`; the only
// extra work is reading its usage counter.
unsafe impl GlobalAlloc for PeakHeap {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = self.heap.alloc(layout);
        self.peak.fetch_max(self.heap.used(), Ordering::Relaxed);
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.heap.dealloc(ptr, layout);
    }
}

#[global_allocator]
static HEAP: PeakHeap = PeakHeap {
    heap: LlffHeap::empty(),
    peak: AtomicUsize::new(0),
};

/// Hand the static heap region to the allocator. Call once, before any allocation.
fn init_heap() {
    static mut HEAP_MEM: [MaybeUninit<u8>; HEAP_SIZE] = [MaybeUninit::uninit(); HEAP_SIZE];
    // SAFETY: runs once at the top of `main`, before any allocation, and
    // `HEAP_MEM` is referenced nowhere else.
    unsafe {
        HEAP.heap
            .init(core::ptr::addr_of_mut!(HEAP_MEM) as usize, HEAP_SIZE);
    }
}

/// Format centiseconds as seconds (`12.34s`).
#[derive(Clone, Copy)]
struct Secs(usize);

impl core::fmt::Display for Secs {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}.{:02}s", self.0 / 100, self.0 % 100)
    }
}

/// Print one result line; `peak` is the scenario's heap high-water mark.
fn report(
    scenario: &semiflow_nostd_check::Scenario,
    result: &semiflow_nostd_check::ScenarioResult,
    peak: usize,
    elapsed: Secs,
) {
    match result {
        Ok(o) => hprintln!(
            "PASS {} err={:e} tol={:e} digest=0x{:016x} heap_peak={} leaked={} time={}",
            scenario.name,
            o.err,
            o.tol,
            o.digest,
            peak,
            HEAP.used(),
            elapsed
        ),
        Err(e) => hprintln!("FAIL {} {} time={}", scenario.name, e, elapsed),
    }
}

/// Exit QEMU through semihosting with the given status.
fn exit(success: bool) -> ! {
    debug::exit(if success {
        debug::EXIT_SUCCESS
    } else {
        debug::EXIT_FAILURE
    });
    loop {
        cortex_m::asm::wfi();
    }
}

#[entry]
fn main() -> ! {
    init_heap();
    hprintln!("semiflow no_std + libm check ({})", env!("TARGET_TRIPLE"));
    let start = clock_cs();
    let mut last = start;
    let mut overall_peak = 0;
    let summary = semiflow_nostd_check::run_all(|scenario, result| {
        let now = clock_cs();
        let peak = HEAP.take_peak();
        overall_peak = overall_peak.max(peak);
        report(scenario, result, peak, Secs(now - last));
        last = now;
    });
    hprintln!(
        "summary: {} passed, {} failed; heap peak {} of {} bytes; {}",
        summary.passed,
        summary.failed,
        overall_peak,
        HEAP_SIZE,
        Secs(clock_cs() - start)
    );
    hprintln!(
        "{}",
        if summary.all_passed() {
            "ALL PASS"
        } else {
            "FAILED"
        }
    );
    exit(summary.all_passed())
}

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    hprintln!("PANIC: {}", info);
    exit(false)
}

#[exception]
unsafe fn HardFault(frame: &ExceptionFrame) -> ! {
    hprintln!("HARD FAULT: {:?}", frame);
    exit(false)
}
