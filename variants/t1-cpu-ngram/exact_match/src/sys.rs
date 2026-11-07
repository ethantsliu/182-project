//! The platform-specific pieces, so the crate also builds (and its tests run) off the x86-64 Linux nodes it is tuned
//! for: non-temporal stores and prefetches (x86-64; plain stores elsewhere), and the Linux mmap/madvise flags and
//! thread pinning (no-ops elsewhere). On x86-64 Linux every function here is what the code called directly before.

/// The Linux-only mmap / madvise flags; 0 (no effect) on other systems.
#[cfg(target_os = "linux")]
pub use libc::{MADV_DONTDUMP, MADV_DONTFORK, MADV_HUGEPAGE, MADV_POPULATE_READ, MADV_POPULATE_WRITE, MAP_POPULATE};
#[cfg(not(target_os = "linux"))]
pub const MADV_DONTDUMP: libc::c_int = 0;
#[cfg(not(target_os = "linux"))]
pub const MADV_DONTFORK: libc::c_int = 0;
#[cfg(not(target_os = "linux"))]
pub const MADV_HUGEPAGE: libc::c_int = 0;
#[cfg(not(target_os = "linux"))]
pub const MADV_POPULATE_READ: libc::c_int = 0;
#[cfg(not(target_os = "linux"))]
pub const MADV_POPULATE_WRITE: libc::c_int = 0;
#[cfg(not(target_os = "linux"))]
pub const MAP_POPULATE: libc::c_int = 0;

/// madvise, skipped for a flag this system lacks (0).
pub unsafe fn madvise(ptr: *mut libc::c_void, bytes: usize, advice: libc::c_int) {
    if cfg!(target_os = "linux") || advice != 0 {
        libc::madvise(ptr, bytes, advice);
    }
}

/// Pin the calling thread to `cpu` (Linux; elsewhere a no-op).
pub fn pin_current_thread(cpu: usize) {
    #[cfg(target_os = "linux")]
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
    #[cfg(not(target_os = "linux"))]
    let _ = cpu;
}

/// AVX-512 F/BW/DQ/VL, detected once; false off x86-64.
#[cfg(target_arch = "x86_64")]
pub fn avx512() -> bool {
    static AVX512: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512dq")
            && is_x86_feature_detected!("avx512vl")
    });
    *AVX512
}
#[cfg(not(target_arch = "x86_64"))]
pub fn avx512() -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
pub fn avx2() -> bool {
    is_x86_feature_detected!("avx2")
}

/// Write an even number of entries at a 16-byte aligned `dst` with non-temporal stores, which skip reading the
/// destination lines for ownership. Call `fence` once the thread is done.
#[cfg(target_arch = "x86_64")]
pub unsafe fn stream(dst: *mut u64, values: &[u64]) {
    use std::arch::x86_64::{__m128i, _mm_loadu_si128, _mm_stream_si128};
    for i in (0..values.len()).step_by(2) {
        _mm_stream_si128(dst.add(i).cast::<__m128i>(), _mm_loadu_si128(values.as_ptr().add(i).cast()));
    }
}
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn stream(dst: *mut u64, values: &[u64]) {
    std::ptr::copy_nonoverlapping(values.as_ptr(), dst, values.len());
}

pub fn fence() {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_sfence()
    };
    #[cfg(not(target_arch = "x86_64"))]
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
}

/// Prefetch the line at `p` into L1 (a no-op off x86-64).
#[inline(always)]
pub unsafe fn prefetch<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    std::arch::x86_64::_mm_prefetch(p.cast(), std::arch::x86_64::_MM_HINT_T0);
    #[cfg(not(target_arch = "x86_64"))]
    let _ = p;
}

/// Stream one 64-byte aligned cache line (8 entries) with a single non-temporal store. Only called when `avx512()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn stream_line_avx512(dst: *mut u64, src: *const u64) {
    use std::arch::x86_64::{_mm512_loadu_si512, _mm512_stream_si512};
    _mm512_stream_si512(dst.cast(), _mm512_loadu_si512(src.cast()));
}
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn stream_line_avx512(_dst: *mut u64, _src: *const u64) {
    unreachable!("avx512() is false off x86-64")
}

/// Stream a slice's buckets (`slots` entries each) to `dst`, each bucket's slots beyond its fill count as zeros
/// (masked loads, so the scratch is never cleared). Only called when `avx512()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl")]
pub unsafe fn out_avx512(slice: &[u64], fill: &[u8], dst: *mut u64, slots: usize) {
    use std::arch::x86_64::{_mm512_maskz_loadu_epi64, _mm512_stream_si512};
    let mut d = dst;
    for (j, &f) in fill.iter().enumerate() {
        let src = slice.as_ptr().add(j * slots);
        for l in (0..slots).step_by(8) {
            let mask = ((1u32 << (f as usize).saturating_sub(l).min(8)) - 1) as u8;
            _mm512_stream_si512(d.cast(), _mm512_maskz_loadu_epi64(mask, src.add(l).cast()));
            d = d.add(8);
        }
    }
}
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn out_avx512(_slice: &[u64], _fill: &[u8], _dst: *mut u64, _slots: usize) {
    unreachable!("avx512() is false off x86-64")
}
