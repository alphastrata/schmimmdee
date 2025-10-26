// ---------- Shuffle (PSHUFB) Helpers ----------

/// Scalar RGBA to ARGB conversion for a 16-byte chunk (4 pixels).
pub fn shuffle_rgba_to_argb_scalar(chunk: &[u8]) -> [u8; 16] {
    let mut out = [0u8; 16];
    for i in 0..4 {
        let base = i * 4;
        out[(i * 4)] = chunk[base + 3]; // A
        out[i * 4 + 1] = chunk[base]; // R
        out[i * 4 + 2] = chunk[base + 1]; // G
        out[i * 4 + 3] = chunk[base + 2]; // B
    }
    out
}

/// SIMD RGBA to ARGB conversion for a 16-byte chunk (4 pixels) using PSHUFB.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
pub unsafe fn shuffle_rgba_to_argb_simd(chunk: &[u8]) -> [u8; 16] {
    use std::arch::x86_64::*;
    const MASK: [u8; 16] = [
        3, 2, 1, 0, // pixel 0
        7, 6, 5, 4, // pixel 1
        11, 10, 9, 8, // pixel 2
        15, 14, 13, 12, // pixel 3
    ];
    let vec = _mm_loadu_si128(chunk.as_ptr() as *const __m128i);
    let mask = _mm_loadu_si128(MASK.as_ptr() as *const __m128i);
    let shuffled = _mm_shuffle_epi8(vec, mask);
    let mut out = [0u8; 16];
    _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, shuffled);
    out
}

#[cfg(not(target_arch = "x86_64"))]
pub fn shuffle_rgba_to_argb_simd(chunk: &[u8]) -> [u8; 16] {
    shuffle_rgba_to_argb_scalar(chunk)
}
