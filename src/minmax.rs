// ---------- Min/Max Helpers ----------

use std::cmp::{min, max};

/// Scalar min/max for a slice of u32.
pub fn minmax_scalar(data: &[u32]) -> (u32, u32) {
    let mut min_val = u32::MAX;
    let mut max_val = u32::MIN;
    for &val in data {
        min_val = min(min_val, val);
        max_val = max(max_val, val);
    }
    (min_val, max_val)
}

/// SIMD min/max for a slice of u32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn minmax_simd(data: &[u32]) -> (u32, u32) {
    use std::arch::x86_64::*;

    if data.is_empty() {
        return (u32::MAX, u32::MIN);
    }

    let mut min_vec = _mm_set1_epi32(i32::MAX);
    let mut max_vec = _mm_set1_epi32(i32::MIN); // Use i32 for comparison with _mm_max_epi32

    let mut i = 0;
    while i + 4 <= data.len() {
        let chunk = _mm_loadu_si128(data.as_ptr().add(i) as *const __m128i);
        min_vec = _mm_min_epi32(min_vec, chunk);
        max_vec = _mm_max_epi32(max_vec, chunk);
        i += 4;
    }

    // Extract results
    let mut min_arr = [0i32; 4];
    let mut max_arr = [0i32; 4];
    _mm_storeu_si128(min_arr.as_mut_ptr() as *mut __m128i, min_vec);
    _mm_storeu_si128(max_arr.as_mut_ptr() as *mut __m128i, max_vec);

    let mut final_min = u32::MAX;
    let mut final_max = u32::MIN;

    for &val in &min_arr {
        final_min = min(final_min, val as u32);
    }
    for &val in &max_arr {
        final_max = max(final_max, val as u32);
    }

    // Handle remaining elements with scalar
    if i < data.len() {
        let (scalar_min, scalar_max) = minmax_scalar(&data[i..]);
        final_min = min(final_min, scalar_min);
        final_max = max(final_max, scalar_max);
    }

    (final_min, final_max)
}

#[cfg(not(target_arch = "x86_64"))]
pub fn minmax_simd(data: &[u32]) -> (u32, u32) {
    minmax_scalar(data)
}
