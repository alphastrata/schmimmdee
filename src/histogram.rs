// ---------- Histogram Helpers ----------

/// Scalar histogram calculation for u8 values.
pub fn histogram_scalar(data: &[u8]) -> [u32; 256] {
    let mut hist = [0u32; 256];
    for &val in data {
        hist[val as usize] += 1;
    }
    hist
}

/// SIMD histogram calculation for u8 values.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn histogram_simd(data: &[u8]) -> [u32; 256] {
    use std::arch::x86_64::*;
    let mut hist = [0u32; 256];

    // Process 32-byte chunks
    let mut i = 0;
    while i + 32 <= data.len() {
        let chunk = _mm256_loadu_si256(data.as_ptr().add(i) as *const __m256i);

        // This is a simplified placeholder. A real SIMD histogram
        // would use techniques like `_mm256_shuffle_epi8` and
        // `_mm256_sad_epu8` or AVX-512VP2INTERSECTD for efficiency.
        // For now, we just process byte by byte within the SIMD loop.
        let mut arr = [0u8; 32];
        _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, chunk);
        for &val in &arr {
            hist[val as usize] += 1;
        }
        i += 32;
    }

    // Process remaining bytes with scalar
    for &val in &data[i..] {
        hist[val as usize] += 1;
    }
    hist
}

#[cfg(not(target_arch = "x86_64"))]
pub fn histogram_simd(data: &[u8]) -> [u32; 256] {
    histogram_scalar(data)
}
