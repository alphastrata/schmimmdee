// ---------- Quantized Dot Product (VPDPBUSD) Helpers ----------

/// Scalar implementation of dot product for u8 and i8.
pub fn dot_product_u8s8_scalar(a: &[u8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i32) * (y as i32))
        .sum()
}

/// SIMD implementation of dot product for u8 and i8 using VPDPBUSD.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avxvnni")] // VPDPBUSD requires AVX-VNNI
pub unsafe fn dot_product_u8s8_simd(a: &[u8], b: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    assert_eq!(a.len(), b.len());

    let mut acc = _mm256_setzero_si256(); // Accumulator for 8 i32 results

    let mut i = 0;
    while i + 32 <= a.len() {
        let a_vec = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
        let b_vec = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);

        acc = _mm256_dpbusd_epi32(acc, a_vec, b_vec);
        i += 32;
    }

    // Sum the 8 i32 values in the accumulator
    let mut sum_arr = [0i32; 8];
    _mm256_storeu_si256(sum_arr.as_mut_ptr() as *mut __m256i, acc);
    let mut final_sum: i32 = sum_arr.iter().sum();

    // Handle remaining elements with scalar
    if i < a.len() {
        final_sum += dot_product_u8s8_scalar(&a[i..], &b[i..]);
    }
    final_sum
}

#[cfg(not(target_arch = "x86_64"))]
pub fn dot_product_u8s8_simd(a: &[u8], b: &[i8]) -> i32 {
    dot_product_u8s8_scalar(a, b)
}
