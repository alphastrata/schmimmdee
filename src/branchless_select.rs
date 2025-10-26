// ---------- Branchless Select Helpers ----------

/// Scalar implementation of branchless select.
pub fn branchless_select_scalar(condition: u64, a: u64, b: u64) -> u64 {
    (a & condition) | (b & !condition)
}

/// SIMD implementation of branchless select.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn branchless_select_simd(condition: &[u64], a: &[u64], b: &[u64]) -> Vec<u64> {
    use std::arch::x86_64::*;
    assert_eq!(condition.len(), a.len());
    assert_eq!(a.len(), b.len());

    let mut result = Vec::with_capacity(a.len());
    for i in (0..a.len()).step_by(4) {
        let cond_vec = _mm256_loadu_si256(condition.as_ptr().add(i) as *const __m256i);
        let a_vec = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
        let b_vec = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);

        let selected = _mm256_blendv_epi8(b_vec, a_vec, cond_vec); // blendv uses the sign bit of the mask
        let mut arr = [0u64; 4];
        _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, selected);
        result.extend_from_slice(&arr);
    }
    result
}

#[cfg(not(target_arch = "x86_64"))]
pub fn branchless_select_simd(condition: &[u64], a: &[u64], b: &[u64]) -> Vec<u64> {
    assert_eq!(condition.len(), a.len());
    assert_eq!(a.len(), b.len());
    let mut result = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        result.push(branchless_select_scalar(condition[i], a[i], b[i]));
    }
    result
}
