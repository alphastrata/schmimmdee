// ---------- Duplicate Detection (VPCONFLICT) Helpers ----------

/// Scalar implementation to check for duplicates in a small slice of u64s.
pub fn has_duplicates_scalar(chunk: &[u64]) -> bool {
    for i in 0..chunk.len() {
        for j in (i + 1)..chunk.len() {
            if chunk[i] == chunk[j] {
                return true;
            }
        }
    }
    false
}

/// SIMD implementation to check for duplicates in a small slice of u64s using VPCONFLICT.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512cd")] // VPCONFLICT requires AVX-512CD
pub unsafe fn has_duplicates_simd(chunk: &[u64]) -> bool {
    use std::arch::x86_64::*;
    assert_eq!(chunk.len(), 8); // VPCONFLICT operates on 8 u64s

    let vec_data = _mm512_loadu_si512(chunk.as_ptr() as *const __m512i);
    let conflict_mask = _mm512_conflict_epi64(vec_data);
    let or_reduction = _mm512_reduce_or_epi64(conflict_mask);

    or_reduction != 0
}

#[cfg(not(target_arch = "x86_64"))]
pub fn has_duplicates_simd(chunk: &[u64]) -> bool {
    has_duplicates_scalar(chunk)
}
