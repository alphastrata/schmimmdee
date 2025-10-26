// ---------- Find MSB (VPLZCNT) Helpers ----------

/// Scalar implementation to find the Most Significant Bit (MSB) position for u32.
/// Returns 31 for 0, otherwise 31 - leading_zeros.
pub fn find_msb_scalar(data: &[u32]) -> Vec<u32> {
    data.iter()
        .map(|&x| {
            if x == 0 {
                0 // Or handle as an error/special case
            } else {
                31 - x.leading_zeros()
            }
        })
        .collect()
}

/// SIMD implementation to find the MSB position for u32 using VPLZCNT.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512cd")] // VPLZCNT requires AVX-512CD
pub unsafe fn find_msb_simd(data: &[u32]) -> Vec<u32> {
    use std::arch::x86_64::*;
    let mut results = Vec::with_capacity(data.len());

    let mut i = 0;
    while i + 16 <= data.len() {
        let data_vec = _mm512_loadu_si512(data.as_ptr().add(i) as *const __m512i);
        let lzcnt_vec = _mm512_lzcnt_epi32(data_vec);

        // MSB position = 31 - LZCNT
        let thirty_one = _mm512_set1_epi32(31);
        let msb_vec = _mm512_sub_epi32(thirty_one, lzcnt_vec);

        let mut arr = [0u32; 16];
        _mm512_storeu_si512(arr.as_mut_ptr() as *mut __m512i, msb_vec);
        results.extend_from_slice(&arr);
        i += 16;
    }

    if i < data.len() {
        results.extend_from_slice(&find_msb_scalar(&data[i..]));
    }
    results
}

#[cfg(not(target_arch = "x86_64"))]
pub fn find_msb_simd(data: &[u32]) -> Vec<u32> {
    find_msb_scalar(data)
}
