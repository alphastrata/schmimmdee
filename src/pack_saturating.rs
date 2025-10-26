// ---------- Pack Saturating (VPMOVDB) Helpers ----------

/// Scalar implementation of packing i32 to i8 with saturation.
pub fn pack_i32_to_i8_saturating_scalar(data: &[i32]) -> Vec<i8> {
    data.iter()
        .map(|&x| {
            if x > i8::MAX as i32 {
                i8::MAX
            } else if x < i8::MIN as i32 {
                i8::MIN
            } else {
                x as i8
            }
        })
        .collect()
}

/// SIMD implementation of packing i32 to i8 with saturation using VPMOVDB.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")] // VPMOVDB requires AVX-512F
pub unsafe fn pack_i32_to_i8_saturating_simd(data: &[i32]) -> Vec<i8> {
    use std::arch::x86_64::*;
    let mut results = Vec::with_capacity(data.len() / 4); // Each _mm512_cvtepi32_epi8 packs 16 i32 to 16 i8

    let mut i = 0;
    while i + 16 <= data.len() {
        let chunk = _mm512_loadu_si512(data.as_ptr().add(i) as *const __m512i);
        let packed = _mm512_cvtepi32_epi8(chunk); // Packs 16 i32 to 16 i8 with saturation

        let mut arr = [0i8; 16];
        _mm_storeu_si128(arr.as_mut_ptr() as *mut __m128i, packed); // Store 128-bit result
        results.extend_from_slice(&arr);
        i += 16;
    }

    if i < data.len() {
        results.extend_from_slice(&pack_i32_to_i8_saturating_scalar(&data[i..]));
    }
    results
}

#[cfg(not(target_arch = "x86_64"))]
pub fn pack_i32_to_i8_saturating_simd(data: &[i32]) -> Vec<i8> {
    pack_i32_to_i8_saturating_scalar(data)
}
