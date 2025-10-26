// ---------- Hamming Distance (VPOPCNT) Helpers ----------

/// Scalar implementation of Hamming distance for two slices of u64.
pub fn hamming_distance_scalar(a: &[u64], b: &[u64]) -> u64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x ^ y).count_ones() as u64)
        .sum()
}

/// SIMD implementation of Hamming distance for two slices of u64 using VPOPCNT.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512vpopcntdq")] // VPOPCNT requires AVX-512VPOPCNTDQ
pub fn hamming_distance_simd(a: &[u64], b: &[u64]) -> u64 {
    if !std::is_x86_feature_detected!("avx512vpopcntdq") {
        return hamming_distance_scalar(a, b);
    }

    use std::arch::x86_64::*;

    assert_eq!(a.len(), b.len());
    let mut p_a = a.as_ptr();
    let mut p_b = b.as_ptr();
    let mut len = a.len();

    let mut acc_vec = unsafe { _mm512_setzero_si512() };
    let chunk_size = 8; // 8 u64s in a 512-bit vector

    while len >= chunk_size {
        unsafe {
            let vec_a = _mm512_loadu_si512(p_a as *const _);
            let vec_b = _mm512_loadu_si512(p_b as *const _);

            let xor_vec = _mm512_xor_si512(vec_a, vec_b);
            let popcnt_vec = _mm512_popcnt_epi64(xor_vec);
            acc_vec = _mm512_add_epi64(acc_vec, popcnt_vec);

            p_a = p_a.add(chunk_size);
            p_b = p_b.add(chunk_size);
            len -= chunk_size;
        }
    }

    let mut sum = unsafe { _mm512_reduce_add_epi64(acc_vec) as u64 };
    if len > 0 {
        sum += hamming_distance_scalar(&a[a.len() - len..], &b[b.len() - len..]);
    }
    sum
}

#[cfg(not(target_arch = "x86_64"))]
pub fn hamming_distance_simd(a: &[u64], b: &[u64]) -> u64 {
    hamming_distance_scalar(a, b)
}
