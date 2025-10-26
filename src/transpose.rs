// ---------- Transpose (VPERMW/D/Q) Helpers ----------

use std::mem;

/// Scalar 8x8 u32 matrix transpose.
pub fn transpose_8x8_u32_scalar(matrix: &[u32; 64]) -> [u32; 64] {
    let mut transposed = [0u32; 64];
    for i in 0..8 {
        for j in 0..8 {
            transposed[j * 8 + i] = matrix[i * 8 + j];
        }
    }
    transposed
}

/// SIMD 8x8 u32 matrix transpose using AVX2 permutations.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn transpose_8x8_u32_simd(matrix: &[u32; 64]) -> [u32; 64] {
    use std::arch::x86_64::*;

    let mut rows: [__m256i; 8] = mem::zeroed();
    for i in 0..8 {
        rows[i] = _mm256_loadu_si256(matrix.as_ptr().add(i * 8) as *const __m256i);
    }

    // Stage 1: Transpose 4x4 blocks of 32-bit integers
    let t0 = _mm256_unpacklo_epi32(rows[0], rows[1]);
    let t1 = _mm256_unpackhi_epi32(rows[0], rows[1]);
    let t2 = _mm256_unpacklo_epi32(rows[2], rows[3]);
    let t3 = _mm256_unpackhi_epi32(rows[2], rows[3]);
    let t4 = _mm256_unpacklo_epi32(rows[4], rows[5]);
    let t5 = _mm256_unpackhi_epi32(rows[4], rows[5]);
    let t6 = _mm256_unpacklo_epi32(rows[6], rows[7]);
    let t7 = _mm256_unpackhi_epi32(rows[6], rows[7]);

    // Stage 2: Transpose 2x2 blocks of 64-bit elements
    let tt0 = _mm256_unpacklo_epi64(t0, t2);
    let tt1 = _mm256_unpackhi_epi64(t0, t2);
    let tt2 = _mm256_unpacklo_epi64(t1, t3);
    let tt3 = _mm256_unpackhi_epi64(t1, t3);
    let tt4 = _mm256_unpacklo_epi64(t4, t6);
    let tt5 = _mm256_unpackhi_epi64(t4, t6);
    let tt6 = _mm256_unpacklo_epi64(t5, t7);
    let tt7 = _mm256_unpackhi_epi64(t5, t7);

    // Stage 3: Permute 128-bit lanes
    rows[0] = _mm256_permute2x128_si256(tt0, tt4, 0x20);
    rows[1] = _mm256_permute2x128_si256(tt1, tt5, 0x20);
    rows[2] = _mm256_permute2x128_si256(tt2, tt6, 0x20);
    rows[3] = _mm256_permute2x128_si256(tt3, tt7, 0x20);
    rows[4] = _mm256_permute2x128_si256(tt0, tt4, 0x31);
    rows[5] = _mm256_permute2x128_si256(tt1, tt5, 0x31);
    rows[6] = _mm256_permute2x128_si256(tt2, tt6, 0x31);
    rows[7] = _mm256_permute2x128_si256(tt3, tt7, 0x31);

    let mut transposed_matrix = [0u32; 64];
    for i in 0..8 {
        _mm256_storeu_si256(
            transposed_matrix.as_mut_ptr().add(i * 8) as *mut __m256i,
            rows[i],
        );
    }
    transposed_matrix
}

#[cfg(not(target_arch = "x86_64"))]
pub fn transpose_8x8_u32_simd(matrix: &[u32; 64]) -> [u32; 64] {
    transpose_8x8_u32_scalar(matrix)
}
