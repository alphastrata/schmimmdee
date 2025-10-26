use crc32fast::hash;

// ---------- CRC32 Helpers ----------

/// Scalar CRC32C calculation.
pub fn crc32c_scalar(data: &[u8]) -> u32 {
    hash(data)
}

/// SIMD CRC32C calculation using PCLMULQDQ.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")] // PCLMULQDQ requires SSE2
pub unsafe fn crc32c_simd(data: &[u8]) -> u32 {
    use std::arch::x86_64::*;
    let mut crc = !0u32; // Initial CRC value

    // Process 16-byte chunks
    let mut i = 0;
    while i + 16 <= data.len() {
        let chunk = _mm_loadu_si128(data.as_ptr().add(i) as *const __m128i);
        crc = _mm_crc32_u8(crc, _mm_extract_epi8(chunk, 0) as u8);
        crc = _mm_crc32_u8(crc, _mm_extract_epi8(chunk, 1) as u8);
        crc = _mm_crc32_u8(crc, _mm_extract_epi8(chunk, 2) as u8);
        crc = _mm_crc32_u8(crc, _mm_extract_epi8(chunk, 3) as u8);
        crc = _mm_crc32_u8(crc, _mm_extract_epi8(chunk, 4) as u8);
        crc = _mm_crc32_u8(crc, _mm_extract_epi8(chunk, 5) as u8);
        crc = _mm_crc32_u8(crc, _mm_extract_epi8(chunk, 6) as u8);
        crc = _mm_crc32_u8(crc, _mm_extract_epi8(chunk, 7) as u8);
        crc = _mm_crc32_u8(crc, _mm_extract_epi8(chunk, 8) as u8);
        crc = _mm_crc32_u8(crc, _mm_extract_epi8(chunk, 9) as u8);
        crc = _mm_crc32_u8(crc, _mm_extract_epi8(chunk, 10) as u8);
        crc = _mm_crc32_u8(crc, _mm_extract_epi8(chunk, 11) as u8);
        crc = _mm_crc32_u8(crc, _mm_extract_epi8(chunk, 12) as u8);
        crc = _mm_crc32_u8(crc, _mm_extract_epi8(chunk, 13) as u8);
        crc = _mm_crc32_u8(crc, _mm_extract_epi8(chunk, 14) as u8);
        crc = _mm_crc32_u8(crc, _mm_extract_epi8(chunk, 15) as u8);
        i += 16;
    }

    // Process remaining bytes with scalar
    for &byte in &data[i..] {
        crc = _mm_crc32_u8(crc, byte);
    }

    !crc // Final XOR
}

#[cfg(not(target_arch = "x86_64"))]
pub fn crc32c_simd(data: &[u8]) -> u32 {
    crc32c_scalar(data)
}
