// ---------- Greyscale Conversion Helpers ----------

/// Scalar greyscale conversion (simple average).
pub fn greyscale_scalar(pixels: &mut [u8]) {
    for chunk in pixels.chunks_exact_mut(4) {
        let avg = (chunk[0] as u16 + chunk[1] as u16 + chunk[2] as u16) / 3;
        chunk[0] = avg as u8;
        chunk[1] = avg as u8;
        chunk[2] = avg as u8;
        // Alpha channel remains unchanged
    }
}

/// SIMD greyscale conversion (simple average).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn greyscale_simd(pixels: &mut [u8]) {
    use std::arch::x86_64::*;

    let mut i = 0;
    while i + 16 <= pixels.len() {
        let pixel_chunk = _mm_loadu_si128(pixels.as_ptr().add(i) as *const __m128i);

        // Unpack to 16-bit integers for wider arithmetic
        let zero = _mm_setzero_si128();
        let p_lo = _mm_unpacklo_epi8(pixel_chunk, zero); // R0 G0 B0 A0 R1 G1 B1 A1
        let p_hi = _mm_unpackhi_epi8(pixel_chunk, zero); // R2 G2 B2 A2 R3 G3 B3 A3

        // Extract R, G, B components for each pixel
        // Pixel 0: R0, G0, B0
        let r0 = _mm_extract_epi16(p_lo, 0);
        let g0 = _mm_extract_epi16(p_lo, 1);
        let b0 = _mm_extract_epi16(p_lo, 2);

        // Pixel 1: R1, G1, B1
        let r1 = _mm_extract_epi16(p_lo, 4);
        let g1 = _mm_extract_epi16(p_lo, 5);
        let b1 = _mm_extract_epi16(p_lo, 6);

        // Calculate average for each pixel
        let avg0 = (r0 + g0 + b0) / 3;
        let avg1 = (r1 + g1 + b1) / 3;

        // Reconstruct the pixel data with greyscale values
        let result_lo = _mm_set_epi16(
            _mm_extract_epi16(p_lo, 7) as i16, // A1
            avg1 as i16, avg1 as i16, avg1 as i16, // B1 G1 R1
            _mm_extract_epi16(p_lo, 3) as i16, // A0
            avg0 as i16, avg0 as i16, avg0 as i16, // B0 G0 R0
        );

        // Store back to memory
        _mm_storeu_si128(pixels.as_mut_ptr().add(i) as *mut __m128i, result_lo);

        i += 16; // Process 4 pixels (16 bytes) at a time
    }

    // Handle remaining pixels with scalar
    if i < pixels.len() {
        greyscale_scalar(&mut pixels[i..]);
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn greyscale_simd(pixels: &mut [u8]) {
    greyscale_scalar(pixels)
}
