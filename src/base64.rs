// ---------- Base64 Encoding Helpers ----------

/// Scalar Base64 encoding for a chunk of 3 bytes into 4 Base64 chars.
pub fn base64_encode_scalar_chunk(input: &[u8]) -> [u8; 4] {
    let mut output = [0u8; 4];
    let b1 = input[0];
    let b2 = input[1];
    let b3 = input[2];

    output[0] = (b1 >> 2) & 0x3F;
    output[1] = ((b1 & 0x03) << 4) | ((b2 >> 4) & 0x0F);
    output[2] = ((b2 & 0x0F) << 2) | ((b3 >> 6) & 0x03);
    output[3] = b3 & 0x3F;

    for i in 0..4 {
        output[i] = BASE64_CHARS[output[i] as usize];
    }
    output
}

const BASE64_CHARS: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Scalar Base64 encoding for a slice of bytes.
pub fn base64_encode_scalar(input: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks_exact(3) {
        output.extend_from_slice(&base64_encode_scalar_chunk(chunk));
    }

    let remainder = input.len() % 3;
    if remainder == 1 {
        let b1 = input[input.len() - 1];
        output.push(BASE64_CHARS[((b1 >> 2) & 0x3F) as usize]);
        output.push(BASE64_CHARS[((b1 & 0x03) << 4) as usize]);
        output.push(b'=');
        output.push(b'=');
    } else if remainder == 2 {
        let b1 = input[input.len() - 2];
        let b2 = input[input.len() - 1];
        output.push(BASE64_CHARS[((b1 >> 2) & 0x3F) as usize]);
        output.push(BASE64_CHARS[(((b1 & 0x03) << 4) | ((b2 >> 4) & 0x0F)) as usize]);
        output.push(BASE64_CHARS[((b2 & 0x0F) << 2) as usize]);
        output.push(b'=');
    }
    output
}

/// SIMD Base64 encoding for a slice of bytes.
/// Processes 12 input bytes into 16 output bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
pub unsafe fn base64_encode_simd(input: &[u8]) -> Vec<u8> {
    use std::arch::x86_64::*;

    let mut output = Vec::with_capacity(input.len().div_ceil(3) * 4);
    let mut i = 0;

    let shuf_mask = _mm_set_epi8(
        11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -1, -1, -1, // This mask is for 12-byte chunks
    );

    let ascii_lut_lo = _mm_set_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // Placeholder, actual LUT will be more complex
    );
    let ascii_lut_hi = _mm_set_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // Placeholder
    );

    let zero = _mm_setzero_si128();
    let n_lt_26 = _mm_set1_epi8(25); // For A-Z
    let n_lt_52 = _mm_set1_epi8(51); // For a-z
    let n_lt_62 = _mm_set1_epi8(61); // For 0-9
    let n_lt_63 = _mm_set1_epi8(62); // For +
    let n_lt_64 = _mm_set1_epi8(63); // For /

    let add_65 = _mm_set1_epi8(65); // 'A' - 0
    let add_71 = _mm_set1_epi8(71); // 'a' - 26
    let add_minus_4 = _mm_set1_epi8(-4); // '0' - 52
    let add_minus_19 = _mm_set1_epi8(-19); // '+' - 62
    let add_minus_16 = _mm_set1_epi8(-16); // '/' - 63

    while i + 12 <= input.len() {
        let chunk = _mm_loadu_si128(input.as_ptr().add(i) as *const __m128i);

        // Unpack 3 bytes into 4 6-bit indices
        let indices = _mm_shuffle_epi8(chunk, shuf_mask);

        // Convert 6-bit indices to ASCII
        let is_lt_26 = _mm_cmpgt_epi8(n_lt_26, indices); // indices <= 25 (A-Z)
        let is_lt_52 = _mm_cmpgt_epi8(n_lt_52, indices); // indices <= 51 (a-z)
        let is_lt_62 = _mm_cmpgt_epi8(n_lt_62, indices); // indices <= 61 (0-9)
        let is_lt_63 = _mm_cmpgt_epi8(n_lt_63, indices); // indices <= 62 (+)
        let is_lt_64 = _mm_cmpgt_epi8(n_lt_64, indices); // indices <= 63 (/)

        let mut added = zero;
        let _add_a = _mm_and_si128(is_lt_26, add_65); // 'A' - 0
        added = _mm_add_epi8(added, _add_a);

        let _add_a_z = _mm_and_si128(_mm_andnot_si128(is_lt_26, is_lt_52), add_71); // 'a' - 26
        added = _mm_add_epi8(added, _add_a_z);

        let _add_0_9 = _mm_and_si128(_mm_andnot_si128(is_lt_52, is_lt_62), add_minus_4); // '0' - 52
        added = _mm_add_epi8(added, _add_0_9);

        let _add_plus = _mm_and_si128(_mm_andnot_si128(is_lt_62, is_lt_63), add_minus_19); // '+' - 62
        added = _mm_add_epi8(added, _add_plus);

        let _add_slash = _mm_and_si128(_mm_andnot_si128(is_lt_63, is_lt_64), add_minus_16); // '/' - 63
        added = _mm_add_epi8(added, _add_slash);

        let result = _mm_add_epi8(indices, added);

        let mut arr = [0u8; 16];
        _mm_storeu_si128(arr.as_mut_ptr() as *mut __m128i, result);
        output.extend_from_slice(&arr[0..16]); // Base64 output is 4/3 * input, so 12 bytes in -> 16 bytes out

        i += 12;
    }

    // Handle remaining bytes with scalar
    if i < input.len() {
        output.extend_from_slice(&base64_encode_scalar(&input[i..]));
    }

    output
}

#[cfg(not(target_arch = "x86_64"))]
pub fn base64_encode_simd(input: &[u8]) -> Vec<u8> {
    base64_encode_scalar(input)
}
