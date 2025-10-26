// ---------- Extract Bitfields (VPMULTISHIFTQB) Helpers ----------

/// Scalar implementation to extract 8-bit bitfields from a u64.
/// `control` specifies the starting bit for each of the 8 8-bit fields.
pub fn multishift_extract_scalar(data: &[u64], controls: &[u64]) -> Vec<u64> {
    assert_eq!(data.len(), controls.len());
    let mut results = Vec::with_capacity(data.len());

    for i in 0..data.len() {
        let mut result_u64 = 0u64;
        let current_data = data[i];
        let current_control = controls[i];

        for byte_idx in 0..8 {
            let shift_amount = (current_control >> (byte_idx * 8)) & 0x3F; // 0-63
            let extracted_byte = (current_data >> shift_amount) & 0xFF;
            result_u64 |= extracted_byte << (byte_idx * 8);
        }
        results.push(result_u64);
    }
    results
}

/// SIMD implementation to extract 8-bit bitfields from u64s using VPMULTISHIFTQB.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512vbmi")] // VPMULTISHIFTQB requires AVX-512VBMI
pub unsafe fn multishift_extract_simd(data: &[u64], controls: &[u64]) -> Vec<u64> {
    use std::arch::x86_64::*;
    assert_eq!(data.len(), controls.len());
    let mut results = Vec::with_capacity(data.len());

    let mut i = 0;
    while i + 8 <= data.len() {
        let data_vec = _mm512_loadu_si512(data.as_ptr().add(i) as *const __m512i);
        let control_vec = _mm512_loadu_si512(controls.as_ptr().add(i) as *const __m512i);

        let extracted_vec = _mm512_multishift_epi64_epi8(control_vec, data_vec);

        let mut arr = [0u64; 8];
        _mm512_storeu_si512(arr.as_mut_ptr() as *mut __m512i, extracted_vec);
        results.extend_from_slice(&arr);
        i += 8;
    }

    if i < data.len() {
        results.extend_from_slice(&multishift_extract_scalar(&data[i..], &controls[i..]));
    }
    results
}

#[cfg(not(target_arch = "x86_64"))]
pub fn multishift_extract_simd(data: &[u64], controls: &[u64]) -> Vec<u64> {
    multishift_extract_scalar(data, controls)
}
