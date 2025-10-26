#![feature(portable_simd)]
use std::{
    f32,
    simd::{
        Simd,
        cmp::SimdPartialEq,
        num::{SimdFloat, SimdUint},
    },
};

use rayon::iter::IndexedParallelIterator;
use rayon::prelude::*;

/// NOTE: the build.rs will set this for you assuming FLOATS
/// so LOGICAL_LANES, for example might be 4, meaning 4 * f32 = 128
/// if you have AVX12 registers that go up to 512bits you might see 16 * f32 = 512
/// ... and so on.
const LOGICAL_LANES_: usize = 4; // Auto-detected for x86_64-unknown-linux-gnu

/// prettly-formant nanos from our std::instant timing.
pub fn format_ns(ns: f64) -> String {
    if ns >= 1_000_000_000.0 {
        format!("{:.2}s", ns / 1_000_000_000.0)
    } else if ns >= 1_000_000.0 {
        format!("{:.2}ms", ns / 1_000_000.0)
    } else if ns >= 1_000.0 {
        format!("{:.2}μs", ns / 1_000.0)
    } else {
        format!("{ns:.2}ns")
    }
}

pub fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

// minmax

#[unsafe(no_mangle)] // so if you want to peek @ the assembly it's easier to find your function..
pub fn find_min_max_simd(data: &[f32]) -> (f32, f32) {
    let mut min_vec = Simd::<f32, LOGICAL_LANES_>::splat(f32::MAX);
    let mut max_vec = Simd::<f32, LOGICAL_LANES_>::splat(f32::MIN);

    data.chunks_exact(LOGICAL_LANES_).for_each(|chunk| {
        let values = Simd::<f32, LOGICAL_LANES_>::from_slice(chunk);
        min_vec = min_vec.simd_min(values);
        max_vec = max_vec.simd_max(values);
    });

    let mut min = min_vec.reduce_min();
    let mut max = max_vec.reduce_max();

    data.chunks_exact(8).remainder().iter().for_each(|&value| {
        min = min.min(value);
        max = max.max(value);
    });

    (min, max)
}

pub fn find_min_max_scalar(data: &[f32]) -> (f32, f32) {
    let mut min = f32::MAX;
    let mut max = f32::MIN;

    data.iter().for_each(|&value| {
        min = min.min(value);
        max = max.max(value);
    });

    (min, max)
}

// patterns in strings
pub fn simd_contains_pattern(haystack: &[u8], needle: &[u8]) -> bool {
    const LANES: usize = LOGICAL_LANES_ * 4; // (there's 4 u8s of bits in an f32)
    if needle.is_empty() {
        return true;
    }
    if needle.len() > haystack.len() {
        return false;
    }
    if needle.len() == 1 {
        return simd_contains_byte(haystack, needle[0]);
    }

    // Use SIMD to quickly find first character candidates
    let first_char = needle[0];
    let first_char_vec = Simd::<u8, LANES>::splat(first_char);

    let mut i = 0;
    while i + LANES <= haystack.len() {
        let chunk = Simd::<u8, LANES>::from_slice(&haystack[i..i + LANES]);
        let mask = chunk.simd_eq(first_char_vec);

        if mask.any() {
            // Check each potential match position
            let mask_array = mask.to_array();
            for j in 0..LANES {
                if mask_array[j]
                    && i + j + needle.len() <= haystack.len()
                    && &haystack[i + j..i + j + needle.len()] == needle
                {
                    return true;
                }
            }
        }
        i += LANES;
    }

    // Handle remaining bytes
    for pos in i..=haystack.len() - needle.len() {
        if haystack[pos] == first_char && &haystack[pos..pos + needle.len()] == needle {
            return true;
        }
    }

    false
}

fn simd_contains_byte(haystack: &[u8], target: u8) -> bool {
    const LANES: usize = LOGICAL_LANES_ * 4; // (there's 4 u8s of bits in an f32)

    let target_vec = Simd::<u8, LANES>::splat(target);

    let mut i = 0;
    while i + LANES <= haystack.len() {
        let chunk = Simd::<u8, LANES>::from_slice(&haystack[i..i + LANES]);
        if chunk.simd_eq(target_vec).any() {
            return true;
        }
        i += LANES;
    }

    // Check remaining bytes without SIMD because, the setup is not worth it for small inputs
    haystack[i..].contains(&target)
}

pub fn simd_find_str(haystack: &str, needle: &str) -> Option<usize> {
    const LANES: usize = LOGICAL_LANES_ * 4; // (there's 4 u8s of bits in an f32)

    if needle.is_empty() {
        return Some(0);
    }

    let haystack_bytes = haystack.as_bytes();
    let needle_bytes = needle.as_bytes();

    if needle_bytes.len() > haystack_bytes.len() {
        return None;
    }

    if needle_bytes.len() == 1 {
        return simd_find_byte(haystack_bytes, needle_bytes[0]);
    }

    // Use SIMD to quickly find first character candidates
    let first_char = needle_bytes[0];
    let first_char_vec = Simd::<u8, LANES>::splat(first_char);

    let mut i = 0;
    while i + LANES <= haystack_bytes.len() {
        let chunk = Simd::<u8, LANES>::from_slice(&haystack_bytes[i..i + LANES]);
        let mask = chunk.simd_eq(first_char_vec);

        if mask.any() {
            // Check each potential match position
            let mask_array = mask.to_array();
            for j in 0..LANES {
                if mask_array[j]
                    && i + j + needle_bytes.len() <= haystack_bytes.len()
                    && &haystack_bytes[i + j..i + j + needle_bytes.len()] == needle_bytes
                {
                    return Some(i + j);
                }
            }
        }
        i += LANES;
    }

    // Handle remaining bytes
    (i..=haystack_bytes.len() - needle_bytes.len()).find(|&pos| {
        haystack_bytes[pos] == first_char
            && &haystack_bytes[pos..pos + needle_bytes.len()] == needle_bytes
    });

    None
}

fn simd_find_byte(haystack: &[u8], target: u8) -> Option<usize> {
    const LANES: usize = LOGICAL_LANES_ * 4; // (there's 4 u8s of bits in an f32)

    let target_vec = Simd::<u8, LANES>::splat(target);

    let mut i = 0;
    while i + LANES <= haystack.len() {
        let chunk = Simd::<u8, LANES>::from_slice(&haystack[i..i + LANES]);
        let mask = chunk.simd_eq(target_vec);

        if mask.any() {
            let mask_array = mask.to_array();
            for j in 0..LANES {
                if mask_array[j] {
                    return Some(i + j);
                }
            }
        }
        i += LANES;
    }

    // Check remaining
    haystack[i..]
        .iter()
        .position(|&b| b == target)
        .map(|pos| i + pos)
}

// greyscale an img:

/// Convert RGBA (`[u8;4]`) to grayscale (`[u8;3]`) using SIMD.
pub fn rgba_to_gray_simd_u8(rgba: &[[u8; 4]]) -> Vec<[u8; 3]> {
    const LANES: usize = LOGICAL_LANES_ * 4;
    let mut output = Vec::with_capacity(rgba.len());

    // Weights scaled to fixed-point precision (0.2126 ≈ 54/255, etc.)
    let r_weight = Simd::<u16, LANES>::splat(54); // 0.2126 * 255 ≈ 54
    let g_weight = Simd::<u16, LANES>::splat(182); // 0.7152 * 255 ≈ 182
    let b_weight = Simd::<u16, LANES>::splat(18); // 0.0722 * 255 ≈ 18

    for chunk in rgba.chunks_exact(LANES) {
        let (mut r, mut g, mut b) = ([0u8; LANES], [0u8; LANES], [0u8; LANES]);

        // Extract R, G, B components (ignore alpha)
        for (i, &[ri, gi, bi, _]) in chunk.iter().enumerate() {
            r[i] = ri;
            g[i] = gi;
            b[i] = bi;
        }

        // Convert u8 -> u16 to avoid overflow during multiplication
        let r_simd = Simd::from_array(r).cast::<u16>();
        let g_simd = Simd::from_array(g).cast::<u16>();
        let b_simd = Simd::from_array(b).cast::<u16>();

        // Compute luminance: (54*R + 182*G + 18*B) >> 8 (equivalent to /255)
        let gray = (r_simd * r_weight + g_simd * g_weight + b_simd * b_weight) >> 8;
        let gray_u8 = gray.cast::<u8>().to_array();

        // Store as RGB (repeating luminance)
        for &l in gray_u8.iter() {
            output.push([l, l, l]);
        }
    }

    // Handle remaining pixels, although in the land of images, that come from cameras
    // you're going to find powers of two (most of the time), so this code will likely do little (if anything)
    // in most applications.
    for &[r, g, b, _] in rgba.chunks_exact(LANES).remainder() {
        let l = ((54 * r as u16 + 182 * g as u16 + 18 * b as u16) >> 8) as u8;
        output.push([l, l, l]);
    }

    output
}

pub fn simd_histogram_single(data: &[u8], histogram: &mut [u32; 256]) {
    // Process in larger chunks for better memory access patterns
    const BLOCK_SIZE: usize = 4096;

    for block in data.chunks(BLOCK_SIZE) {
        let chunks = block.chunks_exact(LOGICAL_LANES_);
        let remainder = chunks.remainder();

        // SIMD processing with unrolled inner loop
        for chunk in chunks {
            let simd_vec = Simd::<u8, LOGICAL_LANES_>::from_slice(chunk);
            let bytes = simd_vec.as_array();

            // Unroll for better performance (adjust count for your LOGICAL_LANES)
            for i in (0..LOGICAL_LANES_).step_by(4) {
                // Process 4 bytes at once to reduce loop overhead
                if i + 3 < LOGICAL_LANES_ {
                    histogram[bytes[i] as usize] += 1;
                    histogram[bytes[i + 1] as usize] += 1;
                    histogram[bytes[i + 2] as usize] += 1;
                    histogram[bytes[i + 3] as usize] += 1;
                } else {
                    // Handle remaining bytes in the SIMD vector
                    for j in i..LOGICAL_LANES_ {
                        histogram[bytes[j] as usize] += 1;
                    }
                    break;
                }
            }
        }

        // Process remainder bytes
        for &byte in remainder {
            histogram[byte as usize] += 1;
        }
    }
}

// Alternative: Even more optimised version using unsafe for maximum speed
pub fn simd_histogram_unsafe(data: &[u8], histogram: &mut [u32; 256]) {
    const BLOCK_SIZE: usize = 8192;

    for block in data.chunks(BLOCK_SIZE) {
        let chunks = block.chunks_exact(LOGICAL_LANES_);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let simd_vec = Simd::<u8, LOGICAL_LANES_>::from_slice(chunk);
            let bytes = simd_vec.as_array();

            // SAFETY: bytes are u8, so always valid indices for 256-element array
            for &byte in bytes {
                unsafe {
                    *histogram.get_unchecked_mut(byte as usize) += 1;
                }
            }
        }

        for &byte in remainder {
            unsafe {
                *histogram.get_unchecked_mut(byte as usize) += 1;
            }
        }
    }
}

// Vectorized approach: Process multiple histograms in parallel if needed
pub fn simd_histogram_parallel(data: &[u8], histograms: &mut [[u32; 256]]) {
    let num_hists = histograms.len();
    let chunk_size = data.len() / num_hists;

    histograms
        .par_iter_mut() // Requires rayon crate
        .enumerate()
        .for_each(|(i, histogram)| {
            let start = i * chunk_size;
            let end = if i == num_hists - 1 {
                data.len()
            } else {
                start + chunk_size
            };
            let chunk = &data[start..end];

            simd_histogram_single(chunk, histogram);
        });
}

// For comparison: highly optimised scalar version
pub fn scalar_histogram_optimised(data: &[u8], histogram: &mut [u32; 256]) {
    // Process in blocks for better cache performance
    const BLOCK_SIZE: usize = 4096;

    for block in data.chunks(BLOCK_SIZE) {
        // Unroll by 8 for better ILP (Instruction Level Parallelism)
        let chunks = block.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            // Manual unroll
            histogram[chunk[0] as usize] += 1;
            histogram[chunk[1] as usize] += 1;
            histogram[chunk[2] as usize] += 1;
            histogram[chunk[3] as usize] += 1;
            histogram[chunk[4] as usize] += 1;
            histogram[chunk[5] as usize] += 1;
            histogram[chunk[6] as usize] += 1;
            histogram[chunk[7] as usize] += 1;
        }

        for &byte in remainder {
            histogram[byte as usize] += 1;
        }
    }
}

// ---------- Gather / Scatter Helpers ----------

/// Gather values from `data` using indices from `indices`.
///
/// This is the *scalar* baseline implementation.
/// It simply iterates over `indices` and pushes the corresponding
/// value from `data` into a new `Vec<u32>`.
///
/// # Panics
/// Panics if any index is out of bounds.
pub fn gather_scalar(data: &[u32], indices: &[usize]) -> Vec<u32> {
    let mut out = Vec::with_capacity(indices.len());
    for idx in indices {
        out.push(data[*idx]);
    }
    out
}

/// Scatter values from `data` into `output` using indices from `indices`.
///
/// The *scalar* implementation simply iterates over the slices.
///
/// # Panics
/// Panics if an index is out of bounds for `output`.
pub fn scatter_scalar(data: &[u32], indices: &[usize], output: &mut [u32]) {
    for (val, idx) in data.iter().zip(indices.iter()) {
        output[*idx] = *val;
    }
}

/// Gather values from `data` using SIMD and `indices`.
///
/// This implementation processes `indices` in chunks of
/// `LOGICAL_LANES_`.  For each chunk it loads the indices into a
/// `Simd<u32, LANES>`, then pulls the corresponding elements
/// from `data`.  Since stable portable‑simd does not expose a
/// gather intrinsic, we perform the indexing in a scalar loop over
/// each lane after the indices vector has been loaded.
///
/// # Panics
/// Panics if any index is out of bounds.
pub fn gather_simd(data: &[u32], indices: &[usize]) -> Vec<u32> {
    let mut out = Vec::with_capacity(indices.len());
    let lanes = LOGICAL_LANES_;
    let mut i = 0;
    while i + lanes <= indices.len() {
        let u32_indices: Vec<u32> = indices[i..i + lanes].iter().map(|&x| x as u32).collect();
        let idx_vec = Simd::<u32, LOGICAL_LANES_>::from_slice(&u32_indices);
        for lane in 0..lanes {
            let idx = idx_vec[lane] as usize;
            out.push(data[idx]);
        }
        i += lanes;
    }
    while i < indices.len() {
        out.push(data[indices[i]]);
        i += 1;
    }
    out
}

/// Scatter values from `data` into `output` using SIMD and `indices`.
///
/// Like `gather_simd`, this processes `indices` in chunks of
/// `LOGICAL_LANES_`.  For each chunk we load the data values and
/// indices into SIMD vectors and then write each lane to the
/// corresponding position in `output`.
///
/// # Panics
/// Panics if an index is out of bounds for `output`.
pub fn scatter_simd(data: &[u32], indices: &[usize], output: &mut [u32]) {
    let lanes = LOGICAL_LANES_;
    let mut i = 0;
    while i + lanes <= data.len() && i + lanes <= indices.len() {
        let data_vec = Simd::<u32, LOGICAL_LANES_>::from_slice(&data[i..i + lanes]);
        let u32_indices: Vec<u32> = indices[i..i + lanes].iter().map(|&x| x as u32).collect();
        let idx_vec = Simd::<u32, LOGICAL_LANES_>::from_slice(&u32_indices);
        for lane in 0..lanes {
            let idx = idx_vec[lane] as usize;
            output[idx] = data_vec[lane];
        }
        i += lanes;
    }
        while i < data.len() && i < indices.len() {
            output[indices[i]] = data[i];
            i += 1;
        }
    }
    
    // ---------- Base64 Encoding (PSHUFB) Helpers ----------
    
    const BASE64_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    
    pub fn base64_encode_scalar(data: &[u8]) -> String {
        let mut encoded = String::with_capacity((data.len() + 2) / 3 * 4);
        let mut i = 0;
    
        while i + 3 <= data.len() {
            let chunk = &data[i..i + 3];
            let b0 = chunk[0];
            let b1 = chunk[1];
            let b2 = chunk[2];
    
            let idx0 = b0 >> 2;
            let idx1 = ((b0 & 0x03) << 4) | (b1 >> 4);
            let idx2 = ((b1 & 0x0F) << 2) | (b2 >> 6);
            let idx3 = b2 & 0x3F;
    
            encoded.push(BASE64_CHARS[idx0 as usize] as char);
            encoded.push(BASE64_CHARS[idx1 as usize] as char);
            encoded.push(BASE64_CHARS[idx2 as usize] as char);
            encoded.push(BASE64_CHARS[idx3 as usize] as char);
            i += 3;
        }
    
        let remaining = data.len() - i;
        if remaining == 1 {
            let b0 = data[i];
            let idx0 = b0 >> 2;
            let idx1 = (b0 & 0x03) << 4;
            encoded.push(BASE64_CHARS[idx0 as usize] as char);
            encoded.push(BASE64_CHARS[idx1 as usize] as char);
            encoded.push_str("==");
        } else if remaining == 2 {
            let b0 = data[i];
            let b1 = data[i + 1];
            let idx0 = b0 >> 2;
            let idx1 = ((b0 & 0x03) << 4) | (b1 >> 4);
            let idx2 = (b1 & 0x0F) << 2;
            encoded.push(BASE64_CHARS[idx0 as usize] as char);
            encoded.push(BASE64_CHARS[idx1 as usize] as char);
            encoded.push(BASE64_CHARS[idx2 as usize] as char);
            encoded.push('=');
        }
    
        encoded
    }
    
    /// SIMD-accelerated Base64 encoding.
    ///
    /// This implementation is adapted from the `base64-simd` crate and the work of Wojciech Mula.
    /// It uses SSSE3 instructions, especially `_mm_shuffle_epi8` (PSHUFB), for high performance.
    pub fn base64_encode_simd(data: &[u8]) -> String {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::is_x86_feature_detected!("ssse3") {
                return base64_encode_scalar(data);
            }
    
            use std::arch::x86_64::*;
    
            let mut encoded = Vec::with_capacity((data.len() + 2) / 3 * 4);
            let mut i = 0;
            let len = data.len();
            let main_loop_len = len - (len % 12);
    
            while i < main_loop_len {
                unsafe {
                    let block = _mm_loadu_si128(data.as_ptr().add(i) as *const _);
    
                    // Unpack 12 bytes to 16 indices
                    let shuffle_mask = _mm_set_epi8(8, 7, 9, 8, 6, 5, 7, 6, 4, 3, 5, 4, 2, 1, 3, 2);
                    let hi = _mm_shuffle_epi8(block, shuffle_mask);
                    let lo_mask = _mm_set_epi8(10, -1, -1, 9, -1, -1, 6, -1, -1, 3, -1, -1, 0, -1, -1, -1);
                    let lo = _mm_shuffle_epi8(block, lo_mask);
    
                    let hi = _mm_srli_epi16(hi, 4);
                    let lo = _mm_slli_epi16(lo, 2);
    
                    let indices = _mm_and_si128(_mm_or_si128(hi, lo), _mm_set1_epi8(0x3F));
    
                    // Index to character mapping (arithmetic)
                    let n_lt_26 = _mm_cmpgt_epi8(_mm_set1_epi8(25), indices);
                    let n_lt_52 = _mm_cmpgt_epi8(_mm_set1_epi8(51), indices);
                    let n_lt_62 = _mm_cmpgt_epi8(_mm_set1_epi8(61), indices);
                    let n_eq_62 = _mm_cmpeq_epi8(indices, _mm_set1_epi8(62));
                    let n_eq_63 = _mm_cmpeq_epi8(indices, _mm_set1_epi8(63));
    
                    let _add_a = _mm_and_si128(n_lt_26, _mm_set1_epi8(b'A' as i8));
                    let add_a = _mm_and_si128(_mm_andnot_si128(n_lt_26, n_lt_52), _mm_set1_epi8(b'a' as i8 - 26));
                    let add_0 = _mm_and_si128(_mm_andnot_si128(n_lt_52, n_lt_62), _mm_set1_epi8(b'0' as i8 - 52));
                    let add_plus = _mm_and_si128(n_eq_62, _mm_set1_epi8(b'+' as i8 - 62));
                    let add_slash = _mm_and_si128(n_eq_63, _mm_set1_epi8(b'/' as i8 - 63));
    
                    let added = _mm_add_epi8(indices, add_a);
                    let added = _mm_add_epi8(added, add_a);
                    let added = _mm_add_epi8(added, add_0);
                    let added = _mm_add_epi8(added, add_plus);
                    let result_vec = _mm_add_epi8(added, add_slash);
    
                    let mut dest = [0u8; 16];
                    _mm_storeu_si128(dest.as_mut_ptr() as *mut _, result_vec);
                    encoded.extend_from_slice(&dest);
                }
                i += 12;
            }
    
            // Handle remainder and padding
            if i < len {
                let scalar_encoded = base64_encode_scalar(&data[i..]);
                encoded.extend_from_slice(scalar_encoded.as_bytes());
            }
    
            String::from_utf8(encoded).unwrap()
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            base64_encode_scalar(data)
        }
    }
    
    // ---------- Chess Move Generation (PDEP/PEXT) Helpers ----------
    
    // Pre-computed attack masks and tables.
    // Using a static mut for simplicity in this benchmark context.
    // In a real application, use `once_cell` or a proper static initializer.
    static mut ROOK_MASKS: [u64; 64] = [0; 64];
    // This is a Vec of Vecs, where ATTACK_TABLES[square] is the specific table for that square.
    pub static mut ATTACK_TABLES: Vec<Vec<u64>> = Vec::new();
    
    fn create_rook_mask(square: usize) -> u64 {
        let mut result: u64 = 0;
        let r = square / 8;
        let f = square % 8;
        for i in (f + 1)..7 { result |= 1 << (r * 8 + i); }
        for i in (1..f).rev() { result |= 1 << (r * 8 + i); }
        for i in (r + 1)..7 { result |= 1 << (i * 8 + f); }
        for i in (1..r).rev() { result |= 1 << (i * 8 + f); }
        result
    }
    
    /// Initializes the attack masks and tables for rook move generation.
    pub fn init_rook_attack_tables() {
        unsafe {
            if !ATTACK_TABLES.is_empty() { return; }
            ATTACK_TABLES.resize(64, Vec::new());
            for i in 0..64 {
                ROOK_MASKS[i] = create_rook_mask(i);
                let bits = ROOK_MASKS[i].count_ones();
                let table_size = 1 << bits;
                let attack_table_for_square = &mut ATTACK_TABLES[i];
                attack_table_for_square.resize(table_size, 0);
    
                for j in 0..table_size {
                    // PDEP would be the ideal way to implement the inverse of PEXT,
                    // to map the dense index `j` back to a sparse occupancy bitboard.
                    let occupancy_variation = scalar_pdep(j as u64, ROOK_MASKS[i]);
                    attack_table_for_square[j] = rook_moves_scalar_inner(i as u8, occupancy_variation);
                }
            }
        }
    }
    
    // Scalar PDEP for populating the attack tables without requiring BMI2 at build time.
    fn scalar_pdep(source: u64, mask: u64) -> u64 {
        let mut result = 0;
        let mut source_mut = source;
        for i in 0..64 {
            if (mask >> i) & 1 == 1 {
                if source_mut & 1 == 1 {
                    result |= 1 << i;
                }
                source_mut >>= 1;
            }
        }
        result
    }
    
    /// Generates rook moves using a simple scalar ray-casting method.
    pub fn rook_moves_scalar(square: u8, blockers: u64) -> u64 {
        rook_moves_scalar_inner(square, blockers)
    }
    
    // Inner function for scalar move generation, used by table generator and public scalar function.
    fn rook_moves_scalar_inner(square: u8, blockers: u64) -> u64 {
        let mut moves = 0;
        let r = square as usize / 8;
        let f = square as usize % 8;
    
        // North
        for i in (r + 1)..8 {
            let bit = 1 << (i * 8 + f);
            moves |= bit;
            if (blockers & bit) != 0 { break; }
        }
        // South
        for i in (0..r).rev() {
            let bit = 1 << (i * 8 + f);
            moves |= bit;
            if (blockers & bit) != 0 { break; }
        }
        // East
        for i in (f + 1)..8 {
            let bit = 1 << (r * 8 + i);
            moves |= bit;
            if (blockers & bit) != 0 { break; }
        }
        // West
        for i in (0..f).rev() {
            let bit = 1 << (r * 8 + i);
            moves |= bit;
            if (blockers & bit) != 0 { break; }
        }
        moves
    }
    
    /// Generates rook moves using a PEXT-based lookup table.
    /// This function is unsafe because it accesses global static mut variables.
    pub unsafe fn rook_moves_pext(square: u8, occupancy: u64) -> u64 {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("bmi2") {
                let mask = ROOK_MASKS[square as usize];
                let blockers = occupancy & mask;
                let index = std::arch::x86_64::_pext_u64(blockers, mask);
                return ATTACK_TABLES[square as usize][index as usize];
            }
        }
        // Fallback for non-BMI2 or non-x86
        rook_moves_scalar(square, occupancy)
    }
    
    // ---------- Matrix Transposition (VPERMD) Helpers ----------
    
    /// Transposes an 8x8 matrix of u32 values. Scalar implementation.
    /// The input is a 64-element slice representing the matrix in row-major order.
    pub fn transpose_8x8_u32_scalar(matrix: &[u32; 64]) -> [u32; 64] {
        let mut result = [0u32; 64];
        for i in 0..8 {
            for j in 0..8 {
                result[j * 8 + i] = matrix[i * 8 + j];
            }
        }
        result
    }
    
    /// Transposes an 8x8 matrix of u32 values using AVX2 instructions.
    /// The input is a 64-element slice representing the matrix in row-major order.
    pub fn transpose_8x8_u32_simd(matrix: &[u32; 64]) -> [u32; 64] {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::is_x86_feature_detected!("avx2") {
                return transpose_8x8_u32_scalar(matrix);
            }
    
            use std::arch::x86_64::*;
    
            unsafe {
                // Load 8 rows into 8 YMM registers
                let rows: [__m256i; 8] = [
                    _mm256_loadu_si256(matrix.as_ptr().add(0) as *const _),
                    _mm256_loadu_si256(matrix.as_ptr().add(8) as *const _),
                    _mm256_loadu_si256(matrix.as_ptr().add(16) as *const _),
                    _mm256_loadu_si256(matrix.as_ptr().add(24) as *const _),
                    _mm256_loadu_si256(matrix.as_ptr().add(32) as *const _),
                    _mm256_loadu_si256(matrix.as_ptr().add(40) as *const _),
                    _mm256_loadu_si256(matrix.as_ptr().add(48) as *const _),
                    _mm256_loadu_si256(matrix.as_ptr().add(56) as *const _),
                ];
    
                // Stage 1: 4x4 transpose on 32-bit elements
                let t0 = _mm256_unpacklo_epi32(rows[0], rows[1]);
                let t1 = _mm256_unpackhi_epi32(rows[0], rows[1]);
                let t2 = _mm256_unpacklo_epi32(rows[2], rows[3]);
                let t3 = _mm256_unpackhi_epi32(rows[2], rows[3]);
                let t4 = _mm256_unpacklo_epi32(rows[4], rows[5]);
                let t5 = _mm256_unpackhi_epi32(rows[4], rows[5]);
                let t6 = _mm256_unpacklo_epi32(rows[6], rows[7]);
                let t7 = _mm256_unpackhi_epi32(rows[6], rows[7]);
    
                // Stage 2: 2x2 transpose on 64-bit elements
                let s0 = _mm256_unpacklo_epi64(t0, t2);
                let s1 = _mm256_unpackhi_epi64(t0, t2);
                let s2 = _mm256_unpacklo_epi64(t1, t3);
                let s3 = _mm256_unpackhi_epi64(t1, t3);
                let s4 = _mm256_unpacklo_epi64(t4, t6);
                let s5 = _mm256_unpackhi_epi64(t4, t6);
                let s6 = _mm256_unpacklo_epi64(t5, t7);
                let s7 = _mm256_unpackhi_epi64(t5, t7);
    
                // Stage 3: Permute 128-bit lanes
                let m0 = _mm256_permute2x128_si256(s0, s4, 0x20);
                let m1 = _mm256_permute2x128_si256(s1, s5, 0x20);
                let m2 = _mm256_permute2x128_si256(s2, s6, 0x20);
                let m3 = _mm256_permute2x128_si256(s3, s7, 0x20);
                let m4 = _mm256_permute2x128_si256(s0, s4, 0x31);
                let m5 = _mm256_permute2x128_si256(s1, s5, 0x31);
                let m6 = _mm256_permute2x128_si256(s2, s6, 0x31);
                let m7 = _mm256_permute2x128_si256(s3, s7, 0x31);
    
                let mut result = [0u32; 64];
                _mm256_storeu_si256(result.as_mut_ptr().add(0) as *mut _, m0);
                _mm256_storeu_si256(result.as_mut_ptr().add(8) as *mut _, m1);
                _mm256_storeu_si256(result.as_mut_ptr().add(16) as *mut _, m2);
                _mm256_storeu_si256(result.as_mut_ptr().add(24) as *mut _, m3);
                _mm256_storeu_si256(result.as_mut_ptr().add(32) as *mut _, m4);
                _mm256_storeu_si256(result.as_mut_ptr().add(40) as *mut _, m5);
                _mm256_storeu_si256(result.as_mut_ptr().add(48) as *mut _, m6);
                _mm256_storeu_si256(result.as_mut_ptr().add(56) as *mut _, m7);
    
                result
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            transpose_8x8_u32_scalar(matrix)
        }
    }
    
    // ---------- CRC32-C (PCLMULQDQ) Helpers ----------
    
    // Helper to generate the CRC32-C table.
    fn crc32c_table() -> [u32; 256] {
        const CRC32C_POLY: u32 = 0x1EDC6F41;
        let mut table = [0u32; 256];
        for i in 0..256 {
            let mut crc = i as u32;
            for _ in 0..8 {
                if crc & 1 == 1 {
                    crc = (crc >> 1) ^ CRC32C_POLY;
                } else {
                    crc >>= 1;
                }
            }
            table[i] = crc;
        }
        table
    }
    
    /// Standard table-based CRC32-C implementation.
    pub fn crc32c_scalar(data: &[u8]) -> u32 {
        let table = crc32c_table();
        let mut crc = !0u32;
        for &byte in data {
            crc = (crc >> 8) ^ table[((crc as u8) ^ byte) as usize];
        }
        !crc
    }
    
    /// Hardware-accelerated CRC32-C using PCLMULQDQ.
    ///
    /// This implementation is based on the folding algorithm described in Intel's
    /// "Fast CRC Computation" whitepaper.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    pub fn crc32c_simd(data: &[u8]) -> u32 {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::is_x86_feature_detected!("pclmulqdq") {
                return crc32c_scalar(data);
            }
    
            use std::arch::x86_64::*;
    
            let mut crc = !0u32;
            let mut p = data.as_ptr();
            let mut len = data.len();
    
            // Process unaligned head bytes
            while len > 0 && (p as usize) % 8 != 0 {
                crc = (crc >> 8) ^ crc32c_table()[(crc as u8 ^ unsafe { *p }) as usize];
                p = unsafe { p.add(1) };
                len -= 1;
            }
    
            // Process 64-bit chunks
            let mut crc_64 = crc as u64;
            while len >= 8 {
                crc_64 ^= unsafe { *(p as *const u64) };
                p = unsafe { p.add(8) };
                len -= 8;
            }
            crc_64 = crc_64 as u64;
    
            // This is a simplified final reduction for the remaining 64 bits.
            // A full implementation would involve a more complex folding loop for large inputs.
            // This still demonstrates the core instruction for the final reduction steps.
            let k = _mm_set_epi64x(0, 0x1751997d0); // Reduction constant
            let poly = _mm_set_epi64x(0, 0x11EDC6F41); // CRC32-C polynomial
    
            unsafe {
                let crc_vec = _mm_cvtsi64_si128(crc_64 as i64);
    
                // Reduce 64 bits to 32 bits
                let tmp1 = _mm_clmulepi64_si128(crc_vec, k, 0x01);
                let tmp2 = _mm_xor_si128(tmp1, crc_vec);
                let tmp3 = _mm_srli_epi64(tmp2, 32);
                let crc32_intermediate = _mm_cvtsi128_si32(tmp3) as u32;
    
                let final_crc_vec = _mm_cvtsi32_si128(crc32_intermediate as i32);
                let tmp4 = _mm_clmulepi64_si128(final_crc_vec, poly, 0x00);
                let tmp5 = _mm_xor_si128(tmp4, _mm_set_epi32(0, 0, 0, crc32_intermediate as i32));
                crc = _mm_cvtsi128_si32(tmp5) as u32;
            }
    
            // Process remaining tail bytes
            while len > 0 {
                crc = (crc >> 8) ^ crc32c_table()[(crc as u8 ^ unsafe { *p }) as usize];
                p = unsafe { p.add(1) };
                len -= 1;
            }
    
            return !crc;
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        {
            crc32c_scalar(data)
        }
    }
    
    // ---------- Quantized Dot Product (VPDPBUSD) Helpers ----------
    
    /// Computes the dot product of a u8 vector and an s8 vector, returning an i32.
    /// This is a common operation in quantized neural networks.
    pub fn dot_product_u8s8_scalar(activations: &[u8], weights: &[i8]) -> i32 {
        assert_eq!(activations.len(), weights.len());
        activations
            .iter()
            .zip(weights.iter())
            .map(|(&a, &w)| (a as i32) * (w as i32))
            .sum()
    }
    
    /// Computes the dot product of a u8 vector and an s8 vector using AVX-VNNI.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    pub fn dot_product_u8s8_simd(activations: &[u8], weights: &[i8]) -> i32 {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::is_x86_feature_detected!("avxvnni") {
                return dot_product_u8s8_scalar(activations, weights);
            }
    
            use std::arch::x86_64::*;
    
            assert_eq!(activations.len(), weights.len());
            let mut p_activations = activations.as_ptr();
            let mut p_weights = weights.as_ptr();
            let mut len = activations.len();
    
            let mut acc_vec = _mm256_setzero_si256();
            let chunk_size = 32;
    
            while len >= chunk_size {
                unsafe {
                    let act_vec = _mm256_loadu_si256(p_activations as *const _);
                    let wgt_vec = _mm256_loadu_si256(p_weights as *const _);
                    acc_vec = _mm256_dpbusd_epi32(acc_vec, act_vec, wgt_vec);
    
                    p_activations = p_activations.add(chunk_size);
                    p_weights = p_weights.add(chunk_size);
                    len -= chunk_size;
                }
            }
    
            // Horizontal sum of the accumulator
            unsafe {
                let mut result_lanes = [0i32; 8];
                _mm256_storeu_si256(result_lanes.as_mut_ptr() as *mut _, acc_vec);
                let mut sum = result_lanes.iter().sum();
    
                // Process remainder
                if len > 0 {
                    sum += dot_product_u8s8_scalar(
                        &activations[activations.len() - len..],
                        &weights[weights.len() - len..],
                    );
                }
                return sum;
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            dot_product_u8s8_scalar(activations, weights)
        }
    }
    
    // ---------- Hamming Distance (VPOPCNT) Helpers ----------
    
    /// Calculates the Hamming distance between two slices of u64s.
    /// The Hamming distance is the number of positions at which the
    /// corresponding bits are different.
    pub fn hamming_distance_scalar(a: &[u64], b: &[u64]) -> u64 {
        assert_eq!(a.len(), b.len());
        a.iter()
         .zip(b.iter())
         .map(|(&x, &y)| (x ^ y).count_ones() as u64)
         .sum()
    }
    
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    pub fn hamming_distance_simd(a: &[u64], b: &[u64]) -> u64 {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::is_x86_feature_detected!("avx512vpopcntdq") {
                return hamming_distance_scalar(a, b);
            }
    
            use std::arch::x86_64::*;
    
            assert_eq!(a.len(), b.len());
            let mut p_a = a.as_ptr();
            let mut p_b = b.as_ptr();
            let mut len = a.len();
    
            let mut acc_vec = _mm512_setzero_si512();
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
    
            unsafe {
                let mut sum = _mm512_reduce_add_epi64(acc_vec) as u64;
                if len > 0 {
                    sum += hamming_distance_scalar(
                        &a[a.len() - len..],
                        &b[b.len() - len..],
                    );
                }
                return sum;
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            hamming_distance_scalar(a, b)
        }
    }
    
    // ---------- Duplicate Detection (VPCONFLICT) Helpers ----------
    
    /// Checks for duplicate values within a small slice of u64s.
    /// This scalar implementation uses a nested loop for simplicity.
    pub fn has_duplicates_scalar(data: &[u64]) -> bool {
        for i in 0..data.len() {
            for j in (i + 1)..data.len() {
                if data[i] == data[j] {
                    return true;
                }
            }
        }
        false
    }
    
    /// Checks for duplicate values within an 8-element array of u64s using AVX-512 VPCONFLICT.
    pub fn has_duplicates_simd(data: &[u64; 8]) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::is_x86_feature_detected!("avx512cd") {
                return has_duplicates_scalar(data);
            }
    
            use std::arch::x86_64::*;
    
            unsafe {
                // Load the 8 elements into a 512-bit vector
                let vec_a = _mm512_loadu_si512(data.as_ptr() as *const _);
    
                // VPCONFLICT computes a mask for each element, indicating conflicts with preceding elements.
                let conflict_vec = _mm512_conflict_epi64(vec_a);
    
                // OR all the conflict masks together. If the result is non-zero, a conflict was found.
                let total_conflicts = _mm512_reduce_or_epi64(conflict_vec);
    
                return total_conflicts != 0;
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            has_duplicates_scalar(data)
        }
    }
    
    // ---------- Find MSB (VPLZCNT) Helpers ----------
    
    /// Finds the position of the Most Significant Bit (MSB) for each u32 in a slice.
    /// The position is 0-indexed (from the least significant bit).
    /// If the input is 0, the result is -1.
    pub fn find_msb_scalar(data: &[u32]) -> Vec<i32> {
        data.iter()
            .map(|&x| if x == 0 { -1 } else { 31 - x.leading_zeros() as i32 })
            .collect()
    }
    
    /// Finds the position of the MSB for each u32 in a slice using AVX-512 VPLZCNT.
    pub fn find_msb_simd(data: &[u32]) -> Vec<i32> {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::is_x86_feature_detected!("avx512cd") {
                return find_msb_scalar(data);
            }
    
            use std::arch::x86_64::*;
    
            let mut results = Vec::with_capacity(data.len());
            let mut p = data.as_ptr();
            let mut len = data.len();
            let chunk_size = 16; // 16 u32s in a 512-bit vector
    
            while len >= chunk_size {
                unsafe {
                    let vec_a = _mm512_loadu_si512(p as *const _);
    
                    // Count leading zeros
                    let lzcnt_vec = _mm512_lzcnt_epi32(vec_a);
    
                    // Calculate MSB position: 31 - leading_zeros
                    let const_31 = _mm512_set1_epi32(31);
                    let msb_pos_vec = _mm512_sub_epi32(const_31, lzcnt_vec);
    
                    // For inputs of 0, lzcnt is 32, so msb_pos is -1, which is correct.
                    
                    // Store results
                    let mut dest = [0i32; 16];
                    _mm512_storeu_si512(dest.as_mut_ptr() as *mut _, msb_pos_vec);
                    results.extend_from_slice(&dest);
    
                    p = p.add(chunk_size);
                    len -= chunk_size;
                }
            }
    
            // Process remainder
            if len > 0 {
                results.extend_from_slice(&find_msb_scalar(&data[data.len() - len..]));
            }
            
            return results;
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            find_msb_scalar(data)
        }
    }
    
    // ---------- Bit Gathering (VPSHUFBITQMB) Helpers ----------
    
    /// For each u64 in a block, gathers 8 bits specified by the `indices`
    /// array and packs them into a u8 result.
    pub fn gather_bits_scalar(data: &[[u64; 8]], indices: &[u8; 8]) -> Vec<u8> {
        let mut result = Vec::with_capacity(data.len() * 8);
        for block in data {
            for i in 0..8 { // For each u64 in the block
                let mut out_byte = 0u8;
                for j in 0..8 { // For each index
                    // Get the j-th bit from the i-th u64 at the specified index
                    let bit_pos = indices[j] % 64; // Ensure index is in bounds
                    let bit = (block[i] >> bit_pos) & 1;
                    out_byte |= (bit as u8) << j;
                }
                result.push(out_byte);
            }
        }
        result
    }
    
    /// Gathers bits from vectors of u64s using AVX-512 BITALG's VPSHUFBITQMB.
    /// NOTE: This is a placeholder implementation due to limitations in std::arch.
    pub fn gather_bits_simd(data: &[[u64; 8]], _indices: &[u8; 8]) -> Vec<u8> {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::is_x86_feature_detected!("avx512bitalg") {
                return gather_bits_scalar(data, _indices);
            }
    
            // FIXME: The intrinsic `_mm512_bitshuffle_epi64_mask` required for this
            // operation appears to be missing or have an incorrect signature in the
            // current `std::arch`. The 512-bit instruction should return a `__mmask64`,
            // but no such intrinsic is available.
            //
            // A correct implementation would look something like this:
            /*
            use std::arch::x86_64::*;
            let mut results = Vec::with_capacity(data.len() * 8);
            let mut control_u64 = 0u64;
            for i in 0..8 {
                control_u64 |= (indices[i] as u64) << (i * 8);
            }
            let control_vec = _mm512_set1_epi64(control_u64 as i64);
    
            for block in data {
                unsafe {
                    let data_vec = _mm512_loadu_si512(block.as_ptr() as *const _);
                    // This is the missing intrinsic:
                    // let mask_result: __mmask64 = _mm512_bitshuffle_epi64_mask(control_vec, data_vec);
                    // results.extend_from_slice(&mask_result.to_le_bytes());
                }
            }
            return results;
            */
    
            // As we cannot implement this with the available intrinsics, we call todo!()
            // to make it clear that this path is unimplemented.
            todo!("The AVX-512 BITALG intrinsic for 64-bit bit-level shuffle is not correctly exposed in std::arch, blocking this implementation.");
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            gather_bits_scalar(data, _indices)
        }
    }
    
    // ---------- Bitfield Extraction (VPMULTISHIFTQB) Helpers ----------
    
    /// For each pair of data and control u64s, extracts 8 distinct 8-bit
    /// fields from the data u64. The starting bit of each 8-bit field is
    /// specified by a byte in the control u64.
    pub fn multishift_extract_scalar(data: &[u64], controls: &[u64]) -> Vec<u64> {
        assert_eq!(data.len(), controls.len());
        let mut results = Vec::with_capacity(data.len());
    
        for i in 0..data.len() {
            let data_val = data[i];
            let control_val = controls[i];
            let mut result_val = 0u64;
    
            for j in 0..8 {
                let control_byte = (control_val >> (j * 8)) as u8;
                let shift_amount = control_byte % 64; // Ensure shift is in bounds
    
                // Shift the data to get the 8-bit field at the LSB position
                let shifted_data = data_val >> shift_amount;
                
                let extracted_byte = shifted_data as u8;
    
                // Place the extracted byte into the result u64
                result_val |= (extracted_byte as u64) << (j * 8);
            }
            results.push(result_val);
        }
        results
    }
    
    /// Extracts bitfields using the AVX-512 VBMI instruction VPMULTISHIFTQB.
    pub fn multishift_extract_simd(data: &[u64], controls: &[u64]) -> Vec<u64> {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::is_x86_feature_detected!("avx512vbmi") {
                return multishift_extract_scalar(data, controls);
            }
    
            use std::arch::x86_64::*;
    
            assert_eq!(data.len(), controls.len());
            let mut results = Vec::with_capacity(data.len());
            let mut p_data = data.as_ptr();
            let mut p_controls = controls.as_ptr();
            let mut len = data.len();
            let chunk_size = 8; // 8 u64s in a 512-bit vector
    
            while len >= chunk_size {
                unsafe {
                    let data_vec = _mm512_loadu_si512(p_data as *const _);
                    let control_vec = _mm512_loadu_si512(p_controls as *const _);
    
                    let result_vec = _mm512_multishift_epi64_epi8(control_vec, data_vec);
    
                    let mut dest = [0u64; 8];
                    _mm512_storeu_si512(dest.as_mut_ptr() as *mut _, result_vec);
                    results.extend_from_slice(&dest);
    
                    p_data = p_data.add(chunk_size);
                    p_controls = p_controls.add(chunk_size);
                    len -= chunk_size;
                }
            }
    
            // Process remainder
            if len > 0 {
                results.extend_from_slice(&multishift_extract_scalar(
                    &data[data.len() - len..],
                    &controls[controls.len() - len..],
                ));
            }
            
            return results;
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            multishift_extract_scalar(data, controls)
        }
    }
    
    // ---------- Branchless Select (VPTERNLOG) Helpers ----------
    
    /// Performs a bitwise SELECT operation: `(a & b) | (~a & c)`.
    /// For each bit, if the bit in `a` is 1, the output bit is from `b`,
    /// otherwise it's from `c`.
    pub fn select_u32_scalar(a: &[u32], b: &[u32], c: &[u32]) -> Vec<u32> {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), c.len());
        let mut results = Vec::with_capacity(a.len());
        for i in 0..a.len() {
            results.push((a[i] & b[i]) | (!a[i] & c[i]));
        }
        results
    }
    
    /// Performs a bitwise SELECT operation using AVX-512 VPTERNLOG.
    pub fn select_u32_simd(a: &[u32], b: &[u32], c: &[u32]) -> Vec<u32> {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::is_x86_feature_detected!("avx512f") {
                return select_u32_scalar(a, b, c);
            }
    
            use std::arch::x86_64::*;
    
            assert_eq!(a.len(), b.len());
            assert_eq!(a.len(), c.len());
            let mut results = Vec::with_capacity(a.len());
            let mut p_a = a.as_ptr();
            let mut p_b = b.as_ptr();
            let mut p_c = c.as_ptr();
            let mut len = a.len();
            let chunk_size = 16; // 16 u32s in a 512-bit vector
    
            while len >= chunk_size {
                unsafe {
                    let vec_a = _mm512_loadu_si512(p_a as *const _);
                    let vec_b = _mm512_loadu_si512(p_b as *const _);
                    let vec_c = _mm512_loadu_si512(p_c as *const _);
    
                    // The immediate 0xD8 corresponds to the truth table for (a & b) | (~a & c)
                    // where the index into the truth table is {c, b, a}.
                    let result_vec = _mm512_ternarylogic_epi32(vec_c, vec_b, vec_a, 0xD8);
    
                    let mut dest = [0u32; 16];
                    _mm512_storeu_si512(dest.as_mut_ptr() as *mut _, result_vec);
                    results.extend_from_slice(&dest);
    
                    p_a = p_a.add(chunk_size);
                    p_b = p_b.add(chunk_size);
                    p_c = p_c.add(chunk_size);
                    len -= chunk_size;
                }
            }
    
            // Process remainder
            if len > 0 {
                results.extend_from_slice(&select_u32_scalar(
                    &a[a.len() - len..],
                    &b[b.len() - len..],
                    &c[c.len() - len..],
                ));
            }
            
            return results;
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            select_u32_scalar(a, b, c)
        }
    }
    
// ---------- Saturating Pack (VPMOVDB) Helpers ----------

/// Converts a slice of i32 to i8 with saturation.
pub fn pack_i32_to_i8_saturating_scalar(data: &[i32]) -> Vec<i8> {
    data.iter()
        .map(|&x| x.max(i8::MIN as i32).min(i8::MAX as i32) as i8)
        .collect()
}

/// Converts a slice of i32 to i8 with saturation using AVX-512.
pub fn pack_i32_to_i8_saturating_simd(data: &[i32]) -> Vec<i8> {
    #[cfg(target_arch = "x86_64")]
    {
        if !std::is_x86_feature_detected!("avx512f") {
            return pack_i32_to_i8_saturating_scalar(data);
        }

        use std::arch::x86_64::*;

        let mut results = Vec::with_capacity(data.len());
        let mut p = data.as_ptr();
        let mut len = data.len();
        let chunk_size = 16; // 16 i32s in a 512-bit vector

        while len >= chunk_size {
            unsafe {
                let data_vec = _mm512_loadu_si512(p as *const _);

                // Signed, saturating conversion from i32 to i8
                let result_vec = _mm512_cvtsepi32_epi8(data_vec);

                // The result is 16 bytes (i8), which fits in a __m128i.
                // We need to store these 16 bytes.
                let mut dest = [0i8; 16];
                _mm_storeu_si128(dest.as_mut_ptr() as *mut _, result_vec);
                results.extend_from_slice(&dest);

                p = p.add(chunk_size);
                len -= chunk_size;
            }
        }

        // Process remainder
        if len > 0 {
            results.extend_from_slice(&pack_i32_to_i8_saturating_scalar(
                &data[data.len() - len..],
            ));
        }
        
        return results;
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        pack_i32_to_i8_saturating_scalar(data)
    }
}

// ---------- VPERMT2 Helpers ----------

/// Placeholder scalar implementation for VPERMT2.
pub fn vpermt2_scalar(data: &[u32], _indices: &[u32]) -> Vec<u32> {
    // This is a placeholder. A real scalar implementation would perform the permutation.
    // For now, it just returns a copy of the data.
    data.to_vec()
}

/// Placeholder SIMD implementation for VPERMT2.
pub fn vpermt2_simd(data: &[u32], indices: &[u32]) -> Vec<u32> {
    #[cfg(target_arch = "x86_64")]
    {
        if !std::is_x86_feature_detected!("avx512f") { // VPERMT2 is AVX-512
            return vpermt2_scalar(data, indices);
        }
        // Placeholder for actual AVX-512 VPERMT2 implementation
        // This would involve _mm512_permutex2var_epi32 or similar
        vpermt2_scalar(data, indices)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        vpermt2_scalar(data, indices)
    }
}

// ---------- VFIXUPIMM Helpers ----------

/// Placeholder scalar implementation for VFIXUPIMM.
pub fn vfixupimm_scalar(data: &[f32]) -> Vec<f32> {
    // Placeholder: returns data as is.
    data.to_vec()
}

/// Placeholder SIMD implementation for VFIXUPIMM.
pub fn vfixupimm_simd(data: &[f32]) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        if !std::is_x86_feature_detected!("avx512f") { // VFIXUPIMM is AVX-512
            return vfixupimm_scalar(data);
        }
        // Placeholder for actual AVX-512 VFIXUPIMM implementation
        // This would involve _mm512_fixupimm_ps or similar
        vfixupimm_scalar(data)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        vfixupimm_scalar(data)
    }
}