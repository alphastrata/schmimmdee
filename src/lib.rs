#![feature(portable_simd)]
use std::{
    f32,
    simd::{
        Simd,
        cmp::SimdPartialEq,
        num::{SimdFloat, SimdUint},
    },
};

/// Update this to reflect the width of YOUR system's registers.
const LOGICAL_LANES: usize = 4; // Auto-detected for x86_64-pc-windows-msvc

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
    let mut min_vec = Simd::<f32, LOGICAL_LANES>::splat(f32::MAX);
    let mut max_vec = Simd::<f32, LOGICAL_LANES>::splat(f32::MIN);

    data.chunks_exact(LOGICAL_LANES).for_each(|chunk| {
        let values = Simd::<f32, LOGICAL_LANES>::from_slice(chunk);
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
    let first_char_vec = Simd::<u8, LOGICAL_LANES>::splat(first_char);

    let mut i = 0;
    while i + LOGICAL_LANES <= haystack.len() {
        let chunk = Simd::<u8, LOGICAL_LANES>::from_slice(&haystack[i..i + LOGICAL_LANES]);
        let mask = chunk.simd_eq(first_char_vec);

        if mask.any() {
            // Check each potential match position
            let mask_array = mask.to_array();
            for j in 0..LOGICAL_LANES {
                if mask_array[j]
                    && i + j + needle.len() <= haystack.len()
                    && &haystack[i + j..i + j + needle.len()] == needle
                {
                    return true;
                }
            }
        }
        i += LOGICAL_LANES;
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
    let target_vec = Simd::<u8, LOGICAL_LANES>::splat(target);

    let mut i = 0;
    while i + LOGICAL_LANES <= haystack.len() {
        let chunk = Simd::<u8, LOGICAL_LANES>::from_slice(&haystack[i..i + LOGICAL_LANES]);
        if chunk.simd_eq(target_vec).any() {
            return true;
        }
        i += LOGICAL_LANES;
    }

    // Check remaining bytes without SIMD because, the setup is not worth it for small inputs
    haystack[i..].contains(&target)
}

pub fn simd_find_str(haystack: &str, needle: &str) -> Option<usize> {
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
    let first_char_vec = Simd::<u8, LOGICAL_LANES>::splat(first_char);

    let mut i = 0;
    while i + LOGICAL_LANES <= haystack_bytes.len() {
        let chunk = Simd::<u8, LOGICAL_LANES>::from_slice(&haystack_bytes[i..i + LOGICAL_LANES]);
        let mask = chunk.simd_eq(first_char_vec);

        if mask.any() {
            // Check each potential match position
            let mask_array = mask.to_array();
            for j in 0..LOGICAL_LANES {
                if mask_array[j]
                    && i + j + needle_bytes.len() <= haystack_bytes.len()
                    && &haystack_bytes[i + j..i + j + needle_bytes.len()] == needle_bytes
                {
                    return Some(i + j);
                }
            }
        }
        i += LOGICAL_LANES;
    }

    // Handle remaining bytes
    (i..=haystack_bytes.len() - needle_bytes.len()).find(|&pos| {
        haystack_bytes[pos] == first_char
            && &haystack_bytes[pos..pos + needle_bytes.len()] == needle_bytes
    });

    None
}

fn simd_find_byte(haystack: &[u8], target: u8) -> Option<usize> {
    let target_vec = Simd::<u8, LOGICAL_LANES>::splat(target);

    let mut i = 0;
    while i + LOGICAL_LANES <= haystack.len() {
        let chunk = Simd::<u8, LOGICAL_LANES>::from_slice(&haystack[i..i + LOGICAL_LANES]);
        let mask = chunk.simd_eq(target_vec);

        if mask.any() {
            let mask_array = mask.to_array();
            for j in 0..LOGICAL_LANES {
                if mask_array[j] {
                    return Some(i + j);
                }
            }
        }
        i += LOGICAL_LANES;
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
    const LANES: usize = LOGICAL_LANES; // Process 16 pixels at once (AVX2-friendly)
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
    let chunks = data.chunks_exact(LOGICAL_LANES);
    let remainder = chunks.remainder();

    // Process SIMD chunks
    for chunk in chunks {
        let simd_vec = Simd::<u8, LOGICAL_LANES>::from_slice(chunk);

        // Unroll the loop for better performance
        let array = simd_vec.as_array();
        for &byte in array {
            // SAFETY: byte is u8, so always valid index for [u32; 256]
            unsafe {
                *histogram.get_unchecked_mut(byte as usize) += 1;
            }
        }
    }

    // Process remainder
    for &byte in remainder {
        histogram[byte as usize] += 1;
    }
}

// Alternative: More aggressive SIMD optimization
pub fn simd_histogram_optimized(data: &[u8], histogram: &mut [u32; 256]) {
    // Process in larger chunks for better cache efficiency
    const CHUNK_SIZE: usize = 8192;

    for chunk in data.chunks(CHUNK_SIZE) {
        let simd_chunks = chunk.chunks_exact(LOGICAL_LANES);
        let remainder = simd_chunks.remainder();

        // SIMD processing
        for simd_chunk in simd_chunks {
            let simd_vec = Simd::<u8, LOGICAL_LANES>::from_slice(simd_chunk);
            let array = simd_vec.as_array();

            // Manual unroll for 32 lanes (adjust for your SIMD_LANES)
            histogram[array[0] as usize] += 1;
            histogram[array[1] as usize] += 1;
            histogram[array[2] as usize] += 1;
            histogram[array[3] as usize] += 1;
            // ... continue for all lanes or use a loop
            for i in 0..LOGICAL_LANES {
                histogram[array[i] as usize] += 1;
            }
        }

        // Process remainder
        for &byte in remainder {
            histogram[byte as usize] += 1;
        }
    }
}