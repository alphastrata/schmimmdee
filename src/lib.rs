#![feature(portable_simd)]
use std::{
    f32,
    simd::{Simd, cmp::SimdPartialEq, num::SimdFloat},
};

/// Update this to reflect the width of YOUR system's registers.
const LOGICAL_LANES: usize = 16;

/// prettly-formant nanos from our std::instant timming.
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

// minmax

#[unsafe(no_mangle)] // so if you want to peek @ the asembly it's easier to find your function..
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

    data.chunks_exact(8)
        .remainder()
        .into_iter()
        .for_each(|&value| {
            min = min.min(value);
            max = max.max(value);
        });

    (min, max)
}

pub fn find_min_max_scalar(data: &[f32]) -> (f32, f32) {
    let mut min = f32::MAX;
    let mut max = f32::MIN;

    data.into_iter().for_each(|&value| {
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
                if mask_array[j] && i + j + needle.len() <= haystack.len() {
                    if &haystack[i + j..i + j + needle.len()] == needle {
                        return true;
                    }
                }
            }
        }
        i += LOGICAL_LANES;
    }

    // Handle remaining bytes
    for pos in i..=haystack.len() - needle.len() {
        if haystack[pos] == first_char {
            if &haystack[pos..pos + needle.len()] == needle {
                return true;
            }
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
                if mask_array[j] && i + j + needle_bytes.len() <= haystack_bytes.len() {
                    if &haystack_bytes[i + j..i + j + needle_bytes.len()] == needle_bytes {
                        return Some(i + j);
                    }
                }
            }
        }
        i += LOGICAL_LANES;
    }

    // Handle remaining bytes
    for pos in i..=haystack_bytes.len() - needle_bytes.len() {
        if haystack_bytes[pos] == first_char {
            if &haystack_bytes[pos..pos + needle_bytes.len()] == needle_bytes {
                return Some(pos);
            }
        }
    }

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
