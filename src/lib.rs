#![feature(portable_simd)]
use std::{
    f32,
    simd::{Simd, num::SimdFloat},
};

const LOGICAL_LANES: usize = 16;

// SIMD version
#[unsafe(no_mangle)]
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

// Scalar version
pub fn find_min_max_scalar(data: &[f32]) -> (f32, f32) {
    let mut min = f32::MAX;
    let mut max = f32::MIN;

    data.into_iter().for_each(|&value| {
        min = min.min(value);
        max = max.max(value);
    });

    (min, max)
}

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
