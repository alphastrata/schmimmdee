#![feature(portable_simd)]
use rand::Rng;
use std::{
    f32,
    hint::black_box,
    simd::{Simd, num::SimdFloat},
    time::Instant,
};

const LOGICAL_LANES: usize = 16;

// SIMD version
#[unsafe(no_mangle)]
fn find_min_max_simd(data: &[f32]) -> (f32, f32) {
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
fn find_min_max_scalar(data: &[f32]) -> (f32, f32) {
    let mut min = f32::MAX;
    let mut max = f32::MIN;

    data.into_iter().for_each(|&value| {
        min = min.min(value);
        max = max.max(value);
    });

    (min, max)
}

fn format_ns(ns: f64) -> String {
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

fn main() {
    let mut rng = rand::rng();
    let sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000];
    let trials = 100;

    println!("{:-^80}", " Benchmark Results ");
    println!(
        "| {:>12} | {:>15} | {:>15} | {:>10} | {:>10} |",
        "Elements", "Scalar", "SIMD", "Speedup", "Valid"
    );
    println!(
        "|{:-^14}|{:-^17}|{:-^17}|{:-^12}|{:-^12}|",
        "", "", "", "", ""
    );

    sizes.iter().for_each(|&size| {
        // Generate random data
        let data: Vec<f32> = (0..size)
            .map(|_| rng.random_range(-(f32::MIN / 2.0)..f32::MAX))
            .collect();

        // warmup to prevent either from winning the benefits of a hot cache.
        (0..3).for_each(|_| {
            let _ = find_min_max_scalar(&data);
            let _ = find_min_max_simd(&data);
        });

        // Benchmark scalar version
        let scalar_time: u128 = (0..trials)
            .map(|_| {
                let start = Instant::now();
                black_box(find_min_max_scalar(&data));
                start.elapsed().as_nanos()
            })
            .sum();

        // Benchmark SIMD version
        let simd_time: u128 = (0..trials)
            .map(|_| {
                let start = Instant::now();
                black_box(find_min_max_simd(&data));
                start.elapsed().as_nanos()
            })
            .sum();

        // Calculate speedup
        let avg_simd = simd_time as f64 / trials as f64;
        let avg_scalar = scalar_time as f64 / trials as f64;
        let speedup = avg_scalar / avg_simd;

        // Verify results
        let (simd_min, simd_max) = find_min_max_simd(&data);
        let (scalar_min, scalar_max) = find_min_max_scalar(&data);
        let valid = simd_min == scalar_min && simd_max == scalar_max;

        // Print formatted results
        println!(
            "| {:>12} | {:>15} | {:>15} | {:>9.2}x | {:>9} |",
            format!("{:e}", size),
            format_ns(avg_scalar),
            format_ns(avg_simd),
            speedup,
            if valid { "✓" } else { "✗" }
        );
    });
    println!("{:-^80}", "");
}
