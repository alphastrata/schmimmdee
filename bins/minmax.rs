use rand::Rng;
use std::{hint::black_box, time::Instant};

use schmimmdee::*;

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
            black_box({
                _ = find_min_max_scalar(&data);
                _ = find_min_max_simd(&data);
            });
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
