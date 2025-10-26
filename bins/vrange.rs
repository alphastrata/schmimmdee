use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::hint::black_box;
use std::time::Instant;

use schmimmdee::{vrange_scalar, vrange_simd, format_ns, format_number};

fn main() {
    if !std::is_x86_feature_detected!("avx512f") {
        eprintln!("Warning: AVX-512F not detected. SIMD implementation will fall back to scalar.");
    }

    const VECTOR_SIZE: usize = 16 * 1024 * 1024; // 16M f32s

    println!("Generating two vectors of {} f32 elements...", format_number(VECTOR_SIZE));
    let mut rng = StdRng::seed_from_u64(42);
    let a: Vec<f32> = (0..VECTOR_SIZE).map(|_| rng.gen_range(-1000.0..1000.0)).collect();
    let b: Vec<f32> = (0..VECTOR_SIZE).map(|_| rng.gen_range(-1000.0..1000.0)).collect();
    let imm8: u8 = rng.gen(); // Random immediate for range operation
    println!("Done.\n");

    let trials = 10;

    println!("{:-^80}", " VRANGEPS/PD Benchmark ");
    println!(
        "| {:>12} | {:>15} | {:>15} | {:>10} | {:>10} |",
        "Method", "Scalar", "SIMD", "Speedup", "Valid"
    );
    println!(
        "|{:-^14}|{:-^17}|{:-^17}|{:-^12}|{:-^12}|",
        "", "", "", "", ""
    );

    // --- Warmup ---
    black_box(vrange_scalar(&a, &b, imm8));
    black_box(vrange_simd(&a, &b, imm8));

    // --- Scalar Benchmark ---
    let scalar_time: u128 = (0..trials)
        .map(|_| {
            let start = Instant::now();
            black_box(vrange_scalar(&a, &b, imm8));
            start.elapsed().as_nanos()
        })
        .sum();

    // --- SIMD Benchmark ---
    let simd_time: u128 = (0..trials)
        .map(|_| {
            let start = Instant::now();
            black_box(vrange_simd(&a, &b, imm8));
            start.elapsed().as_nanos()
        })
        .sum();

    let avg_scalar = scalar_time as f64 / trials as f64;
    let avg_simd = simd_time as f64 / trials as f64;
    let speedup = avg_scalar / avg_simd;

    // Verification
    let scalar_res = vrange_scalar(&a, &b, imm8);
    let simd_res = vrange_simd(&a, &b, imm8);
    let valid = scalar_res == simd_res;
    if !valid {
        eprintln!("Validation FAILED!");
    }
    assert!(valid, "Results do not match!");

    println!(
        "| {:>12} | {:>15} | {:>15} | {:>9.2}x | {:>9} |",
        "vrange",
        format_ns(avg_scalar),
        format_ns(avg_simd),
        speedup,
        if valid { "✓" } else { "✗" }
    );
    println!("{:-^80}", "");
    println!("\nBenchmark complete!");
}
