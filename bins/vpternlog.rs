use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::hint::black_box;
use std::time::Instant;

use schmimmdee::{vpternlog_scalar, vpternlog_simd, format_ns, format_number};

fn main() {
    if !std::is_x86_feature_detected!("avx512f") {
        eprintln!("Warning: AVX-512F not detected. SIMD implementation will fall back to scalar.");
    }

    const VECTOR_SIZE: usize = 16 * 1024 * 1024; // 16M u64s

    println!("Generating three vectors of {} u64 elements...", format_number(VECTOR_SIZE));
    let mut rng = StdRng::seed_from_u64(42);
    let a: Vec<u64> = (0..VECTOR_SIZE).map(|_| rng.gen()).collect();
    let b: Vec<u64> = (0..VECTOR_SIZE).map(|_| rng.gen()).collect();
    let c: Vec<u64> = (0..VECTOR_SIZE).map(|_| rng.gen()).collect();
    let imm8: u8 = rng.gen(); // Random immediate for ternary logic
    println!("Done.\n");

    let trials = 10;

    println!("{:-^80}", " VPTERNLOG Benchmark ");
    println!(
        "| {:>12} | {:>15} | {:>15} | {:>10} | {:>10} |",
        "Method", "Scalar", "SIMD", "Speedup", "Valid"
    );
    println!(
        "|{:-^14}|{:-^17}|{:-^17}|{:-^12}|{:-^12}|",
        "", "", "", "", ""
    );

    // --- Warmup ---
    black_box(vpternlog_scalar(&a, &b, &c, imm8));
    black_box(vpternlog_simd(&a, &b, &c, imm8));

    // --- Scalar Benchmark ---
    let scalar_time: u128 = (0..trials)
        .map(|_| {
            let start = Instant::now();
            black_box(vpternlog_scalar(&a, &b, &c, imm8));
            start.elapsed().as_nanos()
        })
        .sum();

    // --- SIMD Benchmark ---
    let simd_time: u128 = (0..trials)
        .map(|_| {
            let start = Instant::now();
            black_box(vpternlog_simd(&a, &b, &c, imm8));
            start.elapsed().as_nanos()
        })
        .sum();

    let avg_scalar = scalar_time as f64 / trials as f64;
    let avg_simd = simd_time as f64 / trials as f64;
    let speedup = avg_scalar / avg_simd;

    // Verification
    let scalar_res = vpternlog_scalar(&a, &b, &c, imm8);
    let simd_res = vpternlog_simd(&a, &b, &c, imm8);
    let valid = scalar_res == simd_res;
    if !valid {
        eprintln!("Validation FAILED!");
    }
    assert!(valid, "Results do not match!");

    println!(
        "| {:>12} | {:>15} | {:>15} | {:>9.2}x | {:>9} |",
        "vpternlog",
        format_ns(avg_scalar),
        format_ns(avg_simd),
        speedup,
        if valid { "✓" } else { "✗" }
    );
    println!("{:-^80}", "");
    println!("\nBenchmark complete!");
}
