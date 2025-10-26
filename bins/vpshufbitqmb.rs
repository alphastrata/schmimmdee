use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::hint::black_box;
use std::time::Instant;

use schmimmdee::{vpshufbitqmb_scalar, vpshufbitqmb_simd, format_ns, format_number};

fn main() {
    if !std::is_x86_feature_detected!("avx512vbmi2") {
        eprintln!("Warning: AVX-512VBMI2 not detected. SIMD implementation will fall back to scalar.");
    }

    const VECTOR_SIZE: usize = 16 * 1024 * 1024; // 16M u64s

    println!("Generating data and control vectors of {} u64 elements...", format_number(VECTOR_SIZE));
    let mut rng = StdRng::seed_from_u64(42);
    let data: Vec<u64> = (0..VECTOR_SIZE).map(|_| rng.gen()).collect();
    let control: Vec<u64> = (0..VECTOR_SIZE).map(|_| rng.gen()).collect(); // Placeholder control
    println!("Done.\n");

    let trials = 10;

    println!("{:-^80}", " VPSHUFBITQMB Benchmark ");
    println!(
        "| {:>12} | {:>15} | {:>15} | {:>10} | {:>10} |",
        "Method", "Scalar", "SIMD", "Speedup", "Valid"
    );
    println!(
        "|{:-^14}|{:-^17}|{:-^17}|{:-^12}|{:-^12}|",
        "", "", "", "", ""
    );

    // --- Warmup ---
    black_box(vpshufbitqmb_scalar(&data, &control));
    black_box(vpshufbitqmb_simd(&data, &control));

    // --- Scalar Benchmark ---
    let scalar_time: u128 = (0..trials)
        .map(|_| {
            let start = Instant::now();
            black_box(vpshufbitqmb_scalar(&data, &control));
            start.elapsed().as_nanos()
        })
        .sum();

    // --- SIMD Benchmark ---
    let simd_time: u128 = (0..trials)
        .map(|_| {
            let start = Instant::now();
            black_box(vpshufbitqmb_simd(&data, &control));
            start.elapsed().as_nanos()
        })
        .sum();

    let avg_scalar = scalar_time as f64 / trials as f64;
    let avg_simd = simd_time as f64 / trials as f64;
    let speedup = avg_scalar / avg_simd;

    // Verification
    let scalar_res = vpshufbitqmb_scalar(&data, &control);
    let simd_res = vpshufbitqmb_simd(&data, &control);
    let valid = scalar_res == simd_res;
    if !valid {
        eprintln!("Validation FAILED!");
    }
    assert!(valid, "Results do not match!");

    println!(
        "| {:>12} | {:>15} | {:>15} | {:>9.2}x | {:>9} |",
        "vpshufbitqmb",
        format_ns(avg_scalar),
        format_ns(avg_simd),
        speedup,
        if valid { "✓" } else { "✗" }
    );
    println!("{:-^80}", "");
    println!("\nBenchmark complete!");
}
