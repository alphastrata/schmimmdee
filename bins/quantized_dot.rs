/// This benchmark demonstrates a "killer application" of the VPDPBUSD instruction
/// from the AVX-VNNI instruction set: accelerating quantized neural network operations.
///
/// ## Quantization in Neural Networks
/// To improve performance and reduce memory footprint, the weights and activations
/// in a neural network, which are typically 32-bit floating-point numbers, can be
/// "quantized" into 8-bit integers (e.g., u8 for activations, s8 for weights).
///
/// The most common operation in these networks is the dot product, which forms the
/// basis of convolutions and fully-connected layers.
///
/// ## The VPDPBUSD Advantage
/// The `VPDPBUSD` instruction (`_mm256_dpbusd_epi32`) is designed specifically for
/// this workload. In a single instruction, it:
/// 1.  Takes a vector of 32 unsigned 8-bit integers (activations).
/// 2.  Takes a vector of 32 signed 8-bit integers (weights).
/// 3.  Divides them into 8 groups of 4 pairs.
/// 4.  For each group, computes the four products: `(u8 * s8)`.
/// 5.  Sums the four products into a temporary 32-bit integer.
/// 6.  Adds this sum to a 32-bit integer accumulator.
///
/// This allows for an immense throughput increase over scalar code, which must
/// perform these multiplications and accumulations one by one with intermediate
/// promotions to avoid overflow.

use schmimmdee::{dot_product_u8s8_scalar, dot_product_u8s8_simd, format_ns, format_number};
use rand::{Rng, SeedableRng};
use rand::distributions::Standard;
use rand::rngs::StdRng;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    if !std::is_x86_feature_detected!("avxvnni") {
        eprintln!("Warning: AVX-VNNI not detected. SIMD implementation will fall back to scalar.");
    }

    const VECTOR_SIZE: usize = 16 * 1024 * 1024; // 16M elements

    println!("Generating two vectors (u8 and s8) of {} elements...", format_number(VECTOR_SIZE));
    let mut rng = StdRng::seed_from_u64(42);
    let activations: Vec<u8> = (0..VECTOR_SIZE).map(|_| rng.gen()).collect();
    let weights: Vec<i8> = (0..VECTOR_SIZE).map(|_| rng.sample(Standard)).collect();
    println!("Done.\n");

    let trials = 10;

    println!("{:-^80}", " Quantized Dot Product Benchmark (VPDPBUSD) ");
    println!(
        "| {:>12} | {:>15} | {:>15} | {:>10} | {:>10} |",
        "Method", "Scalar", "SIMD", "Speedup", "Valid"
    );
    println!(
        "|{:-^14}|{:-^17}|{:-^17}|{:-^12}|{:-^12}|",
        "", "", "", "", ""
    );

    // --- Warmup ---
    black_box(dot_product_u8s8_scalar(&activations, &weights));
                black_box(unsafe { dot_product_u8s8_simd(&activations, &weights) });
    // --- Scalar Benchmark ---
    let scalar_time: u128 = (0..trials)
        .map(|_| {
            let start = Instant::now();
            black_box(dot_product_u8s8_scalar(&activations, &weights));
            start.elapsed().as_nanos()
        })
        .sum();

    // --- SIMD Benchmark ---
    let simd_time: u128 = (0..trials)
        .map(|_| {
            let start = Instant::now();
            black_box(unsafe { dot_product_u8s8_simd(&activations, &weights) });
            start.elapsed().as_nanos()
        })
        .sum();

    let avg_scalar = scalar_time as f64 / trials as f64;
    let avg_simd = simd_time as f64 / trials as f64;
    let speedup = avg_scalar / avg_simd;

    // Verification
    let scalar_res = dot_product_u8s8_scalar(&activations, &weights);
    let simd_res = unsafe { dot_product_u8s8_simd(&activations, &weights) };
    let valid = scalar_res == simd_res;
    if !valid {
        eprintln!("Validation FAILED: Scalar={}, SIMD={}", scalar_res, simd_res);
    }
    assert!(valid, "Results do not match!");

    println!(
        "| {:>12} | {:>15} | {:>15} | {:>9.2}x | {:>9} |",
        "dot_product",
        format_ns(avg_scalar),
        format_ns(avg_simd),
        speedup,
        if valid { "✓" } else { "✗" }
    );
    println!("{:-^80}", "");
    println!("\nBenchmark complete!");
}
