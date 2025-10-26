/// This benchmark demonstrates a "killer application" of the VPLZCNT instruction:
/// finding the position of the most significant bit (MSB) for a large number of
/// integers simultaneously.
///
/// ## Find Most Significant Bit (MSB)
/// Finding the MSB, or calculating the integer base-2 logarithm, is a primitive
/// operation in many algorithms. It's used in:
/// -   **Data Compression**: To encode the length of runs or the size of numbers.
/// -   **Numerical Algorithms**: For normalization and scaling of values.
/// -   **Data Structures**: In search trees and other structures that depend on
///     the binary representation of keys.
///
/// The position of the MSB is directly related to the number of leading zeros
/// in the binary representation of a number: `msb_pos = (bits - 1) - leading_zeros`.
///
/// ## The VPLZCNT Advantage
/// The `VPLZCNT` instruction (`_mm512_lzcnt_epi32`), part of the AVX-512CD
/// instruction set, computes the number of leading zeros for all 16 32-bit
/// integers in a 512-bit vector at once.
///
/// This benchmark's SIMD implementation:
/// 1.  Loads a vector of 16 `u32` integers.
/// 2.  Uses `VPLZCNT` to get a vector of leading zero counts.
/// 3.  Performs a vectorized subtraction from 31 to get a vector of MSB positions.
///
/// This provides a substantial performance improvement over scalar code, which
/// must process each integer individually.

use schmimmdee::{find_msb_scalar, find_msb_simd, format_ns, format_number};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    if !std::is_x86_feature_detected!("avx512cd") {
        eprintln!("Warning: AVX-512CD not detected. SIMD implementation will fall back to scalar.");
    }

    const VECTOR_SIZE: usize = 16 * 1024 * 1024; // 16M u32s

    println!("Generating a vector of {} u32 elements...", format_number(VECTOR_SIZE));
    let mut rng = StdRng::seed_from_u64(42);
    // Generate numbers with a wide range of MSB positions
    let data: Vec<u32> = (0..VECTOR_SIZE).map(|i| {
        if i % 100 == 0 { 0 } // Include some zeros
        else { rng.gen::<u32>() >> (i % 32) }
    }).collect();
    println!("Done.\n");

    let trials = 10;

    println!("{:-^80}", " Find MSB Benchmark (VPLZCNT) ");
    println!(
        "| {:>12} | {:>15} | {:>15} | {:>10} | {:>10} |",
        "Method", "Scalar", "SIMD", "Speedup", "Valid"
    );
    println!(
        "|{:-^14}|{:-^17}|{:-^17}|{:-^12}|{:-^12}|",
        "", "", "", "", ""
    );

    // --- Warmup ---
    black_box(find_msb_scalar(&data));
    black_box(unsafe { find_msb_simd(&data) });

    // --- Scalar Benchmark ---
    let scalar_time: u128 = (0..trials)
        .map(|_| {
            let start = Instant::now();
            black_box(find_msb_scalar(&data));
            start.elapsed().as_nanos()
        })
        .sum();

    // --- SIMD Benchmark ---
    let simd_time: u128 = (0..trials)
        .map(|_| {
            let start = Instant::now();
            black_box(unsafe { find_msb_simd(&data) });
            start.elapsed().as_nanos()
        })
        .sum();

    let avg_scalar = scalar_time as f64 / trials as f64;
    let avg_simd = simd_time as f64 / trials as f64;
    let speedup = avg_scalar / avg_simd;

    // Verification
    let scalar_res = find_msb_scalar(&data);
    let simd_res = unsafe { find_msb_simd(&data) };
    let valid = scalar_res == simd_res;
    if !valid {
        eprintln!("Validation FAILED!");
        // Find first mismatch
        for i in 0..scalar_res.len() {
            if scalar_res[i] != simd_res[i] {
                eprintln!("Mismatch at index {}: data={}, scalar={}, simd={}", i, data[i], scalar_res[i], simd_res[i]);
                break;
            }
        }
    }
    assert!(valid, "Results do not match!");

    println!(
        "| {:>12} | {:>15} | {:>15} | {:>9.2}x | {:>9} |",
        "find_msb",
        format_ns(avg_scalar),
        format_ns(avg_simd),
        speedup,
        if valid { "✓" } else { "✗" }
    );
    println!("{:-^80}", "");
    println!("\nBenchmark complete!");
}
