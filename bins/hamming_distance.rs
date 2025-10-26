/// This benchmark demonstrates a "killer application" of the VPOPCNT instruction:
/// calculating the Hamming distance between two large bitsets.
///
/// ## Hamming Distance
/// The Hamming distance measures the difference between two strings of equal
/// length. It is the number of positions at which the corresponding symbols
/// (in this case, bits) are different. It's a fundamental metric used in
/// coding theory, cryptography, and bioinformatics for comparing data.
///
/// The distance between two bit vectors `A` and `B` can be calculated as:
/// `popcount(A XOR B)`
/// where `popcount` counts the number of set (1) bits.
///
/// ## The VPOPCNT Advantage
/// The `VPOPCNT` instruction (`_mm512_popcnt_epi64`) is part of the AVX-512
/// instruction set family. It is designed to perform a population count on all
/// elements of a wide SIMD vector in a single instruction.
///
/// This benchmark's SIMD implementation:
/// 1.  Loads 8 `u64` integers (512 bits) from each input vector.
/// 2.  Performs a vectorized XOR operation.
/// 3.  Uses `VPOPCNT` to count the set bits in all 8 XOR results simultaneously.
/// 4.  Accumulates these counts in a vector register.
///
/// This approach provides a massive speedup over scalar code, which can only
/// process one `u64` at a time.

use schmimmdee::{hamming_distance_scalar, hamming_distance_simd, format_ns, format_number};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    if !std::is_x86_feature_detected!("avx512vpopcntdq") {
        eprintln!("Warning: AVX-512 VPOPCNTDQ not detected. SIMD implementation will fall back to scalar.");
    }

    const VECTOR_SIZE: usize = 16 * 1024 * 1024; // 16M u64s

    println!("Generating two vectors of {} u64 elements...", format_number(VECTOR_SIZE));
    let mut rng = StdRng::seed_from_u64(42);
    let vec_a: Vec<u64> = (0..VECTOR_SIZE).map(|_| rng.gen()).collect();
    let vec_b: Vec<u64> = (0..VECTOR_SIZE).map(|_| rng.gen()).collect();
    println!("Done.\n");

    let trials = 10;

    println!("{:-^80}", " Hamming Distance Benchmark (VPOPCNT) ");
    println!(
        "| {:>12} | {:>15} | {:>15} | {:>10} | {:>10} |",
        "Method", "Scalar", "SIMD", "Speedup", "Valid"
    );
    println!(
        "|{:-^14}|{:-^17}|{:-^17}|{:-^12}|{:-^12}|",
        "", "", "", "", ""
    );

    // --- Warmup ---
    black_box(hamming_distance_scalar(&vec_a, &vec_b));
                black_box(unsafe { hamming_distance_simd(&vec_a, &vec_b) });
    // --- Scalar Benchmark ---
    let scalar_time: u128 = (0..trials)
        .map(|_| {
            let start = Instant::now();
            black_box(hamming_distance_scalar(&vec_a, &vec_b));
            start.elapsed().as_nanos()
        })
        .sum();

    // --- SIMD Benchmark ---
    let simd_time: u128 = (0..trials)
        .map(|_| {
            let start = Instant::now();
    black_box(unsafe { hamming_distance_simd(&vec_a, &vec_b) });
            start.elapsed().as_nanos()
        })
        .sum();

    let avg_scalar = scalar_time as f64 / trials as f64;
    let avg_simd = simd_time as f64 / trials as f64;
    let speedup = avg_scalar / avg_simd;

    // Verification
    let scalar_res = hamming_distance_scalar(&vec_a, &vec_b);
    let simd_res = unsafe { hamming_distance_simd(&vec_a, &vec_b) };
    let valid = scalar_res == simd_res;
    if !valid {
        eprintln!("Validation FAILED: Scalar={}, SIMD={}", scalar_res, simd_res);
    }
    assert!(valid, "Results do not match!");

    println!(
        "| {:>12} | {:>15} | {:>15} | {:>9.2}x | {:>9} |",
        "h_distance",
        format_ns(avg_scalar),
        format_ns(avg_simd),
        speedup,
        if valid { "✓" } else { "✗" }
    );
    println!("{:-^80}", "");
    println!("\nBenchmark complete!");
}
