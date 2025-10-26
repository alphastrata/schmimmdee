/// This benchmark demonstrates an application of the AVX-512 BITALG instruction
/// `VPSHUFBITQMB`: gathering arbitrary bits from a vector of integers.
///
/// ## Bit Gathering
/// Extracting specific, non-contiguous bits from a data stream is a common
/// task in many domains, including:
/// -   **Data Decompression**: Reading packed data fields that don't align to
///     byte boundaries.
/// -   **Hardware Emulation**: Accessing packed bit-fields in simulated hardware
///     registers.
/// -   **Cryptography**: Performing bit-level permutations and substitutions.
///
/// ## The VPSHUFBITQMB Advantage
/// The `VPSHUFBITQMB` instruction is a powerful bit-level shuffle. For each
/// 64-bit lane, it can use 8 indices (each 0-63) to select any 8 bits from
/// the corresponding 64-bit data element and pack them into a result byte.
///
/// This benchmark's SIMD implementation is structured to use this instruction
/// to perform this operation across a full 512-bit vector (8 lanes) at once.
///
/// **NOTE:** As of this writing, the required `_mm512_bitshuffle_epi64_mask`
/// intrinsic appears to be missing or incorrectly defined in Rust's `std::arch`.
/// The SIMD function contains a `todo!()` placeholder where the intrinsic call
/// would be. The benchmark will therefore not show a speedup until this is resolved.

use schmimmdee::{vpshufbitqmb_scalar, vpshufbitqmb_simd, format_ns, format_number};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    if !std::is_x86_feature_detected!("avx512vbmi2") {
        eprintln!("Warning: AVX-512 BITALG not detected. SIMD implementation will fall back to scalar.");
    }

    const NUM_BLOCKS: usize = 1024 * 1024; // 1M blocks of 8 u64s
    const BLOCK_SIZE: usize = 8;

    println!("Generating {} blocks of {} u64s...", format_number(NUM_BLOCKS), BLOCK_SIZE);
    let mut rng = StdRng::seed_from_u64(42);
    let data: Vec<u64> = (0..NUM_BLOCKS * BLOCK_SIZE).map(|_| rng.gen()).collect();
    
    let control: Vec<u64> = (0..NUM_BLOCKS * BLOCK_SIZE).map(|_| rng.gen()).collect();
    println!("Done.\n");

    let trials = 10;

    println!("{:-^80}", " Bit Gathering Benchmark (VPSHUFBITQMB) ");
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
    unsafe { black_box(vpshufbitqmb_simd(&data, &control)) };

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
            unsafe { black_box(vpshufbitqmb_simd(&data, &control)) };
            start.elapsed().as_nanos()
        })
        .sum();

    let avg_scalar = scalar_time as f64 / trials as f64;
    let avg_simd = simd_time as f64 / trials as f64;
    let speedup = avg_scalar / avg_simd;

    // Verification
    let scalar_res = vpshufbitqmb_scalar(&data, &control);
    let simd_res = unsafe { vpshufbitqmb_simd(&data, &control) };
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