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

use schmimmdee::{gather_bits_scalar, gather_bits_simd, format_ns, format_number};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    if !std::is_x86_feature_detected!("avx512bitalg") {
        eprintln!("Warning: AVX-512 BITALG not detected. SIMD implementation will fall back to scalar.");
    }

    const NUM_BLOCKS: usize = 1 * 1024 * 1024; // 1M blocks of 8 u64s
    const BLOCK_SIZE: usize = 8;

    println!("Generating {} blocks of {} u64s...", format_number(NUM_BLOCKS), BLOCK_SIZE);
    let mut rng = StdRng::seed_from_u64(42);
    let data: Vec<[u64; BLOCK_SIZE]> = (0..NUM_BLOCKS)
        .map(|_| [0u64; BLOCK_SIZE].map(|_| rng.gen()))
        .collect();
    
    let mut indices = [0u8; 8];
    for i in 0..8 { indices[i] = rng.gen_range(0..64); }
    println!("Gathering bits at indices: {:?}\n", indices);

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
    black_box(gather_bits_scalar(&data, &indices));
    // SIMD version will panic due to todo!(), so we can't warm it up directly
    // black_box(gather_bits_simd(&data, &indices));

    // --- Scalar Benchmark ---
    let scalar_time: u128 = (0..trials)
        .map(|_| {
            let start = Instant::now();
            black_box(gather_bits_scalar(&data, &indices));
            start.elapsed().as_nanos()
        })
        .sum();

    // --- SIMD Benchmark ---
    // We expect this to panic. The purpose is to have the code in place.
    let simd_result = std::panic::catch_unwind(|| {
        gather_bits_simd(&data, &indices)
    });

    let avg_scalar = scalar_time as f64 / trials as f64;
    
    // Verification is not possible as the SIMD version panics.
    let valid = false;

    println!(
        "| {:>12} | {:>15} | {:>15} | {:>9.2}x | {:>9} |",
        "gather_bits",
        format_ns(avg_scalar),
        "(unimplemented)",
        0.0,
        if valid { "✓" } else { "✗" }
    );
    println!("{:-^80}", "");
    match simd_result {
        Ok(_) => println!("Note: SIMD function did not panic as expected."),
        Err(_) => println!("Note: SIMD function panicked as expected due to missing intrinsic."),
    }
    println!("\nBenchmark complete!");
}
