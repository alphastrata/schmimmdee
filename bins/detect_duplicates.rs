/// This benchmark demonstrates a "killer application" of the VPCONFLICT instruction:
/// efficiently detecting duplicate values within a vector.
///
/// ## Conflict Detection
/// Finding duplicate items in a set is a common problem in computer science,
/// appearing in database unique constraints, hash table implementations, and
/// parallel algorithm synchronization (e.g., histogram updates).
///
/// A naive scalar approach requires comparing every element with every other
/// element, an O(n^2) operation for a vector of length n.
///
/// ## The VPCONFLICT Advantage
/// The `VPCONFLICT` instruction (`_mm512_conflict_epi64`), part of the AVX-512CD
/// (Conflict Detection) instruction set, is designed to solve this problem in
/// hardware.
///
/// In a single instruction, it takes a vector of elements and, for each element, 
/// produces a bitmask identifying all preceding elements in the vector that have
/// the same value.
///
/// This benchmark's SIMD implementation:
/// 1.  Loads a chunk of 8 `u64` integers into a 512-bit vector.
/// 2.  Uses `VPCONFLICT` to generate the conflict masks.
/// 3.  Performs a vectorized OR reduction (`_mm512_reduce_or_epi64`) on the result.
/// 4.  If the final result is anything other than zero, it means a duplicate was
///     found within the chunk.
///
/// This allows checking for duplicates among 8 `u64` values with just a few
/// instructions, offering a significant speedup.

use schmimmdee::{has_duplicates_scalar, has_duplicates_simd, format_ns, format_number};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    if !std::is_x86_feature_detected!("avx512cd") {
        eprintln!("Warning: AVX-512CD not detected. SIMD implementation will fall back to scalar.");
    }

    const NUM_CHUNKS: usize = 1024 * 1024; // 1M chunks of 8 u64s
    const CHUNK_SIZE: usize = 8;

    println!("Generating {} chunks of {} u64s...", format_number(NUM_CHUNKS), CHUNK_SIZE);
    let mut rng = StdRng::seed_from_u64(42);
    let chunks: Vec<[u64; CHUNK_SIZE]> = (0..NUM_CHUNKS).map(|i| {
        let mut chunk = [0u64; CHUNK_SIZE];
        // Every 4th chunk has a duplicate to ensure both code paths are tested
        if i % 4 == 0 {
            for i in 0..CHUNK_SIZE { chunk[i] = rng.gen(); }
            let pos1 = rng.gen_range(0..CHUNK_SIZE);
            let mut pos2 = rng.gen_range(0..CHUNK_SIZE);
            while pos1 == pos2 {
                pos2 = rng.gen_range(0..CHUNK_SIZE);
            }
            chunk[pos2] = chunk[pos1];
        } else {
            // Fill with unique random numbers (highly likely to be unique)
            for i in 0..CHUNK_SIZE { chunk[i] = rng.gen(); }
        }
        chunk
    }).collect();
    println!("Done.\n");

    let trials = 10;

    println!("{:-^80}", " Duplicate Detection Benchmark (VPCONFLICT) ");
    println!(
        "| {:>12} | {:>15} | {:>15} | {:>10} | {:>10} |",
        "Method", "Scalar", "SIMD", "Speedup", "Valid"
    );
    println!(
        "|{:-^14}|{:-^17}|{:-^17}|{:-^12}|{:-^12}|",
        "", "", "", "", ""
    );

    // --- Warmup ---
    black_box(has_duplicates_scalar(&chunks[0]));
    black_box(unsafe { has_duplicates_simd(&chunks[0]) });

    // --- Scalar Benchmark ---
    let scalar_time: u128 = (0..trials)
        .map(|_| {
            let start = Instant::now();
            for chunk in &chunks {
                black_box(has_duplicates_scalar(chunk));
            }
            start.elapsed().as_nanos()
        })
        .sum();

    // --- SIMD Benchmark ---
    let simd_time: u128 = (0..trials)
        .map(|_| {
            let start = Instant::now();
            for chunk in &chunks {
                black_box(unsafe { has_duplicates_simd(chunk) });
            }
            start.elapsed().as_nanos()
        })
        .sum();

    let avg_scalar = scalar_time as f64 / trials as f64;
    let avg_simd = simd_time as f64 / trials as f64;
    let speedup = avg_scalar / avg_simd;

    // Verification
    let mut valid = true;
    for chunk in chunks.iter().take(10000) { // Verify first 10k chunks
        let scalar_res = has_duplicates_scalar(chunk);
        let simd_res = unsafe { has_duplicates_simd(chunk) };
        if scalar_res != simd_res {
            valid = false;
            eprintln!("Validation FAILED for chunk: {:?}", chunk);
            eprintln!("Scalar={}, SIMD={}", scalar_res, simd_res);
            break;
        }
    }
    assert!(valid, "Results do not match!");

    println!(
        "| {:>12} | {:>15} | {:>15} | {:>9.2}x | {:>9} |",
        "has_dupes",
        format_ns(avg_scalar),
        format_ns(avg_simd),
        speedup,
        if valid { "✓" } else { "✗" }
    );
    println!("{:-^80}", "");
    println!("\nBenchmark complete!");
}
