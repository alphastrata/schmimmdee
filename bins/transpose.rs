/// This benchmark demonstrates a "killer application" of SIMD permutation
/// instructions: high-speed matrix transposition.
///
/// The benchmark transposes a large matrix by breaking it down into 8x8 blocks
/// of 32-bit integers and transposing each block using AVX2 instructions.
///
/// ## The SIMD Algorithm
/// An 8x8 matrix of 32-bit integers fits perfectly into eight 256-bit YMM
/// registers (8 rows * 8 integers/row * 4 bytes/integer = 256 bytes).
///
/// The transposition is performed in three stages using a shuffle network:
/// 1.  **Stage 1**: Transposes 4x4 sub-blocks of 32-bit integers using
///     `_mm256_unpacklo_epi32` and `_mm256_unpackhi_epi32`. This interleaves
///     elements from pairs of rows.
/// 2.  **Stage 2**: Transposes 2x2 sub-blocks of 64-bit elements using
///     `_mm256_unpacklo_epi64` and `_mm256_unpackhi_epi64`.
/// 3.  **Stage 3**: Swaps the 128-bit lanes between registers using
///     `_mm256_permute2x128_si256`. This is a powerful cross-lane permutation
///     that completes the transposition.
///
/// This entire 8x8 block transposition is done without any loops, making it
/// dramatically faster than a scalar element-by-element swap.

use schmimmdee::{format_ns, format_number, transpose_8x8_u32_scalar, transpose_8x8_u32_simd};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    if !std::is_x86_feature_detected!("avx2") {
        eprintln!("Warning: AVX2 not detected. SIMD implementation will fall back to scalar.");
    }

    const NUM_BLOCKS: usize = 65536; // 64k blocks of 8x8
    const BLOCK_ELEMENTS: usize = 64;
    let total_elements = NUM_BLOCKS * BLOCK_ELEMENTS;

    println!("Generating {} 8x8 matrices ({} u32 elements)...", NUM_BLOCKS, format_number(total_elements));
    let mut rng = StdRng::seed_from_u64(42);
    let matrices: Vec<[u32; 64]> = (0..NUM_BLOCKS)
        .map(|_| [0u32; 64].map(|_| rng.gen()))
        .collect();
    println!("Done.\n");

    let trials = 10;

    println!("{:^80}", " 8x8 Matrix Transposition Benchmark (AVX2) ");
    println!(
        "| {:>12} | {:>15} | {:>15} | {:>10} | {:>10} |",
        "Method", "Scalar", "SIMD", "Speedup", "Valid"
    );
    println!(
        "|{:-^14}|{:-^17}|{:-^17}|{:-^12}|{:-^12}|",
        "", "", "", "", ""
    );

    // --- Scalar Benchmark ---
    let scalar_time: u128 = (0..trials)
        .map(|_| {
            let start = Instant::now();
            for block in &matrices {
                black_box(transpose_8x8_u32_scalar(block));
            }
            start.elapsed().as_nanos()
        })
        .sum();

    // --- SIMD Benchmark ---
    let simd_time: u128 = (0..trials)
        .map(|_| {
            let start = Instant::now();
            for block in &matrices {
                black_box(unsafe { transpose_8x8_u32_simd(block) });
            }
            start.elapsed().as_nanos()
        })
        .sum();

    let avg_scalar = scalar_time as f64 / trials as f64;
    let avg_simd = simd_time as f64 / trials as f64;
    let speedup = avg_scalar / avg_simd;

    // Verification
    let scalar_res = transpose_8x8_u32_scalar(&matrices[0]);
    let simd_res = unsafe { transpose_8x8_u32_simd(&matrices[0]) };
    let valid = scalar_res == simd_res;
    assert!(valid, "Results do not match!");

    println!(
        "| {:>12} | {:>15} | {:>15} | {:>9.2}x | {:>9} |",
        "transpose",
        format_ns(avg_scalar),
        format_ns(avg_simd),
        speedup,
        if valid { "✓" } else { "✗" }
    );
    println!("{:^80}", "");
    println!("\nBenchmark complete!");
}
