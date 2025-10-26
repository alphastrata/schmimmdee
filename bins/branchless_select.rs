/// This benchmark demonstrates a "killer application" of the VPTERNLOG instruction:
/// implementing arbitrary 3-input boolean logic without branches.
///
/// ## Ternary Logic
/// Most CPU bitwise instructions take one or two inputs (e.g., NOT, AND, OR, XOR).
/// The `VPTERNLOG` instruction (`_mm512_ternarylogic_epi32`) is unique in that it
/// can compute *any* 3-input boolean function in a single instruction.
///
/// It works like a hardware lookup table. For each of the 512 bit positions in
/// the three input vectors (a, b, c), it uses the 3 bits as an index into an
/// 8-bit immediate value (the "truth table"). The result bit is the bit from
/// the immediate at that index. This allows for 256 possible functions.
///
/// ## The `SELECT` Operation
/// This benchmark implements the `SELECT` function: `(a & b) | (~a & c)`.
/// This is equivalent to the C ternary expression `a ? b : c`. In scalar code,
/// this can lead to branches, but `VPTERNLOG` can compute it for all 512 bits
/// in parallel with a single instruction and the immediate `0xD8`.

use schmimmdee::{select_u32_scalar, select_u32_simd, format_ns, format_number};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    if !std::is_x86_feature_detected!("avx512f") {
        eprintln!("Warning: AVX-512F not detected. SIMD implementation will fall back to scalar.");
    }

    const VECTOR_SIZE: usize = 16 * 1024 * 1024; // 16M u32s

    println!("Generating three vectors of {} u32 elements...", format_number(VECTOR_SIZE));
    let mut rng = StdRng::seed_from_u64(42);
    let vec_a: Vec<u32> = (0..VECTOR_SIZE).map(|_| rng.gen()).collect();
    let vec_b: Vec<u32> = (0..VECTOR_SIZE).map(|_| rng.gen()).collect();
    let vec_c: Vec<u32> = (0..VECTOR_SIZE).map(|_| rng.gen()).collect();
    println!("Done.\n");

    let trials = 10;

    println!("{:-^80}", " Branchless Select Benchmark (VPTERNLOG) ");
    println!(
        "| {:>12} | {:>15} | {:>15} | {:>10} | {:>10} |",
        "Method", "Scalar", "SIMD", "Speedup", "Valid"
    );
    println!(
        "|{:-^14}|{:-^17}|{:-^17}|{:-^12}|{:-^12}|",
        "", "", "", "", ""
    );

    // --- Warmup ---
    black_box(select_u32_scalar(&vec_a, &vec_b, &vec_c));
    black_box(select_u32_simd(&vec_a, &vec_b, &vec_c));

    // --- Scalar Benchmark ---
    let scalar_time: u128 = (0..trials)
        .map(|_| {
            let start = Instant::now();
            black_box(select_u32_scalar(&vec_a, &vec_b, &vec_c));
            start.elapsed().as_nanos()
        })
        .sum();

    // --- SIMD Benchmark ---
    let simd_time: u128 = (0..trials)
        .map(|_| {
            let start = Instant::now();
            black_box(select_u32_simd(&vec_a, &vec_b, &vec_c));
            start.elapsed().as_nanos()
        })
        .sum();

    let avg_scalar = scalar_time as f64 / trials as f64;
    let avg_simd = simd_time as f64 / trials as f64;
    let speedup = avg_scalar / avg_simd;

    // Verification
    let scalar_res = select_u32_scalar(&vec_a, &vec_b, &vec_c);
    let simd_res = select_u32_simd(&vec_a, &vec_b, &vec_c);
    let valid = scalar_res == simd_res;
    assert!(valid, "Results do not match!");

    println!(
        "| {:>12} | {:>15} | {:>15} | {:>9.2}x | {:>9} |",
        "select",
        format_ns(avg_scalar),
        format_ns(avg_simd),
        speedup,
        if valid { "✓" } else { "✗" }
    );
    println!("{:-^80}", "");
    println!("\nBenchmark complete!");
}
