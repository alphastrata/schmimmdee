/// This benchmark demonstrates a "killer application" of the VPMOVDB family of
/// instructions: high-speed, saturating data conversion.
///
/// ## Saturating Narrowing
/// When converting data from a wider integer type to a narrower one (e.g., i32 to i8),
/// there is a risk of the value being out of range for the narrower type. A standard
/// cast (`as i8`) will truncate the bits, causing the value to "wrap around",
/// which is usually undesirable.
///
/// A "saturating" conversion, by contrast, clamps the value to the minimum or
/// maximum value of the destination type. This is the correct behavior for many
/// applications, especially in graphics and audio processing, where wrapping would
/// cause severe visual or audible artifacts.
///
/// ## The VPMOVDB Advantage
/// The `VPMOVSDB` instruction (`_mm512_cvtsepi32_epi8`) is designed to perform this
/// signed, saturating conversion from 16 32-bit integers to 16 8-bit integers
/// across a 512-bit vector in a single instruction. This avoids the need for
/// slow, branchy scalar code that would have to check the bounds of every element
/// individually.

use schmimmdee::{pack_i32_to_i8_saturating_scalar, pack_i32_to_i8_saturating_simd, format_ns, format_number};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    if !std::is_x86_feature_detected!("avx512f") {
        eprintln!("Warning: AVX-512F not detected. SIMD implementation will fall back to scalar.");
    }

    const VECTOR_SIZE: usize = 16 * 1024 * 1024; // 16M i32s

    println!("Generating a vector of {} i32 elements...", format_number(VECTOR_SIZE));
    let mut rng = StdRng::seed_from_u64(42);
    // Generate numbers both inside and outside the i8 range
    let data: Vec<i32> = (0..VECTOR_SIZE).map(|_| rng.gen_range(-500..500)).collect();
    println!("Done.\n");

    let trials = 10;

    println!("{:^80}", " Saturating Pack i32->i8 Benchmark (VPMOVSDB) ");
    println!(
        "| {:>12} | {:>15} | {:>15} | {:>10} | {:>10} |",
        "Method", "Scalar", "SIMD", "Speedup", "Valid"
    );
    println!(
        "|{:-^14}|{:-^17}|{:-^17}|{:-^12}|{:-^12}|",
        "", "", "", "", ""
    );

    // --- Warmup ---
    black_box(pack_i32_to_i8_saturating_scalar(&data));
    black_box(unsafe { pack_i32_to_i8_saturating_simd(&data) });

    // --- Scalar Benchmark ---
    let scalar_time: u128 = (0..trials)
        .map(|_| {
            let start = Instant::now();
            black_box(pack_i32_to_i8_saturating_scalar(&data));
            start.elapsed().as_nanos()
        })
        .sum();

    // --- SIMD Benchmark ---
    let simd_time: u128 = (0..trials)
        .map(|_| {
            let start = Instant::now();
            black_box(unsafe { pack_i32_to_i8_saturating_simd(&data) });
            start.elapsed().as_nanos()
        })
        .sum();

    let avg_scalar = scalar_time as f64 / trials as f64;
    let avg_simd = simd_time as f64 / trials as f64;
    let speedup = avg_scalar / avg_simd;

    // Verification
    let scalar_res = pack_i32_to_i8_saturating_scalar(&data);
    let simd_res = unsafe { pack_i32_to_i8_saturating_simd(&data) };
    let valid = scalar_res == simd_res;
    assert!(valid, "Results do not match!");

    println!(
        "| {:>12} | {:>15} | {:>15} | {:>9.2}x | {:>9} |",
        "pack_i32_i8",
        format_ns(avg_scalar),
        format_ns(avg_simd),
        speedup,
        if valid { "✓" } else { "✗" }
    );
    println!("{:^80}", "");
    println!("\nBenchmark complete!");
}
