/// This benchmark tests a SIMD-accelerated Base64 encoding implementation
/// against a standard scalar one.
///
/// The SIMD implementation uses the SSSE3 instruction set, particularly the
/// `_mm_shuffle_epi8` (PSHUFB) instruction, to achieve high performance.
///
/// The algorithm processes 12 bytes of input at a time, producing 16 bytes of
/// encoded output. The key steps are:
/// 1.  **Unpacking**: The 12 input bytes are rearranged using `PSHUFB` to align
///     the bits correctly for easy extraction of 6-bit Base64 indices.
/// 2.  **Index Extraction**: A series of bitwise shifts and AND operations
///     extracts sixteen 6-bit indices from the rearranged bytes.
/// 3.  **Character Lookup**: The 6-bit indices are translated into ASCII Base64
///     characters using SIMD arithmetic, which is faster than traditional
///     branching lookups.
///
/// This approach is significantly faster than scalar methods because it avoids
/// branching and processes data in parallel.

use schmimmdee::{base64_encode_scalar, base64_encode_simd, format_ns, format_number};
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

fn main() {
    if !std::is_x86_feature_detected!("ssse3") {
        eprintln!("Warning: SSSE3 not detected. SIMD implementation will fall back to scalar.");
    }

    let data_path = "datasets/enwiki-latest-all-titles-in-ns0";
    if !Path::new(data_path).exists() {
        eprintln!("Error: Data file not found at {}. Please download it to run this benchmark.", data_path);
        return;
    }

    println!("Reading data for benchmark...");
    let data = fs::read(data_path).expect("Failed to read data file");
    let data_size = 10 * 1024 * 1024; // 10 MB slice
    let data_slice = &data[..data_size.min(data.len())];
    println!("Using {} of data for encoding.\n", format_number(data_slice.len()));

    let trials = 10;

    println!("{:-^80}", " Base64 Encoding Benchmark (PSHUFB) ");
    println!(
        "| {:>12} | {:>15} | {:>15} | {:>10} | {:>10} |",
        "Method", "Scalar", "SIMD", "Speedup", "Valid"
    );
    println!(
        "|{:-^14}|{:-^17}|{:-^17}|{:-^12}|{:-^12}|",
        "", "", "", "", ""
    );

    // Warmup
    black_box(base64_encode_scalar(data_slice));
    unsafe { black_box(base64_encode_simd(data_slice)) };

    // Benchmark
    let scalar_time: u128 = (0..trials)
        .map(|_| {
            let start = Instant::now();
            black_box(base64_encode_scalar(data_slice));
            start.elapsed().as_nanos()
        })
        .sum();

    let simd_time: u128 = (0..trials)
        .map(|_| {
            let start = Instant::now();
            unsafe { black_box(base64_encode_simd(data_slice)) };
            start.elapsed().as_nanos()
        })
        .sum();

    let avg_scalar = scalar_time as f64 / trials as f64;
    let avg_simd = simd_time as f64 / trials as f64;
    let speedup = avg_scalar / avg_simd;

    // Verification
    let scalar_result = base64_encode_scalar(data_slice);
    let simd_result = unsafe { base64_encode_simd(data_slice) };
    let valid = scalar_result == simd_result;
    if !valid {
        eprintln!("Validation failed: Scalar and SIMD results do not match.");
    }
    assert!(valid, "Results do not match!");

    println!(
        "| {:>12} | {:>15} | {:>15} | {:>9.2}x | {:>9} |",
        "base64_encode",
        format_ns(avg_scalar),
        format_ns(avg_simd),
        speedup,
        if valid { "✓" } else { "✗" }
    );
    println!("{:-^80}", "");
    println!("\nBenchmark complete!");
}