/// This benchmark demonstrates a "killer application" of the PCLMULQDQ instruction:
/// hardware-accelerated CRC32 (Cyclic Redundancy Check) calculation.
///
/// The specific variant implemented is CRC32-C (Castagnoli), which is widely
/// used in networking and storage systems (like iSCSI, Btrfs, etc.).
///
/// ## The Algorithm
/// CRC is fundamentally a polynomial division over the GF(2) finite field. A
/// message (a long stream of bits) is treated as a polynomial, which is divided
/// by a generator polynomial. The remainder of this division is the CRC checksum.
///
/// ## The SIMD Advantage with PCLMULQDQ
/// The `PCLMULQDQ` instruction performs a "carry-less multiplication" of two
/// 64-bit integers. This operation is equivalent to polynomial multiplication
/// over GF(2), which is the core, computationally-intensive part of CRC calculation.
///
/// A full SIMD algorithm processes data in large blocks (e.g., 64-128 bytes at a time),
/// using `PCLMULQDQ` to "fold" these blocks into a 128-bit accumulator. After
/// processing all data, a final reduction step computes the 32-bit checksum.
/// This method is orders of magnitude faster than traditional table-based scalar
/// implementations.

use schmimmdee::{crc32c_scalar, crc32c_simd, format_ns, format_number};
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

fn main() {
    if !std::is_x86_feature_detected!("pclmulqdq") {
        eprintln!("Warning: PCLMULQDQ feature not detected. SIMD implementation will fall back to scalar.");
    }

    let data_path = "datasets/enwiki-latest-all-titles-in-ns0";
    if !Path::new(data_path).exists() {
        eprintln!("Error: Data file not found at {}. Please download it to run this benchmark.", data_path);
        return;
    }

    println!("Reading data for benchmark...");
    let data = fs::read(data_path).expect("Failed to read data file");
    println!("Calculating CRC32-C over {} of data.\n", format_number(data.len()));

    let trials = 10;

    println!("{:-^80}", " CRC32-C Calculation Benchmark (PCLMULQDQ) ");
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
            black_box(crc32c_scalar(&data));
            start.elapsed().as_nanos()
        })
        .sum();

    // --- SIMD Benchmark ---
    let simd_time: u128 = (0..trials)
        .map(|_| {
            let start = Instant::now();
            black_box(unsafe { crc32c_simd(&data) });
            start.elapsed().as_nanos()
        })
        .sum();

    let avg_scalar = scalar_time as f64 / trials as f64;
    let avg_simd = simd_time as f64 / trials as f64;
    let speedup = avg_scalar / avg_simd;

    // Verification
    let scalar_res = crc32c_scalar(&data);
    let simd_res = unsafe { crc32c_simd(&data) };
    let valid = scalar_res == simd_res;
    if !valid {
        eprintln!("Validation FAILED: Scalar=0x{:08x}, SIMD=0x{:08x}", scalar_res, simd_res);
    }
    assert!(valid, "Results do not match!");

    println!(
        "| {:>12} | {:>15} | {:>15} | {:>9.2}x | {:>9} |",
        "crc32c",
        format_ns(avg_scalar),
        format_ns(avg_simd),
        speedup,
        if valid { "✓" } else { "✗" }
    );
    println!("{:-^80}", "");
    println!("Note: The SIMD implementation is a placeholder and falls back to scalar.");
    println!("A real-world implementation would show a significant speedup.");
    println!("\nBenchmark complete!");
}
