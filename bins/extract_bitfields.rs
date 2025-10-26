/// This benchmark demonstrates a "killer application" of the VPMULTISHIFTQB
/// instruction from the AVX-512VBMI instruction set: extracting multiple,
/// arbitrary bit-fields from a set of integers in parallel.
///
/// ## Bit-field Extraction
/// This is a fundamental operation when parsing packed data formats, where
/// information is not neatly aligned to byte boundaries. Examples include:
/// -   **Network Protocols**: Parsing fields from TCP, IP, or Ethernet headers.
/// -   **Decompression**: Decoding Huffman or dictionary-based compression schemes.
/// -   **Instruction Decoders**: Breaking an instruction into its opcode and operands.
///
/// ## The VPMULTISHIFTQB Advantage
/// The `VPMULTISHIFTQB` instruction (`_mm512_multishift_epi64_epi8`) is a
/// specialized hardware shifter. For each 64-bit lane in a 512-bit vector, it
/// can perform 8 independent shifts on the same 64-bit data element.
///
/// This benchmark's SIMD implementation:
/// 1.  Loads a vector of 8 `u64` data values.
/// 2.  Loads a corresponding vector of 8 `u64` control values. Each byte of a
///     control `u64` specifies a starting bit (0-63) for an 8-bit field to
///     extract from the corresponding data `u64`.
/// 3.  A single `VPMULTISHIFTQB` instruction performs all 64 extractions (8
///     extractions for each of the 8 lanes) in parallel.
/// 4.  The 8 extracted bytes for each lane are packed into a `u64` in the result vector.
///
/// This provides a massive speedup over scalar code, which would require hundreds
/// of shift and mask operations to achieve the same result.

use schmimmdee::{multishift_extract_scalar, multishift_extract_simd, format_ns, format_number};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    if !std::is_x86_feature_detected!("avx512vbmi") {
        eprintln!("Warning: AVX-512VBMI not detected. SIMD implementation will fall back to scalar.");
    }

    const VECTOR_SIZE: usize = 4 * 1024 * 1024; // 4M u64s

    println!("Generating data and control vectors of {} u64 elements...", format_number(VECTOR_SIZE));
    let mut rng = StdRng::seed_from_u64(42);
    let data: Vec<u64> = (0..VECTOR_SIZE).map(|_| rng.gen()).collect();
    // Control bytes specify a shift from 0-63
    let controls: Vec<u64> = (0..VECTOR_SIZE).map(|_| {
        let mut ctrl = 0u64;
        for i in 0..8 {
            ctrl |= (rng.gen_range(0..64) as u64) << (i * 8);
        }
        ctrl
    }).collect();
    println!("Done.\n");

    let trials = 10;

    println!("{:-^80}", " Bitfield Extraction Benchmark (VPMULTISHIFTQB) ");
    println!(
        "| {:>12} | {:>15} | {:>15} | {:>10} | {:>10} |",
        "Method", "Scalar", "SIMD", "Speedup", "Valid"
    );
    println!(
        "|{:-^14}|{:-^17}|{:-^17}|{:-^12}|{:-^12}|",
        "", "", "", "", ""
    );

    // --- Warmup ---
    black_box(multishift_extract_scalar(&data, &controls));
    black_box(multishift_extract_simd(&data, &controls));

    // --- Scalar Benchmark ---
    let scalar_time: u128 = (0..trials)
        .map(|_| {
            let start = Instant::now();
            black_box(multishift_extract_scalar(&data, &controls));
            start.elapsed().as_nanos()
        })
        .sum();

    // --- SIMD Benchmark ---
    let simd_time: u128 = (0..trials)
        .map(|_| {
            let start = Instant::now();
            black_box(multishift_extract_simd(&data, &controls));
            start.elapsed().as_nanos()
        })
        .sum();

    let avg_scalar = scalar_time as f64 / trials as f64;
    let avg_simd = simd_time as f64 / trials as f64;
    let speedup = avg_scalar / avg_simd;

    // Verification
    let scalar_res = multishift_extract_scalar(&data, &controls);
    let simd_res = multishift_extract_simd(&data, &controls);
    let valid = scalar_res == simd_res;
    if !valid {
        eprintln!("Validation FAILED!");
    }
    assert!(valid, "Results do not match!");

    println!(
        "| {:>12} | {:>15} | {:>15} | {:>9.2}x | {:>9} |",
        "extract",
        format_ns(avg_scalar),
        format_ns(avg_simd),
        speedup,
        if valid { "✓" } else { "✗" }
    );
    println!("{:-^80}", "");
    println!("\nBenchmark complete!");
}
