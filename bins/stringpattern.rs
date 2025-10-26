use schmimmdee::{string_pattern_match_scalar, string_pattern_match_simd, format_ns};
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

fn main() {
    if !std::is_x86_feature_detected!("sse2") {
        eprintln!("Warning: SSE2 not detected. SIMD implementation will fall back to scalar.");
    }

    let data_path = "datasets/enwiki-latest-all-titles-in-ns0";

    // Check if the data file exists
    if !Path::new(data_path).exists() {
        eprintln!("Error: Data file not found at {data_path}");
        eprintln!("Please download it from:");
        eprintln!("https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-all-titles-in-ns0.gz");
        eprintln!("Extract it and place it in the datasets/ directory.");
        std::process::exit(1);
    }

    // Read and process the data
    println!("Reading Wikipedia titles data...");
    let raw_data = match fs::read_to_string(data_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error reading file: {e}");
            std::process::exit(1);
        }
    };

    println!("Processing data...");
    // Replace newlines with commas and underscores with spaces
    let processed_data = raw_data.replace('\n', ",").replace('_', " ");

    println!("Data size: {} bytes", processed_data.len());
    println!("Processing complete!\n");

    // Search terms to test
    let search_terms = vec![
        "Path of Exile 2",
        "AVX-512",
        "Bannana", // Note: intentionally misspelled
    ];

    let trials = 10;

    for term in &search_terms {
        println!("{:-^80}", format!(" {} Search Benchmark ", term));
        println!(
            "| {:>12} | {:>15} | {:>15} | {:>10} | {:>10} |",
            "Method", "Scalar", "SIMD", "Speedup", "Valid"
        );
        println!(
            "|{:-^14}|{:-^17}|{:-^17}|{:-^12}|{:-^12}|",
            "", "", "", "", ""
        );

        // Warmup to prevent either from winning the benefits of a hot cache
        (0..3).for_each(|_| {
            black_box(string_pattern_match_scalar(processed_data.as_bytes(), term.as_bytes()));
            unsafe { black_box(string_pattern_match_simd(processed_data.as_bytes(), term.as_bytes())) };
        });

        // Benchmark scalar version
        let scalar_time: u128 = (0..trials)
            .map(|_| {
                let start = Instant::now();
                black_box(string_pattern_match_scalar(processed_data.as_bytes(), term.as_bytes()));
                start.elapsed().as_nanos()
            })
            .sum();

        // Benchmark SIMD version
        let simd_time: u128 = (0..trials)
            .map(|_| {
                let start = Instant::now();
                unsafe { black_box(string_pattern_match_simd(processed_data.as_bytes(), term.as_bytes())) };
                start.elapsed().as_nanos()
            })
            .sum();

        let avg_scalar = scalar_time as f64 / trials as f64;
        let avg_simd = simd_time as f64 / trials as f64;
        let speedup = avg_scalar / avg_simd;

        // Verification
        let scalar_res = string_pattern_match_scalar(processed_data.as_bytes(), term.as_bytes());
        let simd_res = unsafe { string_pattern_match_simd(processed_data.as_bytes(), term.as_bytes()) };
        let valid = scalar_res == simd_res;
        if !valid {
            eprintln!("Validation FAILED!");
        }
        assert!(valid, "Results do not match!");

        println!(
            "| {:>12} | {:>15} | {:>15} | {:>9.2}x | {:>9} |",
            "string_match",
            format_ns(avg_scalar),
            format_ns(avg_simd),
            speedup,
            if valid { "✓" } else { "✗" }
        );
        println!("{:-^80}", "");
        println!();
    }

    println!("Benchmark complete!");
}