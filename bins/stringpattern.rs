use schmimmdee::{format_ns, simd_contains_pattern, simd_find_str};
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

fn main() {
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
            "Method", "Std Lib", "SIMD", "Speedup", "Valid"
        );
        println!(
            "|{:-^14}|{:-^17}|{:-^17}|{:-^12}|{:-^12}|",
            "", "", "", "", ""
        );

        // Warmup to prevent either from winning the benefits of a hot cache
        (0..3).for_each(|_| {
            black_box(processed_data.contains(term));
            black_box(simd_contains_pattern(
                processed_data.as_bytes(),
                term.as_bytes(),
            ));
            black_box(processed_data.find(term));
            black_box(simd_find_str(&processed_data, term));
        });

        // Benchmark contains operations
        let std_contains_time: u128 = (0..trials)
            .map(|_| {
                let start = Instant::now();
                black_box(processed_data.contains(term));
                start.elapsed().as_nanos()
            })
            .sum();

        let simd_contains_time: u128 = (0..trials)
            .map(|_| {
                let start = Instant::now();
                black_box(simd_contains_pattern(
                    processed_data.as_bytes(),
                    term.as_bytes(),
                ));
                start.elapsed().as_nanos()
            })
            .sum();

        // Calculate contains averages and speedup
        let avg_std_contains = std_contains_time as f64 / trials as f64;
        let avg_simd_contains = simd_contains_time as f64 / trials as f64;
        let contains_speedup = avg_std_contains / avg_simd_contains;

        // Verify contains results
        let std_contains_result = processed_data.contains(term);
        let simd_contains_result =
            simd_contains_pattern(processed_data.as_bytes(), term.as_bytes());

        // Assert that results are exactly the same
        assert_eq!(
            std_contains_result, simd_contains_result,
            "Contains results don't match for term '{term}': std={std_contains_result}, simd={simd_contains_result}"
        );

        let contains_valid = std_contains_result == simd_contains_result;

        // Benchmark find operations
        let std_find_time: u128 = (0..trials)
            .map(|_| {
                let start = Instant::now();
                black_box(processed_data.find(term));
                start.elapsed().as_nanos()
            })
            .sum();

        let simd_find_time: u128 = (0..trials)
            .map(|_| {
                let start = Instant::now();
                black_box(simd_find_str(&processed_data, term));
                start.elapsed().as_nanos()
            })
            .sum();

        // Calculate find averages and speedup
        let avg_std_find = std_find_time as f64 / trials as f64;
        let avg_simd_find = simd_find_time as f64 / trials as f64;
        let find_speedup = avg_std_find / avg_simd_find;

        // Verify find results
        let std_find_result = processed_data.find(term);
        let simd_find_result = simd_find_str(&processed_data, term);

        // Assert that results are exactly the same
        assert_eq!(
            std_find_result, simd_find_result,
            "Find results don't match for term '{term}': std={std_find_result:?}, simd={simd_find_result:?}"
        );

        let find_valid = std_find_result == simd_find_result;

        // Print formatted results
        println!(
            "| {:>12} | {:>15} | {:>15} | {:>9.2}x | {:>9} |",
            "contains",
            format_ns(avg_std_contains),
            format_ns(avg_simd_contains),
            contains_speedup,
            if contains_valid { "✓" } else { "✗" }
        );

        println!(
            "| {:>12} | {:>15} | {:>15} | {:>9.2}x | {:>9} |",
            "find",
            format_ns(avg_std_find),
            format_ns(avg_simd_find),
            find_speedup,
            if find_valid { "✓" } else { "✗" }
        );

        println!("{:-^80}", "");
        println!();
    }

    println!("Benchmark complete!");
}
