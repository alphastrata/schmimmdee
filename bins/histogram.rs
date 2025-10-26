use schmimmdee::{histogram_scalar, histogram_simd, format_ns, format_number};
use std::{collections::HashMap, fs, hint::black_box, path::Path, time::Instant};

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
    let processed_data = raw_data.replace(['\n', '_'], " ");

    // Step 1: Create word count HashMap
    println!("\nCreating word counts...");
    let word_counts = create_word_counts(&processed_data);
    println!("Unique words: {}", word_counts.len());

    // Step 2: Prepare data for histograms
    let all_chars: Vec<u8> = processed_data.bytes().collect();

    // Test different data sizes
    let sizes = [
        1_000,
        10_000,
        100_000,
        1_000_000,
        if all_chars.len() > 10_000_000 {
            10_000_000
        } else {
            all_chars.len()
        },
        all_chars.len(),
    ];
    let trials = 10;

    println!("\n{:-^80}", " Histogram Benchmark Results ");
    println!(
        "| {:>12} | {:>15} | {:>15} | {:>10} | {:>10} |",
        "Elements", "Scalar", "SIMD", "Speedup", "Valid"
    );
    println!(
        "|{:-^14}|{:-^17}|{:-^17}|{:-^12}|{:-^12}|",
        "", "", "", "", ""
    );

    for &size in &sizes {
        if size == 0 {
            continue;
        }

        let data_slice = &all_chars[..size.min(all_chars.len())];

        // Warmup to prevent cache effects
        for _ in 0..3 {
            black_box(histogram_scalar(data_slice));
            unsafe { black_box(histogram_simd(data_slice)) };
        }

        // Benchmark scalar version
        let scalar_time: u128 = (0..trials)
            .map(|_| {
                let start = Instant::now();
                black_box(histogram_scalar(data_slice));
                start.elapsed().as_nanos()
            })
            .sum();

        // Benchmark SIMD version
        let simd_time: u128 = (0..trials)
            .map(|_| {
                let start = Instant::now();
                unsafe { black_box(histogram_simd(data_slice)) };
                start.elapsed().as_nanos()
            })
            .sum();

        // Calculate averages and speedup
        let avg_scalar = scalar_time as f64 / trials as f64;
        let avg_simd = simd_time as f64 / trials as f64;
        let speedup = avg_scalar / avg_simd;

        // Verify results match
        let std_hist = histogram_scalar(data_slice);
        let simd_hist = unsafe { histogram_simd(data_slice) };
        let valid = std_hist == simd_hist;
        assert!(valid);

        // Print formatted results
        println!(
            "| {:>12} | {:>15} | {:>15} | {:>9.2}x | {:>9} |",
            format_number(size),
            format_ns(avg_scalar),
            format_ns(avg_simd),
            speedup,
            if valid { "✓" } else { "✗" }
        );
    }

    println!("{:-^80}", "");
}

fn create_word_counts(text: &str) -> HashMap<String, u32> {
    text.split_whitespace()
        .filter(|word| !word.is_empty())
        .fold(HashMap::new(), |mut counts, word| {
            *counts.entry(word.to_lowercase()).or_insert(0) += 1;
            counts
        })
}