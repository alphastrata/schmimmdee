// Fixed main.rs
use std::{collections::HashMap, fs, path::Path, time::Instant};

use schmimmdee::simd_histogram;

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
    let processed_data = raw_data.replace('\n', " ").replace('_', " ");

    // Step 1: Create word count HashMap
    println!("\nCreating word counts...");
    let word_counts = create_word_counts(&processed_data);
    println!("Unique words: {}", word_counts.len());

    // Step 2: Prepare data for histograms
    let all_chars: Vec<u8> = processed_data.bytes().collect();
    let mut simd_hist = [0u32; 256]; // Make it mutable
    let mut std_hist = [0u32; 256];

    // Step 3: Benchmark implementations
    println!("\nBenchmarking histogram implementations...");

    // Benchmark SIMD version - pass a mutable reference to the array directly
    let simd_start = Instant::now();
    simd_histogram(&all_chars, &mut [&mut simd_hist]); // Pass slice of mutable references
    let simd_duration = simd_start.elapsed();
    println!("SIMD histogram: {:?}", simd_duration);

    // Benchmark standard version
    let std_start = Instant::now();
    standard_histogram(&all_chars, &mut std_hist);
    let std_duration = std_start.elapsed();
    println!("Standard histogram: {:?}", std_duration);

    // Verify results match
    assert_eq!(simd_hist, std_hist, "Histogram results differ!");
    println!("Results verified identical");
}

fn create_word_counts(text: &str) -> HashMap<String, u32> {
    text.split_whitespace()
        .filter(|word| !word.is_empty())
        .fold(HashMap::new(), |mut counts, word| {
            *counts.entry(word.to_lowercase()).or_insert(0) += 1;
            counts
        })
}

fn standard_histogram(data: &[u8], histogram: &mut [u32; 256]) {
    data.iter().for_each(|&byte| histogram[byte as usize] += 1);
}
