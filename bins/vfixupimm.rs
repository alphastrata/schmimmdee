use schmimmdee::{format_ns, vfixupimm_scalar, vfixupimm_simd};
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let data_size = 1024;
    let data: Vec<f32> = (0..data_size).map(|i| i as f32).collect();

    let trials = 1000;

    println!("{:-^80}", " VFIXUPIMM Benchmark ");
    println!(
        "| {:>12} | {:>15} | {:>15} | {:>10} | {:>10} |",
        "Method", "Scalar", "SIMD", "Speedup", "Valid"
    );
    println!(
        "|{:-^14}|{:-^17}|{:-^17}|{:-^12}|{:-^12}|",
        "", "", "", "", ""
    );

    // Warmup
    (0..3).for_each(|_| {
        black_box(vfixupimm_scalar(&data));
        black_box(vfixupimm_simd(&data));
    });

    // Benchmark scalar
    let scalar_time: u128 = (0..trials)
        .map(|_| {
            let start = Instant::now();
            black_box(vfixupimm_scalar(&data));
            start.elapsed().as_nanos()
        })
        .sum();

    // Benchmark SIMD
    let simd_time: u128 = (0..trials)
        .map(|_| {
            let start = Instant::now();
            black_box(vfixupimm_simd(&data));
            start.elapsed().as_nanos()
        })
        .sum();

    // Calculate averages and speedup
    let avg_scalar = scalar_time as f64 / trials as f64;
    let avg_simd = simd_time as f64 / trials as f64;
    let speedup = avg_scalar / avg_simd;

    // Verify results
    let scalar_result = vfixupimm_scalar(&data);
    let simd_result = vfixupimm_simd(&data);
    let valid = scalar_result == simd_result;

    // Print formatted results
    println!(
        "| {:>12} | {:>15} | {:>15} | {:>9.2}x | {:>9} |",
        "VFIXUPIMM",
        format_ns(avg_scalar),
        format_ns(avg_simd),
        speedup,
        if valid { "✓" } else { "✗" }
    );

    println!("{:-^80}", "");
    println!();

    println!("Benchmark complete!");
}
