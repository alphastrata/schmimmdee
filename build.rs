use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let target = env::var("TARGET").unwrap_or_default();
    let target_features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();

    // Determine optimal SIMD width based on target architecture and features
    let logical_lanes = if target.contains("x86_64") || target.contains("i686") {
        if target_features.contains("avx512f") {
            8 // AVX-512: 512 bits / 64 bits per u64 = 8 lanes
        } else if target_features.contains("avx2") || target_features.contains("avx") {
            4 // AVX/AVX2: 256 bits / 64 bits per u64 = 4 lanes
        } else if target_features.contains("sse2") {
            2 // SSE2: 128 bits / 64 bits per u64 = 2 lanes
        } else {
            1 // Fallback to scalar
        }
    } else if target.contains("aarch64") {
        if target_features.contains("neon") {
            2 // ARM NEON: 128 bits / 64 bits per u64 = 2 lanes
        } else {
            1 // Fallback to scalar
        }
    } else if target.contains("wasm32") {
        if target_features.contains("simd128") {
            2 // WebAssembly SIMD: 128 bits / 64 bits per u64 = 2 lanes
        } else {
            1 // Fallback to scalar
        }
    } else {
        1 // Conservative fallback for unknown targets
    };

    // Read existing lib.rs
    let lib_path = Path::new("src").join("lib.rs");
    if let Ok(content) = fs::read_to_string(&lib_path) {
        // Find and replace the LOGICAL_LANES line
        let new_content = content
            .lines()
            .map(|line| {
                if line
                    .trim_start()
                    .starts_with("const LOGICAL_LANES: usize =")
                {
                    format!(
                        "const LOGICAL_LANES: usize = {logical_lanes}; // Auto-detected for {target}"
                    )
                } else {
                    line.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        // Write back only if changed
        if new_content != content {
            fs::write(&lib_path, new_content).expect("Failed to update lib.rs");
            println!("cargo:warning=Updated LOGICAL_LANES to {logical_lanes} for target: {target}");
        }
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/lib.rs");
}
