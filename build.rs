// build.rs
use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let target = env::var("TARGET").unwrap_or_default();
    let target_features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();

    // Determine optimal SIMD width based on target architecture and features
    let logical_lanes = if target.contains("x86_64") || target.contains("i686") {
        if target_features.contains("avx512f") {
            16 // AVX-512: 512 bits / 32 bits per f32 = 16 lanes
        } else if target_features.contains("avx2") || target_features.contains("avx") {
            8 // AVX/AVX2: 256 bits / 32 bits per f32 = 8 lanes
        } else if target_features.contains("sse2") {
            4 // SSE2: 128 bits / 32 bits per f32 = 4 lanes
        } else {
            1 // Fallback to scalar
        }
    } else if target.contains("aarch64") {
        if target_features.contains("neon") {
            4 // ARM NEON: 128 bits / 32 bits per f32 = 4 lanes
        } else {
            1 // Fallback to scalar
        }
    } else if target.contains("wasm32") {
        if target_features.contains("simd128") {
            4 // WebAssembly SIMD: 128 bits / 32 bits per f32 = 4 lanes
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
                        "const LOGICAL_LANES: usize = {}; // Auto-detected for {}",
                        logical_lanes, target
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
            println!(
                "cargo:warning=Updated LOGICAL_LANES to {} for target: {}",
                logical_lanes, target
            );
        }
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/lib.rs");
}
