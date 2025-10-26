use std::arch::x86_64::*;
use std::env;
use std::path::Path;
use std::time::Instant;
use image::io::Reader as ImageReader;

/// SIMD implementation of RGBA → ARGB conversion.
/// Works only on x86_64 with SSSE3 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn shuffle_rgba_to_argb(chunk: &[u8]) -> [u8; 16] {
    const MASK: [u8; 16] = [
        3, 2, 1, 0,   // pixel 0
        7, 6, 5, 4,   // pixel 1
        11, 10, 9, 8, // pixel 2
        15, 14, 13, 12, // pixel 3
    ];
    let vec = _mm_loadu_si128(chunk.as_ptr() as *const __m128i);
    let mask = _mm_loadu_si128(MASK.as_ptr() as *const __m128i);
    let shuffled = _mm_shuffle_epi8(vec, mask);
    let mut out = [0u8; 16];
    _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, shuffled);
    out
}

/// Naïve, byte‑wise RGBA → ARGB conversion.
fn shuffle_rgba_to_argb_naive(chunk: &[u8]) -> [u8; 16] {
    let mut out = [0u8; 16];
    for i in 0..4 {
        let base = i * 4;
        out[(i * 4)] = chunk[base + 3]; // A
        out[i * 4 + 1] = chunk[base]; // R
        out[i * 4 + 2] = chunk[base + 1]; // G
        out[i * 4 + 3] = chunk[base + 2]; // B
    }
    out
}

fn main() {
    // 1️⃣ Determine which image to process
    let args: Vec<String> = env::args().collect();
    let img_path = if args.len() > 1 {
        Path::new(&args[1])
    } else {
        Path::new("./assets/lenna.png")
    };

    // 2️⃣ Load the image
    let img = ImageReader::open(img_path)
        .expect("Failed to open image")
        .decode()
        .expect("Failed to decode image");

    // 3️⃣ Convert to RGBA8 and get the raw byte buffer
    let rgba: image::RgbaImage = img.to_rgba8();
    let raw = rgba.into_raw();

    // Sanity check: data should be a multiple of 16 bytes
    assert_eq!(
        raw.len() % 16,
        0,
        "Image data length must be a multiple of 16"
    );

    // 4️⃣ Warm‑up to avoid cold‑cache effects
    for chunk in raw.chunks_exact(16) {
        unsafe { shuffle_rgba_to_argb(chunk); };
    }
    for chunk in raw.chunks_exact(16) {
        let _ = shuffle_rgba_to_argb_naive(chunk);
    }

    // 5️⃣ Benchmark
    let trials = 10;
    let simd_time: u128 = (0..trials)
        .map(|_| {
            let start = Instant::now();
            let mut out = Vec::with_capacity(raw.len());
            for chunk in raw.chunks_exact(16) {
                let buf = unsafe { shuffle_rgba_to_argb(chunk) };
                out.extend_from_slice(&buf);
            }
            start.elapsed().as_nanos()
        })
        .sum();

    let naive_time: u128 = (0..trials)
        .map(|_| {
            let start = Instant::now();
            let mut out = Vec::with_capacity(raw.len());
            for chunk in raw.chunks_exact(16) {
                let buf = shuffle_rgba_to_argb_naive(chunk);
                out.extend_from_slice(&buf);
            }
            start.elapsed().as_nanos()
        })
        .sum();

    let avg_simd = simd_time as f64 / trials as f64;
    let avg_naive = naive_time as f64 / trials as f64;
    let speedup = avg_naive / avg_simd;

    // 6️⃣ Output results
    println!("Image: {}", img_path.display());
    println!("SIMD conversion average: {:.2} µs", avg_simd / 1_000.0);
    println!("Naïve conversion average: {:.2} µs", avg_naive / 1_000.0);
    println!("Speedup: {:.2}×", speedup);
}
