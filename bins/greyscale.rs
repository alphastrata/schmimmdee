use image::{DynamicImage, ImageBuffer, Rgb};
use schmimmdee::rgba_to_gray_simd_u8;

/// Extract RGBA pixels as `Vec<[u8; 4]>` from a `DynamicImage`.
pub fn image_to_rgba_u8(img: &DynamicImage) -> Vec<[u8; 4]> {
    let rgba = img.to_rgba8();
    rgba.pixels().map(|p| [p[0], p[1], p[2], p[3]]).collect()
}

pub fn vec_to_dynamic_u8(pixels: Vec<[u8; 3]>, width: u32, height: u32) -> DynamicImage {
    let raw_data: Vec<u8> = pixels.into_iter().flat_map(|[r, g, b]| [r, g, b]).collect();
    let buffer = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, raw_data)
        .expect("Invalid buffer dimensions");
    DynamicImage::ImageRgb8(buffer)
}

fn main() -> Result<(), image::ImageError> {
    // Load image
    let img = image::io::Reader::open("./assets/lenna.png")?.decode()?;
    let (width, height) = (img.width(), img.height());

    // Convert to RGBA u8
    let rgba_u8 = image_to_rgba_u8(&img);

    // Benchmark SIMD version
    let start_simd = std::time::Instant::now();
    let simd_gray = rgba_to_gray_simd_u8(&rgba_u8);
    let simd_time = start_simd.elapsed();

    // Benchmark baseline (image crate's grayscale)
    let start_baseline = std::time::Instant::now();
    let baseline_gray = img.grayscale();
    let baseline_time = start_baseline.elapsed();

    println!("SIMD: {simd_time:?}, Baseline: {baseline_time:?}");

    // Save results
    vec_to_dynamic_u8(simd_gray, width, height).save("lena_simd_gray.png")?;
    baseline_gray.save("lena_baseline_gray.png")?;

    Ok(())
}
