use image::{DynamicImage, Rgba, imageops::grayscale, io::Reader as ImageReader};
use std::simd::{Simd, SimdFloat, SimdPartialOrd, StdFloat};

fn image_to_rgba_vec(img: &DynamicImage) -> Vec<[f32; 4]> {
    // Convert to RGBA8 if not already in that format
    let rgba_img = img.to_rgba8();
    let pixels = rgba_img.pixels();

    // Map each Rgba<u8> pixel to [f32; 4] (normalized to 0.0..=1.0)
    pixels
        .map(|Rgba([r, g, b, a])| {
            [
                *r as f32 / 255.0,
                *g as f32 / 255.0,
                *b as f32 / 255.0,
                *a as f32 / 255.0,
            ]
        })
        .collect()
}

fn main() -> Result<(), image::ImageError> {
    // Open the image from the assets folder
    let img = ImageReader::open("./assets/test.png")?.decode()?;

    // Convert the image to grayscale
    let gray_img = grayscale(&img);

    // Save the grayscale image (optional)
    gray_img.save("./assets/test_gray.png")?;

    Ok(())
}
