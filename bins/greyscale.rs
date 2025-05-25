use image::{DynamicImage, Rgba, imageops::grayscale, io::Reader as ImageReader};

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
use std::simd::{Simd, SimdFloat, SimdPartialOrd, StdFloat};

/// Convert RGBA ([f32;4]) to grayscale ([f32;3]) using SIMD.
/// Assumes input is normalized (0.0..=1.0).
pub fn rgba_to_gray_simd(rgba: &[[f32; 4]]) -> Vec<[f32; 3]> {
    // We'll process 8 pixels at a time (8x RGBA = 32 floats)
    const LANES: usize = 8;
    let mut output = Vec::with_capacity(rgba.len());

    // Weights for R, G, B (alpha ignored)
    let r_weight = Simd::<f32, LANES>::splat(0.2126);
    let g_weight = Simd::<f32, LANES>::splat(0.7152);
    let b_weight = Simd::<f32, LANES>::splat(0.0722);

    for chunk in rgba.chunks_exact(LANES) {
        // Transpose RGBA into separate R, G, B, A SIMD vectors
        let (mut r, mut g, mut b, _) = ([0.0; LANES], [0.0; LANES], [0.0; LANES], [0.0; LANES]);

        for (i, &[ri, gi, bi, _]) in chunk.iter().enumerate() {
            r[i] = ri;
            g[i] = gi;
            b[i] = bi;
        }

        let r_simd = Simd::from_array(r);
        let g_simd = Simd::from_array(g);
        let b_simd = Simd::from_array(b);

        // Compute luminance: 0.2126*R + 0.7152*G + 0.0722*B
        let gray = r_simd.mul_add(r_weight, g_simd.mul_add(g_weight, b_simd * b_weight));

        // Store as RGB (repeating luminance for all 3 channels)
        let gray_array = gray.to_array();
        for &l in gray_array.iter() {
            output.push([l, l, l]); // RGB (no alpha)
        }
    }

    // Handle remaining pixels (if input isn't a multiple of LANES)
    for &[r, g, b, _] in rgba.chunks_exact(LANES).remainder() {
        let l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        output.push([l, l, l]);
    }

    output
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
