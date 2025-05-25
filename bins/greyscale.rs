use image::{imageops::grayscale, io::Reader as ImageReader};

fn main() -> Result<(), image::ImageError> {
    // Open the image from the assets folder
    let img = ImageReader::open("./assets/test.png")?.decode()?;

    // Convert the image to grayscale
    let gray_img = grayscale(&img);

    // Save the grayscale image (optional)
    gray_img.save("./assets/test_gray.png")?;

    Ok(())
}
