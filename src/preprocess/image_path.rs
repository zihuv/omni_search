use std::path::Path;

use image::{DynamicImage, ImageReader};

use crate::error::Error;

pub(crate) fn load_image_from_path(path: &Path) -> Result<DynamicImage, Error> {
    let reader = ImageReader::open(path)
        .map_err(|error| Error::image_preprocess(format!("{}: {error}", path.display())))?;

    let reader = reader
        .with_guessed_format()
        .map_err(|error| Error::image_preprocess(format!("{}: {error}", path.display())))?;

    reader
        .decode()
        .map_err(|error| Error::image_preprocess(format!("{}: {error}", path.display())))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Cursor;

    use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
    use tempfile::tempdir;

    use super::load_image_from_path;

    fn sample_image() -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_pixel(3, 2, Rgb([12, 34, 56])))
    }

    #[test]
    fn decodes_webp_bytes_from_png_path() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("mislabeled.png");
        let image = sample_image();
        let mut bytes = Cursor::new(Vec::new());

        image.write_to(&mut bytes, ImageFormat::WebP).unwrap();
        fs::write(&path, bytes.into_inner()).unwrap();

        let decoded = load_image_from_path(&path).unwrap();
        assert_eq!(decoded.width(), 3);
        assert_eq!(decoded.height(), 2);
    }

    #[test]
    fn decodes_png_bytes_from_png_path() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("normal.png");
        let image = sample_image();
        let mut bytes = Cursor::new(Vec::new());

        image.write_to(&mut bytes, ImageFormat::Png).unwrap();
        fs::write(&path, bytes.into_inner()).unwrap();

        let decoded = load_image_from_path(&path).unwrap();
        assert_eq!(decoded.width(), 3);
        assert_eq!(decoded.height(), 2);
    }

    #[cfg(feature = "avif")]
    #[test]
    fn decodes_avif_bytes_from_png_path() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("mislabeled.png");
        let image = sample_image();
        let mut bytes = Cursor::new(Vec::new());

        image.write_to(&mut bytes, ImageFormat::Avif).unwrap();
        fs::write(&path, bytes.into_inner()).unwrap();

        let decoded = load_image_from_path(&path).unwrap();
        assert_eq!(decoded.width(), 3);
        assert_eq!(decoded.height(), 2);
    }
}
