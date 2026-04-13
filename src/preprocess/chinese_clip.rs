use image::{DynamicImage, imageops::FilterType};
use ndarray::{Array, ArrayD, IxDyn};

use crate::error::Error;
use crate::manifest::CropMode;

pub(crate) struct ClipImagePreprocessConfig {
    pub image_size: usize,
    pub resize_shortest_edge: usize,
    pub crop: CropMode,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

pub(crate) fn preprocess_image(
    image: &DynamicImage,
    config: &ClipImagePreprocessConfig,
) -> Result<ArrayD<f32>, Error> {
    if config.crop != CropMode::Center {
        return Err(Error::image_preprocess(
            "only center crop is supported for clip_image preprocess",
        ));
    }

    let image = image.to_rgb8();
    let (width, height) = image.dimensions();
    let short_edge = width.min(height).max(1);
    let scale = config.resize_shortest_edge as f32 / short_edge as f32;
    let resized_width = ((width as f32 * scale).round() as u32).max(config.image_size as u32);
    let resized_height = ((height as f32 * scale).round() as u32).max(config.image_size as u32);
    let resized =
        image::imageops::resize(&image, resized_width, resized_height, FilterType::Triangle);

    let crop_size = config.image_size as u32;
    let left = (resized_width - crop_size) / 2;
    let top = (resized_height - crop_size) / 2;
    let cropped = image::imageops::crop_imm(&resized, left, top, crop_size, crop_size).to_image();

    let plane = config.image_size * config.image_size;
    let mut values = vec![0.0f32; 3 * plane];
    for y in 0..config.image_size {
        for x in 0..config.image_size {
            let pixel = cropped.get_pixel(x as u32, y as u32);
            for channel in 0..3 {
                let value = pixel[channel] as f32 / 255.0;
                values[channel * plane + (y * config.image_size) + x] =
                    (value - config.mean[channel]) / config.std[channel];
            }
        }
    }

    Array::from_shape_vec(IxDyn(&[1, 3, config.image_size, config.image_size]), values)
        .map_err(|error| Error::image_preprocess(error.to_string()))
}

pub(crate) fn stack_image_batches(
    images: &[ArrayD<f32>],
    image_size: usize,
) -> Result<ArrayD<f32>, Error> {
    let batch = images.len();
    let mut values = Vec::with_capacity(batch * 3 * image_size * image_size);
    for image in images {
        if image.shape() != [1, 3, image_size, image_size] {
            return Err(Error::image_preprocess(format!(
                "all chinese clip images must have shape [1,3,{image_size},{image_size}]"
            )));
        }
        values.extend_from_slice(
            image
                .as_slice()
                .ok_or_else(|| Error::image_preprocess("image tensor is not contiguous"))?,
        );
    }

    Array::from_shape_vec(IxDyn(&[batch, 3, image_size, image_size]), values)
        .map_err(|error| Error::image_preprocess(error.to_string()))
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, Rgb, RgbImage};

    use crate::manifest::CropMode;

    use super::{ClipImagePreprocessConfig, preprocess_image};

    #[test]
    fn preprocesses_clip_image() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(320, 240, Rgb([128, 64, 32])));
        let tensor = preprocess_image(
            &image,
            &ClipImagePreprocessConfig {
                image_size: 224,
                resize_shortest_edge: 224,
                crop: CropMode::Center,
                mean: [0.5, 0.5, 0.5],
                std: [0.5, 0.5, 0.5],
            },
        )
        .unwrap();
        assert_eq!(tensor.shape(), [1, 3, 224, 224]);
    }
}
