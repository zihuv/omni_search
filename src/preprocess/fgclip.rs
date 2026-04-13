use std::fs;
use std::path::Path;

use image::{DynamicImage, imageops::FilterType};
use ndarray::{Array, ArrayD, IxDyn};

use crate::error::Error;

pub(crate) struct FgClipImageInputs {
    pub pixel_values: ArrayD<f32>,
    pub pixel_attention_mask: ArrayD<i32>,
    pub spatial_height: usize,
    pub spatial_width: usize,
}

pub(crate) fn read_f32_file(path: &Path) -> Result<Vec<f32>, Error> {
    let bytes = fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(Error::image_preprocess(format!(
            "{} has {} bytes, not divisible by 4",
            path.display(),
            bytes.len()
        )));
    }

    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

pub(crate) fn determine_max_patches(
    width: u32,
    height: u32,
    patch_size: usize,
    default_max_patches: usize,
) -> usize {
    let raw = ((width as usize) / patch_size) * ((height as usize) / patch_size);
    let mut buckets = vec![128usize, 256, 576, 784, default_max_patches];
    buckets.sort_unstable();
    buckets.dedup();
    buckets
        .into_iter()
        .find(|candidate| raw <= *candidate)
        .unwrap_or(default_max_patches)
}

pub(crate) fn preprocess_image(
    image: &DynamicImage,
    patch_size: usize,
    max_patches: usize,
) -> Result<FgClipImageInputs, Error> {
    let image = image.to_rgb8();
    let (original_width, original_height) = image.dimensions();
    let (target_height, target_width) = get_image_size_for_max_num_patches(
        original_height as usize,
        original_width as usize,
        patch_size,
        max_patches,
    );
    let resized = image::imageops::resize(
        &image,
        target_width as u32,
        target_height as u32,
        FilterType::Triangle,
    );

    let spatial_height = target_height / patch_size;
    let spatial_width = target_width / patch_size;
    let valid_patches = spatial_height * spatial_width;
    let channels = patch_size * patch_size * 3;
    if valid_patches > max_patches {
        return Err(Error::image_preprocess(format!(
            "internal error: {valid_patches} valid patches > max_patches {max_patches}"
        )));
    }

    let mut pixel_values = vec![0.0f32; max_patches * channels];
    for patch_y in 0..spatial_height {
        for patch_x in 0..spatial_width {
            let patch_index = patch_y * spatial_width + patch_x;
            let mut dst = patch_index * channels;
            for y in 0..patch_size {
                for x in 0..patch_size {
                    let pixel = resized.get_pixel(
                        (patch_x * patch_size + x) as u32,
                        (patch_y * patch_size + y) as u32,
                    );
                    for channel in 0..3 {
                        pixel_values[dst] = pixel[channel] as f32 / 127.5 - 1.0;
                        dst += 1;
                    }
                }
            }
        }
    }

    let mut mask = vec![0i32; max_patches];
    for item in mask.iter_mut().take(valid_patches) {
        *item = 1;
    }

    Ok(FgClipImageInputs {
        pixel_values: Array::from_shape_vec(IxDyn(&[1, max_patches, channels]), pixel_values)
            .map_err(|error| Error::image_preprocess(error.to_string()))?,
        pixel_attention_mask: Array::from_shape_vec(IxDyn(&[1, max_patches]), mask)
            .map_err(|error| Error::image_preprocess(error.to_string()))?,
        spatial_height,
        spatial_width,
    })
}

pub(crate) fn stack_pixel_values(images: &[FgClipImageInputs]) -> Result<ArrayD<f32>, Error> {
    let batch = images.len();
    let shape = images
        .first()
        .ok_or_else(|| Error::image_preprocess("cannot stack an empty image batch"))?
        .pixel_values
        .shape()
        .to_vec();
    let max_patches = shape[1];
    let channels = shape[2];
    let mut values = Vec::with_capacity(batch * max_patches * channels);
    for image in images {
        if image.pixel_values.shape() != [1, max_patches, channels] {
            return Err(Error::image_preprocess(format!(
                "all fgclip pixel arrays must have shape [1,{max_patches},{channels}]"
            )));
        }
        values.extend_from_slice(
            image
                .pixel_values
                .as_slice()
                .ok_or_else(|| Error::image_preprocess("pixel array is not contiguous"))?,
        );
    }
    Array::from_shape_vec(IxDyn(&[batch, max_patches, channels]), values)
        .map_err(|error| Error::image_preprocess(error.to_string()))
}

pub(crate) fn stack_attention_masks(images: &[FgClipImageInputs]) -> Result<ArrayD<i32>, Error> {
    let batch = images.len();
    let max_patches = images
        .first()
        .ok_or_else(|| Error::image_preprocess("cannot stack an empty image batch"))?
        .pixel_attention_mask
        .shape()[1];
    let mut values = Vec::with_capacity(batch * max_patches);
    for image in images {
        if image.pixel_attention_mask.shape() != [1, max_patches] {
            return Err(Error::image_preprocess(format!(
                "all fgclip masks must have shape [1,{max_patches}]"
            )));
        }
        values.extend_from_slice(
            image
                .pixel_attention_mask
                .as_slice()
                .ok_or_else(|| Error::image_preprocess("mask array is not contiguous"))?,
        );
    }
    Array::from_shape_vec(IxDyn(&[batch, max_patches]), values)
        .map_err(|error| Error::image_preprocess(error.to_string()))
}

pub(crate) fn stack_f32_batches(
    arrays: &[ArrayD<f32>],
    shape: [usize; 3],
) -> Result<ArrayD<f32>, Error> {
    let mut values = Vec::with_capacity(shape.iter().product());
    for array in arrays {
        if array.shape() != [1, shape[1], shape[2]] {
            return Err(Error::image_preprocess(format!(
                "all arrays must have shape [1,{},{}]",
                shape[1], shape[2]
            )));
        }
        values.extend_from_slice(
            array
                .as_slice()
                .ok_or_else(|| Error::image_preprocess("array is not contiguous"))?,
        );
    }
    Array::from_shape_vec(IxDyn(&shape), values)
        .map_err(|error| Error::image_preprocess(error.to_string()))
}

pub(crate) fn build_positional_embedding(
    base_pos: &[f32],
    source_height: usize,
    source_width: usize,
    target_height: usize,
    target_width: usize,
    max_patches: usize,
    channels: usize,
) -> Result<ArrayD<f32>, Error> {
    if base_pos.len() != source_height * source_width * channels {
        return Err(Error::image_preprocess(format!(
            "unexpected vision position embedding length {}, expected {}",
            base_pos.len(),
            source_height * source_width * channels
        )));
    }

    let mut output = vec![0.0f32; max_patches * channels];
    for out_y in 0..target_height {
        let in_y = linear_source_coordinate(out_y, target_height, source_height);
        let y0 = in_y.floor().clamp(0.0, (source_height - 1) as f32) as usize;
        let y1 = (y0 + 1).min(source_height - 1);
        let wy = in_y - y0 as f32;

        for out_x in 0..target_width {
            let in_x = linear_source_coordinate(out_x, target_width, source_width);
            let x0 = in_x.floor().clamp(0.0, (source_width - 1) as f32) as usize;
            let x1 = (x0 + 1).min(source_width - 1);
            let wx = in_x - x0 as f32;
            let token = out_y * target_width + out_x;
            for channel in 0..channels {
                let top = lerp(
                    base_pos[((y0 * source_width + x0) * channels) + channel],
                    base_pos[((y0 * source_width + x1) * channels) + channel],
                    wx,
                );
                let bottom = lerp(
                    base_pos[((y1 * source_width + x0) * channels) + channel],
                    base_pos[((y1 * source_width + x1) * channels) + channel],
                    wx,
                );
                output[token * channels + channel] = lerp(top, bottom, wy);
            }
        }
    }

    let valid = target_height * target_width;
    if valid > 0 && valid < max_patches {
        for token in valid..max_patches {
            let src = output[..channels].to_vec();
            output[token * channels..(token + 1) * channels].copy_from_slice(&src);
        }
    }

    Array::from_shape_vec(IxDyn(&[1, max_patches, channels]), output)
        .map_err(|error| Error::image_preprocess(error.to_string()))
}

fn get_image_size_for_max_num_patches(
    image_height: usize,
    image_width: usize,
    patch_size: usize,
    max_num_patches: usize,
) -> (usize, usize) {
    fn scaled_size(scale: f64, size: usize, patch_size: usize) -> usize {
        let scaled = size as f64 * scale;
        let patched = (scaled / patch_size as f64).ceil() as usize * patch_size;
        patched.max(patch_size)
    }

    let eps = 1e-5f64;
    let mut scale_min = eps / 10.0;
    let mut scale_max = 100.0;
    while scale_max - scale_min >= eps {
        let scale = (scale_min + scale_max) / 2.0;
        let target_height = scaled_size(scale, image_height, patch_size);
        let target_width = scaled_size(scale, image_width, patch_size);
        let num_patches = (target_height / patch_size) * (target_width / patch_size);
        if num_patches <= max_num_patches {
            scale_min = scale;
        } else {
            scale_max = scale;
        }
    }

    (
        scaled_size(scale_min, image_height, patch_size),
        scaled_size(scale_min, image_width, patch_size),
    )
}

fn linear_source_coordinate(output_index: usize, output_size: usize, input_size: usize) -> f32 {
    let source = (output_index as f32 + 0.5) * input_size as f32 / output_size as f32 - 0.5;
    source.clamp(0.0, (input_size - 1) as f32)
}

fn lerp(a: f32, b: f32, weight: f32) -> f32 {
    a + (b - a) * weight
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, Rgb, RgbImage};

    use super::{build_positional_embedding, determine_max_patches, preprocess_image};

    #[test]
    fn determines_patch_bucket() {
        assert_eq!(determine_max_patches(1920, 1080, 16, 1024), 1024);
        assert_eq!(determine_max_patches(320, 240, 16, 1024), 576);
    }

    #[test]
    fn preprocesses_fgclip_image() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(64, 64, Rgb([255, 0, 0])));
        let encoded = preprocess_image(&image, 16, 128).unwrap();
        assert_eq!(encoded.pixel_values.shape(), [1, 128, 768]);
        assert_eq!(encoded.pixel_attention_mask.shape(), [1, 128]);
    }

    #[test]
    fn resizes_positional_embedding() {
        let base = vec![0.0f32; 16 * 16 * 4];
        let pos = build_positional_embedding(&base, 16, 16, 2, 2, 8, 4).unwrap();
        assert_eq!(pos.shape(), [1, 8, 4]);
    }
}
