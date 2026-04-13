use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use image::DynamicImage;
use image::imageops::FilterType;
use ndarray::{Array, ArrayD, IxDyn};
use ort::session::Session;
use ort::value::TensorRef;
use serde::Serialize;

#[derive(Serialize)]
struct Report {
    repeats: usize,
    bytes_len: usize,
    intra_threads: usize,
    avg_ms: AvgMs,
}

#[derive(Serialize)]
struct AvgMs {
    fs_read: f64,
    decode_memory: f64,
    decode_path: f64,
    preprocess: f64,
    ort_run_only: f64,
    ort_extract_only: f64,
    ort_total: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image_path = PathBuf::from(
        std::env::var_os("OMNI_IMAGE_PATH")
            .unwrap_or_else(|| "D:\\code\\omni_search\\samples\\pic1.jpg".into()),
    );
    let model_path = PathBuf::from(
        std::env::var_os("OMNI_IMAGE_ONNX")
            .unwrap_or_else(|| {
                "D:\\code\\vl-embedding-test\\artifacts\\chinese-clip-vit-base-patch16\\onnx\\vit-b-16.img.fp32.onnx"
                    .into()
            }),
    );
    let intra_threads = std::env::var("OMNI_INTRA_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4);
    let repeats = std::env::var("OMNI_REPEATS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(30);

    let bytes = fs::read(&image_path)?;
    let image = image::load_from_memory(&bytes)?;
    let tensor = preprocess_image(&image);
    let mut session = Session::builder()?
        .with_intra_threads(intra_threads)?
        .commit_from_file(&model_path)?;

    {
        let pixel_values = TensorRef::from_array_view(tensor.view())?;
        let warm_outputs = session.run(ort::inputs!["pixel_values" => pixel_values])?;
        let _ = warm_outputs["image_features"].try_extract_array::<f32>()?;
    }

    let fs_read = measure_avg(repeats, || {
        let _ = fs::read(&image_path)?;
        Ok::<_, Box<dyn std::error::Error>>(())
    })?;
    let decode_memory = measure_avg(repeats, || {
        let _ = image::load_from_memory(&bytes)?;
        Ok::<_, Box<dyn std::error::Error>>(())
    })?;
    let decode_path = measure_avg(repeats, || {
        let _ = image::open(&image_path)?;
        Ok::<_, Box<dyn std::error::Error>>(())
    })?;
    let preprocess = measure_avg(repeats, || {
        let _ = preprocess_image(&image);
        Ok::<_, Box<dyn std::error::Error>>(())
    })?;
    let ort_run_only = measure_avg(repeats, || {
        let pixel_values = TensorRef::from_array_view(tensor.view())?;
        let _ = session.run(ort::inputs!["pixel_values" => pixel_values])?;
        Ok::<_, Box<dyn std::error::Error>>(())
    })?;
    let ort_extract_only = measure_avg(repeats, || {
        let pixel_values = TensorRef::from_array_view(tensor.view())?;
        let outputs = session.run(ort::inputs!["pixel_values" => pixel_values])?;
        let _ = outputs["image_features"].try_extract_array::<f32>()?;
        Ok::<_, Box<dyn std::error::Error>>(())
    })?;
    let ort_total = measure_avg(repeats, || {
        let bytes = fs::read(&image_path)?;
        let image = image::load_from_memory(&bytes)?;
        let tensor = preprocess_image(&image);
        let pixel_values = TensorRef::from_array_view(tensor.view())?;
        let outputs = session.run(ort::inputs!["pixel_values" => pixel_values])?;
        let output = outputs["image_features"].try_extract_array::<f32>()?;
        let mut values = output.iter().copied().collect::<Vec<_>>();
        normalize_vector(&mut values)?;
        Ok::<_, Box<dyn std::error::Error>>(())
    })?;

    let report = Report {
        repeats,
        bytes_len: bytes.len(),
        intra_threads,
        avg_ms: AvgMs {
            fs_read,
            decode_memory,
            decode_path,
            preprocess,
            ort_run_only,
            ort_extract_only,
            ort_total,
        },
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn measure_avg(
    repeats: usize,
    mut f: impl FnMut() -> Result<(), Box<dyn std::error::Error>>,
) -> Result<f64, Box<dyn std::error::Error>> {
    let start = Instant::now();
    for _ in 0..repeats {
        f()?;
    }
    Ok(start.elapsed().as_secs_f64() * 1000.0 / repeats as f64)
}

fn preprocess_image(image: &DynamicImage) -> ArrayD<f32> {
    let image = image.to_rgb8();
    let resized = image::imageops::resize(&image, 224, 224, FilterType::CatmullRom);

    let plane = 224usize * 224usize;
    let mean = [0.48145466f32, 0.4578275, 0.40821073];
    let std = [0.26862954f32, 0.261_302_6, 0.275_777_1];
    let mut values = vec![0.0f32; 3 * plane];
    for y in 0..224usize {
        for x in 0..224usize {
            let pixel = resized.get_pixel(x as u32, y as u32);
            for channel in 0..3 {
                let value = pixel[channel] as f32 / 255.0;
                values[channel * plane + (y * 224usize) + x] =
                    (value - mean[channel]) / std[channel];
            }
        }
    }

    Array::from_shape_vec(IxDyn(&[1, 3, 224, 224]), values).unwrap()
}

fn normalize_vector(values: &mut [f32]) -> Result<(), Box<dyn std::error::Error>> {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm <= f32::MIN_POSITIVE {
        return Err("zero norm".into());
    }
    for value in values {
        *value /= norm;
    }
    Ok(())
}
