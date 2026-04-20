use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use omni_search::{
    ModelBundle, OmniSearch, RuntimeConfig, env_path_resolved, is_supported_image_path,
    load_dotenv_from, runtime_config_from_env,
};
use serde::Serialize;

#[derive(Serialize)]
struct Output {
    bundle: String,
    family: String,
    model_id: String,
    repeats: usize,
    runtime: RuntimeSummary,
    texts: Vec<TextEmbedding>,
    images: Vec<ImageEmbedding>,
    timing_ms: TimingMs,
}

#[derive(Serialize)]
struct TextEmbedding {
    text: String,
    embedding: Vec<f32>,
}

#[derive(Serialize)]
struct ImageEmbedding {
    path: String,
    embedding: Vec<f32>,
}

#[derive(Serialize)]
struct RuntimeSummary {
    device: String,
    provider_policy: String,
    intra_threads: usize,
    inter_threads: Option<usize>,
    fgclip_max_patches: Option<usize>,
}

#[derive(Serialize)]
struct TimingMs {
    cold_text: f64,
    cold_image: f64,
    warm_text_avg: f64,
    warm_image_avg: f64,
    warm_image_batch_avg: f64,
    warm_image_batch_per_image_avg: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    load_dotenv_from(&root)?;
    let bundle_dir = env_path_resolved("OMNI_BUNDLE_DIR", &root)
        .unwrap_or_else(|| root.join("models/fgclip2_flat"));
    let samples_dir =
        env_path_resolved("OMNI_SAMPLES_DIR", &root).unwrap_or_else(|| root.join("samples"));
    let repeats = env::var("OMNI_REPEATS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(30);
    let texts = env::var("OMNI_TEXTS")
        .ok()
        .map(|value| {
            value
                .split('|')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>()
        })
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| vec!["山".to_owned(), "海边".to_owned(), "灯笼".to_owned()]);
    let image_paths = list_images(&samples_dir)?;

    let bundle = ModelBundle::load_from_dir(&bundle_dir)?;
    let model_id = bundle.info().model_id.clone();
    let runtime = runtime_config_from_env()?;

    let sdk = new_sdk(&bundle_dir, runtime.clone())?;
    sdk.preload_text()?;
    sdk.preload_image()?;

    let text_embeddings = texts
        .iter()
        .map(|text| {
            let embedding = sdk.embed_text(text)?;
            Ok(TextEmbedding {
                text: text.clone(),
                embedding: embedding.as_slice().to_vec(),
            })
        })
        .collect::<Result<Vec<_>, omni_search::Error>>()?;
    let image_embeddings = image_paths
        .iter()
        .map(|path| {
            let embedding = sdk.embed_image_path(path)?;
            Ok(ImageEmbedding {
                path: path.display().to_string(),
                embedding: embedding.as_slice().to_vec(),
            })
        })
        .collect::<Result<Vec<_>, omni_search::Error>>()?;

    let cold_text = {
        let sdk = new_sdk(&bundle_dir, runtime.clone())?;
        measure_once(|| {
            let _ = sdk.embed_text(&texts[0])?;
            Ok::<_, omni_search::Error>(())
        })?
    };
    let cold_image = {
        let sdk = new_sdk(&bundle_dir, runtime.clone())?;
        measure_once(|| {
            let _ = sdk.embed_image_path(&image_paths[0])?;
            Ok::<_, omni_search::Error>(())
        })?
    };

    let warm_text_avg = {
        let sdk = new_sdk(&bundle_dir, runtime.clone())?;
        sdk.preload_text()?;
        let _ = sdk.embed_text(&texts[0])?;
        measure_repeated(repeats, || {
            let _ = sdk.embed_text(&texts[0])?;
            Ok::<_, omni_search::Error>(())
        })?
    };
    let warm_image_avg = {
        let sdk = new_sdk(&bundle_dir, runtime.clone())?;
        sdk.preload_image()?;
        let _ = sdk.embed_image_path(&image_paths[0])?;
        measure_repeated(repeats, || {
            let _ = sdk.embed_image_path(&image_paths[0])?;
            Ok::<_, omni_search::Error>(())
        })?
    };
    let warm_image_batch_avg = {
        let sdk = new_sdk(&bundle_dir, runtime.clone())?;
        sdk.preload_image()?;
        let _ = sdk.embed_image_paths(&image_paths)?;
        measure_repeated(repeats, || {
            let _ = sdk.embed_image_paths(&image_paths)?;
            Ok::<_, omni_search::Error>(())
        })?
    };

    let output = Output {
        bundle: bundle_dir.display().to_string(),
        family: bundle.info().model_family.to_string(),
        model_id,
        repeats,
        runtime: RuntimeSummary {
            device: runtime.device.to_string(),
            provider_policy: runtime.provider_policy.to_string(),
            intra_threads: runtime.intra_threads,
            inter_threads: runtime.inter_threads,
            fgclip_max_patches: runtime.fgclip_max_patches,
        },
        texts: text_embeddings,
        images: image_embeddings,
        timing_ms: TimingMs {
            cold_text,
            cold_image,
            warm_text_avg,
            warm_image_avg,
            warm_image_batch_avg,
            warm_image_batch_per_image_avg: warm_image_batch_avg / image_paths.len() as f64,
        },
    };
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn new_sdk(bundle_dir: &Path, runtime: RuntimeConfig) -> Result<OmniSearch, omni_search::Error> {
    OmniSearch::builder()
        .from_local_model_dir(bundle_dir)
        .runtime_config(runtime)
        .build()
}

fn measure_once(
    f: impl FnOnce() -> Result<(), omni_search::Error>,
) -> Result<f64, omni_search::Error> {
    let start = Instant::now();
    f()?;
    Ok(start.elapsed().as_secs_f64() * 1000.0)
}

fn measure_repeated(
    repeats: usize,
    mut f: impl FnMut() -> Result<(), omni_search::Error>,
) -> Result<f64, omni_search::Error> {
    let start = Instant::now();
    for _ in 0..repeats {
        f()?;
    }
    Ok(start.elapsed().as_secs_f64() * 1000.0 / repeats as f64)
}

fn list_images(root: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut images = fs::read_dir(root)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && is_supported_image_path(path))
        .collect::<Vec<_>>();
    images.sort();
    if images.is_empty() {
        return Err(format!("no images found under {}", root.display()).into());
    }
    Ok(images)
}
