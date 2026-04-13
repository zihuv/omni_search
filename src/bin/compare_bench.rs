use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use omni_search::{ModelBundle, OmniSearch, OmniSearchConfig, RuntimeConfig};
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
    let bundle_dir =
        env_path("OMNI_BUNDLE_DIR").unwrap_or_else(|| root.join("models/fgclip2_bundle"));
    let samples_dir = env_path("OMNI_SAMPLES_DIR").unwrap_or_else(|| root.join("samples"));
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
    let family = bundle.info().model_family.clone();
    let model_id = bundle.info().model_id.clone();
    let runtime = runtime_config_from_env()?;

    let sdk = new_sdk(&bundle_dir, family.clone(), runtime.clone())?;
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
        let sdk = new_sdk(&bundle_dir, family.clone(), runtime.clone())?;
        measure_once(|| {
            let _ = sdk.embed_text(&texts[0])?;
            Ok::<_, omni_search::Error>(())
        })?
    };
    let cold_image = {
        let sdk = new_sdk(&bundle_dir, family.clone(), runtime.clone())?;
        measure_once(|| {
            let _ = sdk.embed_image_path(&image_paths[0])?;
            Ok::<_, omni_search::Error>(())
        })?
    };

    let warm_text_avg = {
        let sdk = new_sdk(&bundle_dir, family.clone(), runtime.clone())?;
        sdk.preload_text()?;
        let _ = sdk.embed_text(&texts[0])?;
        measure_repeated(repeats, || {
            let _ = sdk.embed_text(&texts[0])?;
            Ok::<_, omni_search::Error>(())
        })?
    };
    let warm_image_avg = {
        let sdk = new_sdk(&bundle_dir, family, runtime.clone())?;
        sdk.preload_image()?;
        let _ = sdk.embed_image_path(&image_paths[0])?;
        measure_repeated(repeats, || {
            let _ = sdk.embed_image_path(&image_paths[0])?;
            Ok::<_, omni_search::Error>(())
        })?
    };
    let warm_image_batch_avg = {
        let sdk = new_sdk(
            &bundle_dir,
            bundle.info().model_family.clone(),
            runtime.clone(),
        )?;
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

fn new_sdk(
    bundle_dir: &Path,
    family: omni_search::ModelFamily,
    runtime: RuntimeConfig,
) -> Result<OmniSearch, omni_search::Error> {
    OmniSearch::new(OmniSearchConfig::from_local_bundle(
        family, bundle_dir, runtime,
    ))
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

fn env_path(name: &str) -> Option<PathBuf> {
    env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn runtime_config_from_env() -> Result<RuntimeConfig, Box<dyn std::error::Error>> {
    let mut runtime = RuntimeConfig::default();
    if let Some(intra_threads) = env_usize("OMNI_INTRA_THREADS")? {
        runtime.intra_threads = intra_threads;
    }
    if let Some(inter_threads) = env_usize("OMNI_INTER_THREADS")? {
        runtime.inter_threads = Some(inter_threads);
    }
    if let Some(fgclip_max_patches) = env_usize("OMNI_FGCLIP_MAX_PATCHES")? {
        runtime.fgclip_max_patches = Some(fgclip_max_patches);
    }
    Ok(runtime)
}

fn env_usize(name: &str) -> Result<Option<usize>, Box<dyn std::error::Error>> {
    let Some(value) = env::var_os(name) else {
        return Ok(None);
    };
    let value = value
        .into_string()
        .map_err(|_| format!("{name} must be valid UTF-8"))?;
    if value.trim().is_empty() {
        return Ok(None);
    }
    let parsed = value
        .parse::<usize>()
        .map_err(|error| format!("failed to parse {name}='{value}' as usize: {error}"))?;
    if parsed == 0 {
        return Err(format!("{name} must be greater than 0").into());
    }
    Ok(Some(parsed))
}

fn list_images(root: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut images = fs::read_dir(root)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| {
                        matches!(
                            ext.to_ascii_lowercase().as_str(),
                            "jpg" | "jpeg" | "png" | "webp" | "bmp"
                        )
                    })
        })
        .collect::<Vec<_>>();
    images.sort();
    if images.is_empty() {
        return Err(format!("no images found under {}", root.display()).into());
    }
    Ok(images)
}
