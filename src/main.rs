use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use omni_search::{ModelBundle, OmniSearch, OmniSearchConfig, RuntimeConfig, top_k};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = project_root();
    let bundle_dir =
        env_path("OMNI_BUNDLE_DIR").unwrap_or_else(|| root.join("models/fgclip2_bundle"));
    let samples_dir = env_path("OMNI_SAMPLES_DIR").unwrap_or_else(|| root.join("samples"));
    let args = env::args().skip(1).collect::<Vec<_>>();
    let (query, top_k_count, query_image_arg) = match args.as_slice() {
        [] => ("山".to_owned(), 10usize, None),
        [query] => (query.clone(), 10usize, None),
        [query, second] => match second.parse::<usize>() {
            Ok(top_k) => (query.clone(), top_k, None),
            Err(_) => (query.clone(), 10usize, Some(PathBuf::from(second))),
        },
        [query, top_k, image, ..] => (
            query.clone(),
            top_k.parse().unwrap_or(10),
            Some(PathBuf::from(image)),
        ),
    };
    let image_paths = list_images(&samples_dir)?;
    let query_image_path = if let Some(query_image) = query_image_arg {
        let query_image = if query_image.is_absolute() {
            query_image
        } else {
            root.join(query_image)
        };
        if !query_image.is_file() {
            return Err(format!("query image not found: {}", query_image.display()).into());
        }
        query_image
    } else {
        image_paths
            .first()
            .cloned()
            .ok_or_else(|| format!("no images found under {}", samples_dir.display()))?
    };

    let bundle = ModelBundle::load_from_dir(&bundle_dir)?;
    let model_family = bundle.info().model_family.clone();
    let runtime = runtime_config_from_env()?;
    let sdk = OmniSearch::new(OmniSearchConfig::from_local_bundle(
        model_family.clone(),
        &bundle_dir,
        runtime.clone(),
    ))?;

    println!("bundle: {}", bundle_dir.display());
    println!("family: {model_family}");
    println!("model: {:?}", bundle.info());
    println!(
        "runtime: intra_threads={}, inter_threads={:?}, fgclip_max_patches={:?}",
        runtime.intra_threads, runtime.inter_threads, runtime.fgclip_max_patches
    );
    println!("samples: {}", samples_dir.display());
    println!("images: {}", image_paths.len());
    println!("text query: {query}");
    println!("image query: {}", query_image_path.display());

    sdk.preload_text()?;
    let text = sdk.embed_text(&query)?;
    let texts = sdk.embed_texts(&[query.clone(), "海边".to_owned()])?;

    sdk.preload_image()?;
    let image_batch = sdk.embed_image_paths(&image_paths)?;
    let image_query = sdk.embed_image_path(&query_image_path)?;

    if let Some(first_image) = image_paths.first() {
        let first_embedding = sdk.embed_image_path(first_image)?;
        let first_bytes = fs::read(first_image)?;
        let first_from_bytes = sdk.embed_image_bytes(&first_bytes)?;
        println!("first image dims(path): {}", first_embedding.dims);
        println!("first image dims(bytes): {}", first_from_bytes.dims);
    }

    // Text->image and image->image both use the same public embedding + top_k flow.
    let text_to_image_ranking = top_k(
        text.as_slice(),
        image_paths.iter().cloned().zip(image_batch.iter().cloned()),
        top_k_count.min(image_batch.len()),
    )?;
    let image_to_image_ranking = top_k(
        image_query.as_slice(),
        image_paths
            .iter()
            .cloned()
            .zip(image_batch.iter().cloned())
            .filter(|(path, _)| path != &query_image_path),
        top_k_count.min(image_batch.len().saturating_sub(1)),
    )?;

    println!("text dims: {}", text.dims);
    println!("embed_texts count: {}", texts.len());
    println!("embed_image_paths count: {}", image_batch.len());
    println!("text_to_image top_k:");
    for (index, item) in text_to_image_ranking.iter().enumerate() {
        println!(
            "  {:02}. {:.9}  {}",
            index + 1,
            item.score,
            item.item.display()
        );
    }
    println!("image_to_image top_k:");
    for (index, item) in image_to_image_ranking.iter().enumerate() {
        println!(
            "  {:02}. {:.9}  {}",
            index + 1,
            item.score,
            item.item.display()
        );
    }

    sdk.unload_all();
    Ok(())
}

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
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
    let mut images = Vec::new();
    collect_images(root, &mut images)?;
    images.sort();
    if images.is_empty() {
        return Err(format!("no images found under {}", root.display()).into());
    }
    Ok(images)
}

fn collect_images(
    root: &Path,
    images: &mut Vec<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    for entry in fs::read_dir(root)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_images(&path, images)?;
            continue;
        }
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| {
                matches!(
                    ext.to_ascii_lowercase().as_str(),
                    "jpg" | "jpeg" | "png" | "webp" | "bmp"
                )
            })
        {
            images.push(path);
        }
    }
    Ok(())
}
