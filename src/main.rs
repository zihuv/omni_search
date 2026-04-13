use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use omni_search::{ModelBundle, ModelFamily, OmniSearch, OmniSearchConfig, RuntimeConfig, top_k};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = project_root();
    let bundle_dir = root.join("models/fgclip2_bundle");
    let samples_dir = root.join("samples");
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
    let sdk = OmniSearch::new(OmniSearchConfig::from_local_bundle(
        ModelFamily::FgClip,
        &bundle_dir,
        RuntimeConfig::default(),
    ))?;

    println!("bundle: {}", bundle_dir.display());
    println!("model: {:?}", bundle.info());
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
