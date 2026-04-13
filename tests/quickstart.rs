use std::fs;
use std::path::PathBuf;

use omni_search::{
    ModelBundle, ModelFamily, OmniSearch, OmniSearchConfig, RuntimeConfig, cosine_similarity,
    score_embeddings, top_k,
};

#[test]
fn fgclip_quickstart_smoke() -> Result<(), Box<dyn std::error::Error>> {
    let root = project_root();
    let bundle_dir = root.join("models/fgclip2_bundle");
    let image_path = first_image(root.join("samples"))?;

    let bundle = ModelBundle::load_from_dir(&bundle_dir)?;
    let sdk = OmniSearch::new(OmniSearchConfig::from_local_bundle(
        ModelFamily::FgClip,
        &bundle_dir,
        RuntimeConfig::default(),
    ))?;

    let query = "山".to_owned();
    let text = sdk.embed_text(&query)?;
    let texts = sdk.embed_texts(&[query.clone(), "海边".to_owned()])?;
    let image_from_path = sdk.embed_image_path(&image_path)?;
    let image_bytes = fs::read(&image_path)?;
    let image_from_bytes = sdk.embed_image_bytes(&image_bytes)?;
    let image_batch = sdk.embed_image_paths(std::slice::from_ref(&image_path))?;
    let image_query = sdk.embed_image_path(&image_path)?;

    let score = score_embeddings(&text, &image_from_path)?;
    let cosine = cosine_similarity(text.as_slice(), image_from_path.as_slice())?;
    let ranking = top_k(
        text.as_slice(),
        vec![
            ("same_text", texts[0].clone()),
            ("image_path", image_from_path.clone()),
            ("image_bytes", image_from_bytes.clone()),
        ],
        2,
    )?;
    let image_ranking = top_k(
        image_query.as_slice(),
        vec![(image_path.clone(), image_query.clone())],
        1,
    )?;

    assert_eq!(bundle.info().model_id, "fgclip2-base");
    assert_eq!(text.dims, 768);
    assert_eq!(texts.len(), 2);
    assert_eq!(image_batch.len(), 1);
    assert!((score - cosine).abs() < 1e-6);
    assert!(!ranking.is_empty());
    assert!(!image_ranking.is_empty());

    sdk.unload_all();
    Ok(())
}

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn first_image(samples_dir: PathBuf) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut images = fs::read_dir(samples_dir)?
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
    images
        .into_iter()
        .next()
        .ok_or_else(|| "no sample image found".into())
}
