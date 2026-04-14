use std::fs;
use std::path::PathBuf;

use omni_search::{
    ModelBundle, OmniSearch, cosine_similarity, probe_local_model_dir, score_embeddings, top_k,
};

#[test]
#[ignore = "requires local OMNI_TEST_BUNDLE_DIR and OMNI_TEST_SAMPLE_IMAGE fixtures"]
fn fgclip_quickstart_smoke() -> Result<(), Box<dyn std::error::Error>> {
    let bundle_dir = bundle_dir()?;
    let image_path = sample_image()?;

    let bundle = ModelBundle::load_from_dir(&bundle_dir)?;
    let probe = probe_local_model_dir(&bundle_dir);
    assert!(probe.ok);
    let sdk = OmniSearch::builder()
        .from_local_model_dir(&bundle_dir)
        .build()?;

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

fn bundle_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    env_path("OMNI_TEST_BUNDLE_DIR").ok_or_else(|| {
        "set OMNI_TEST_BUNDLE_DIR to a local model bundle before running this ignored test".into()
    })
}

fn sample_image() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let path = env_path("OMNI_TEST_SAMPLE_IMAGE").ok_or_else(|| {
        "set OMNI_TEST_SAMPLE_IMAGE to a local image before running this ignored test".to_owned()
    })?;
    if !path.is_file() {
        return Err(format!(
            "OMNI_TEST_SAMPLE_IMAGE does not point to a file: {}",
            path.display()
        )
        .into());
    }
    Ok(path)
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}
