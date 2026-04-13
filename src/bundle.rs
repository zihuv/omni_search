use std::fs;
use std::path::{Path, PathBuf};

use crate::config::{ModelConfig, ModelFamily, ModelSource};
use crate::error::Error;
use crate::manifest::{ImagePreprocessConfig, ModelManifest};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelInfo {
    pub model_family: ModelFamily,
    pub model_id: String,
    pub model_revision: String,
    pub embedding_dim: usize,
    pub normalize_output: bool,
}

#[derive(Clone, Debug)]
pub struct ModelBundle {
    root: PathBuf,
    manifest_path: PathBuf,
    manifest: ModelManifest,
    info: ModelInfo,
    text_onnx_path: PathBuf,
    image_onnx_path: PathBuf,
    tokenizer_path: PathBuf,
    token_embedding_path: Option<PathBuf>,
    vision_pos_embedding_path: Option<PathBuf>,
}

impl ModelBundle {
    pub fn load_from_dir(path: impl AsRef<Path>) -> Result<Self, Error> {
        let root = absolutize(path.as_ref())?;
        if !root.is_dir() {
            return Err(Error::invalid_bundle(format!(
                "bundle root is not a directory: {}",
                root.display()
            )));
        }

        let manifest_path = root.join("manifest.json");
        let manifest_bytes = fs::read(&manifest_path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                Error::invalid_bundle(format!("missing {}", manifest_path.display()))
            } else {
                error.into()
            }
        })?;
        let manifest: ModelManifest = serde_json::from_slice(&manifest_bytes).map_err(|error| {
            Error::invalid_bundle(format!(
                "failed to parse manifest {}: {error}",
                manifest_path.display()
            ))
        })?;
        manifest.validate()?;

        let text_onnx_path = require_file(&root, &manifest.text.onnx, "text.onnx")?;
        let image_onnx_path = require_file(&root, &manifest.image.onnx, "image.onnx")?;
        let tokenizer_path = require_file(&root, &manifest.text.tokenizer, "tokenizer")?;
        let token_embedding_path = manifest
            .text
            .token_embedding
            .as_ref()
            .map(|asset| require_file(&root, &asset.file, "text.token_embedding"))
            .transpose()?;
        let vision_pos_embedding_path = match &manifest.image.preprocess {
            ImagePreprocessConfig::FgclipPatchTokens {
                vision_pos_embedding,
                ..
            } => Some(require_file(
                &root,
                vision_pos_embedding,
                "image.preprocess.vision_pos_embedding",
            )?),
            ImagePreprocessConfig::ClipImage { .. } => None,
        };

        let info = ModelInfo {
            model_family: manifest.family.clone(),
            model_id: manifest.model_id.clone(),
            model_revision: manifest.model_revision.clone(),
            embedding_dim: manifest.embedding_dim,
            normalize_output: manifest.normalize_output,
        };

        Ok(Self {
            root,
            manifest_path,
            manifest,
            info,
            text_onnx_path,
            image_onnx_path,
            tokenizer_path,
            token_embedding_path,
            vision_pos_embedding_path,
        })
    }

    pub fn info(&self) -> &ModelInfo {
        &self.info
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn manifest_path(&self) -> &Path {
        &self.manifest_path
    }

    pub fn family(&self) -> &ModelFamily {
        &self.info.model_family
    }

    pub fn model_id(&self) -> &str {
        &self.info.model_id
    }

    pub(crate) fn load_for_config(config: &ModelConfig) -> Result<Self, Error> {
        let bundle = match &config.source {
            ModelSource::LocalBundleDir(path) => Self::load_from_dir(path)?,
        };
        if bundle.info.model_family != config.family {
            return Err(Error::invalid_config(format!(
                "model family mismatch: config={}, bundle={}",
                config.family, bundle.info.model_family
            )));
        }
        Ok(bundle)
    }

    pub(crate) fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    pub(crate) fn text_onnx_path(&self) -> &Path {
        &self.text_onnx_path
    }

    pub(crate) fn image_onnx_path(&self) -> &Path {
        &self.image_onnx_path
    }

    pub(crate) fn tokenizer_path(&self) -> &Path {
        &self.tokenizer_path
    }

    pub(crate) fn token_embedding_path(&self) -> Option<&Path> {
        self.token_embedding_path.as_deref()
    }

    pub(crate) fn vision_pos_embedding_path(&self) -> Option<&Path> {
        self.vision_pos_embedding_path.as_deref()
    }
}

fn require_file(root: &Path, path: &Path, label: &str) -> Result<PathBuf, Error> {
    let resolved = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };
    if !resolved.is_file() {
        return Err(Error::invalid_bundle(format!(
            "{label} file does not exist: {}",
            resolved.display()
        )));
    }
    Ok(resolved)
}

fn absolutize(path: &Path) -> Result<PathBuf, Error> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::ModelBundle;

    #[test]
    fn loads_fgclip_bundle_layout() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        fs::create_dir_all(root.join("onnx")).unwrap();
        fs::create_dir_all(root.join("assets")).unwrap();
        fs::write(root.join("onnx/text.onnx"), []).unwrap();
        fs::write(root.join("onnx/image.onnx"), []).unwrap();
        fs::write(root.join("assets/tokenizer.json"), "{}").unwrap();
        fs::write(root.join("assets/text_token_embedding.bin"), []).unwrap();
        fs::write(root.join("assets/vision_pos_embedding.bin"), []).unwrap();
        fs::write(
            root.join("manifest.json"),
            r#"{
                "schema_version": 1,
                "family": "fg_clip",
                "model_id": "fgclip2-base",
                "model_revision": "2026-04-13",
                "embedding_dim": 768,
                "normalize_output": true,
                "text": {
                  "onnx": "onnx/text.onnx",
                  "output_name": "text_features",
                  "tokenizer": "assets/tokenizer.json",
                  "context_length": 64,
                  "input": { "kind": "token_embeds" },
                  "token_embedding": {
                    "file": "assets/text_token_embedding.bin",
                    "dtype": "f16",
                    "embedding_dim": 768
                  }
                },
                "image": {
                  "onnx": "onnx/image.onnx",
                  "output_name": "image_features",
                  "preprocess": {
                    "kind": "fgclip_patch_tokens",
                    "patch_size": 16,
                    "default_max_patches": 1024,
                    "vision_pos_embedding": "assets/vision_pos_embedding.bin"
                  }
                }
            }"#,
        )
        .unwrap();

        let bundle = ModelBundle::load_from_dir(root).unwrap();
        assert_eq!(bundle.model_id(), "fgclip2-base");
        assert_eq!(bundle.info().embedding_dim, 768);
    }
}
