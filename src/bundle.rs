use std::fs;
use std::path::{Path, PathBuf};

use crate::config::{ModelConfig, ModelFamily, ModelSource, ModelSourceKind};
use crate::error::Error;
use crate::manifest::{ImagePreprocessConfig, ModelManifest};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelInfo {
    pub model_family: ModelFamily,
    pub source_kind: ModelSourceKind,
    pub model_id: String,
    pub model_revision: String,
    pub embedding_dim: usize,
    pub context_length: usize,
    pub normalize_output: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalModelDirProbe {
    pub ok: bool,
    pub normalized_path: PathBuf,
    pub manifest_path: PathBuf,
    pub source_kind: Option<ModelSourceKind>,
    pub family: Option<ModelFamily>,
    pub model_id: Option<String>,
    pub model_revision: Option<String>,
    pub embedding_dim: Option<usize>,
    pub context_length: Option<usize>,
    pub missing_files: Vec<PathBuf>,
    pub warnings: Vec<String>,
    pub error: Option<String>,
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
            source_kind: ModelSourceKind::LocalBundleDir,
            model_id: manifest.model_id.clone(),
            model_revision: manifest.model_revision.clone(),
            embedding_dim: manifest.embedding_dim,
            context_length: manifest.text.context_length,
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

    pub fn probe_dir(path: impl AsRef<Path>) -> LocalModelDirProbe {
        probe_local_model_dir(path)
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

pub fn probe_local_model_dir(path: impl AsRef<Path>) -> LocalModelDirProbe {
    let normalized_path = absolutize(path.as_ref()).unwrap_or_else(|_| path.as_ref().to_path_buf());
    let manifest_path = normalized_path.join("manifest.json");

    let mut probe = LocalModelDirProbe {
        ok: false,
        normalized_path: normalized_path.clone(),
        manifest_path: manifest_path.clone(),
        source_kind: None,
        family: None,
        model_id: None,
        model_revision: None,
        embedding_dim: None,
        context_length: None,
        missing_files: Vec::new(),
        warnings: Vec::new(),
        error: None,
    };

    if !normalized_path.is_dir() {
        probe.error = Some(format!(
            "bundle root is not a directory: {}",
            normalized_path.display()
        ));
        return probe;
    }

    let manifest_bytes = match fs::read(&manifest_path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            probe.missing_files.push(manifest_path);
            probe.error = Some("missing manifest.json".to_owned());
            return probe;
        }
        Err(error) => {
            probe.error = Some(format!(
                "failed to read manifest {}: {error}",
                manifest_path.display()
            ));
            return probe;
        }
    };

    let manifest: ModelManifest = match serde_json::from_slice(&manifest_bytes) {
        Ok(manifest) => manifest,
        Err(error) => {
            probe.error = Some(format!(
                "failed to parse manifest {}: {error}",
                manifest_path.display()
            ));
            return probe;
        }
    };

    probe.source_kind = Some(ModelSourceKind::LocalBundleDir);
    probe.family = Some(manifest.family.clone());
    probe.model_id = Some(manifest.model_id.clone());
    probe.model_revision = Some(manifest.model_revision.clone());
    probe.embedding_dim = Some(manifest.embedding_dim);
    probe.context_length = Some(manifest.text.context_length);

    if let Err(error) = manifest.validate() {
        probe.error = Some(error.to_string());
        return probe;
    }

    probe.missing_files = required_bundle_files(&normalized_path, &manifest)
        .into_iter()
        .filter(|path| !path.is_file())
        .collect();

    if !probe.missing_files.is_empty() {
        probe.error = Some("one or more required bundle files are missing".to_owned());
        return probe;
    }

    probe.ok = true;
    probe
}

fn required_bundle_files(root: &Path, manifest: &ModelManifest) -> Vec<PathBuf> {
    let mut files = vec![
        resolve_bundle_path(root, &manifest.text.onnx),
        resolve_bundle_path(root, &manifest.image.onnx),
        resolve_bundle_path(root, &manifest.text.tokenizer),
    ];
    if let Some(token_embedding) = manifest.text.token_embedding.as_ref() {
        files.push(resolve_bundle_path(root, &token_embedding.file));
    }
    if let ImagePreprocessConfig::FgclipPatchTokens {
        vision_pos_embedding,
        ..
    } = &manifest.image.preprocess
    {
        files.push(resolve_bundle_path(root, vision_pos_embedding));
    }
    files
}

fn resolve_bundle_path(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

fn require_file(root: &Path, path: &Path, label: &str) -> Result<PathBuf, Error> {
    let resolved = resolve_bundle_path(root, path);
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

    use super::{ModelBundle, ModelSourceKind, probe_local_model_dir};

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
        assert_eq!(bundle.info().source_kind, ModelSourceKind::LocalBundleDir);
        assert_eq!(bundle.info().embedding_dim, 768);
        assert_eq!(bundle.info().context_length, 64);
    }

    #[test]
    fn probes_fgclip_bundle_layout() {
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

        let probe = probe_local_model_dir(root);
        assert!(probe.ok);
        assert_eq!(probe.source_kind, Some(ModelSourceKind::LocalBundleDir));
        assert_eq!(probe.family, Some(crate::config::ModelFamily::FgClip));
        assert_eq!(probe.model_id.as_deref(), Some("fgclip2-base"));
        assert_eq!(probe.model_revision.as_deref(), Some("2026-04-13"));
        assert_eq!(probe.embedding_dim, Some(768));
        assert_eq!(probe.context_length, Some(64));
        assert!(probe.missing_files.is_empty());
        assert!(probe.error.is_none());
    }

    #[test]
    fn probe_reports_missing_manifest() {
        let dir = tempdir().unwrap();
        let probe = probe_local_model_dir(dir.path());

        assert!(!probe.ok);
        assert_eq!(probe.source_kind, None);
        assert_eq!(probe.family, None);
        assert_eq!(probe.missing_files, vec![dir.path().join("manifest.json")]);
        assert_eq!(probe.error.as_deref(), Some("missing manifest.json"));
    }
}
