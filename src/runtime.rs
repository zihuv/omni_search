use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::backend::{EmbeddingBackend, create_backend};
use crate::bundle::{ModelBundle, ModelInfo};
use crate::config::{OmniSearchConfig, RuntimeConfig};
use crate::embedding::Embedding;
use crate::error::Error;

#[derive(Clone, Debug, Default)]
pub struct RuntimeState {
    pub text_loaded: bool,
    pub image_loaded: bool,
    pub last_text_used_at: Option<Instant>,
    pub last_image_used_at: Option<Instant>,
}

pub struct OmniSearch {
    model_info: ModelInfo,
    backend: Box<dyn EmbeddingBackend + Send>,
}

impl OmniSearch {
    pub fn new(config: OmniSearchConfig) -> Result<Self, Error> {
        validate_runtime_config(&config.runtime)?;
        let bundle = ModelBundle::load_for_config(&config.model)?;
        Self::from_loaded_bundle(bundle, config.runtime)
    }

    pub fn from_local_model_dir(
        path: impl AsRef<Path>,
        runtime: RuntimeConfig,
    ) -> Result<Self, Error> {
        validate_runtime_config(&runtime)?;
        let bundle = ModelBundle::load_from_dir(path)?;
        Self::from_loaded_bundle(bundle, runtime)
    }

    fn from_loaded_bundle(bundle: ModelBundle, runtime: RuntimeConfig) -> Result<Self, Error> {
        let model_info = bundle.info().clone();
        let backend = create_backend(bundle, runtime)?;
        Ok(Self {
            model_info,
            backend,
        })
    }

    pub fn model_info(&self) -> &ModelInfo {
        &self.model_info
    }

    pub fn embed_text(&self, text: &str) -> Result<Embedding, Error> {
        self.backend.embed_text(text)
    }

    pub fn embed_texts(&self, texts: &[String]) -> Result<Vec<Embedding>, Error> {
        self.backend.embed_texts(texts)
    }

    pub fn embed_image_path(&self, path: impl AsRef<Path>) -> Result<Embedding, Error> {
        self.backend.embed_image_path(path.as_ref())
    }

    pub fn embed_image_bytes(&self, bytes: &[u8]) -> Result<Embedding, Error> {
        self.backend.embed_image_bytes(bytes)
    }

    pub fn embed_image_paths(&self, paths: &[PathBuf]) -> Result<Vec<Embedding>, Error> {
        self.backend.embed_image_paths(paths)
    }

    pub fn preload_text(&self) -> Result<(), Error> {
        self.backend.preload_text()
    }

    pub fn preload_image(&self) -> Result<(), Error> {
        self.backend.preload_image()
    }

    pub fn unload_text(&self) -> bool {
        self.backend.unload_text()
    }

    pub fn unload_image(&self) -> bool {
        self.backend.unload_image()
    }

    pub fn unload_all(&self) -> usize {
        let mut unloaded = 0;
        if self.unload_text() {
            unloaded += 1;
        }
        if self.unload_image() {
            unloaded += 1;
        }
        unloaded
    }

    pub fn runtime_state(&self) -> RuntimeState {
        self.backend.runtime_state()
    }
}

fn validate_runtime_config(runtime: &RuntimeConfig) -> Result<(), Error> {
    if runtime.intra_threads == 0 {
        return Err(Error::invalid_config(
            "runtime.intra_threads must be greater than 0",
        ));
    }
    if matches!(runtime.inter_threads, Some(0)) {
        return Err(Error::invalid_config(
            "runtime.inter_threads must be greater than 0 when set",
        ));
    }
    Ok(())
}
