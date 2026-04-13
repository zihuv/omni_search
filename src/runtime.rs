use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::backend::{EmbeddingBackend, create_backend};
use crate::bundle::{ModelBundle, ModelInfo};
use crate::config::OmniSearchConfig;
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
    backend: Box<dyn EmbeddingBackend>,
}

impl OmniSearch {
    pub fn new(config: OmniSearchConfig) -> Result<Self, Error> {
        if config.runtime.intra_threads == 0 {
            return Err(Error::invalid_config(
                "runtime.intra_threads must be greater than 0",
            ));
        }
        if matches!(config.runtime.inter_threads, Some(0)) {
            return Err(Error::invalid_config(
                "runtime.inter_threads must be greater than 0 when set",
            ));
        }

        let bundle = ModelBundle::load_for_config(&config.model)?;
        let model_info = bundle.info().clone();
        let backend = create_backend(bundle, config.runtime)?;
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
