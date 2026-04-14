use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::backend::{EmbeddingBackend, create_backend};
use crate::bundle::{ModelBundle, ModelInfo};
use crate::config::{ModelConfig, ModelFamily, RuntimeConfig, RuntimeConfigBuilder};
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

#[derive(Clone, Debug)]
enum ModelSelection {
    Config(ModelConfig),
    LocalModelDir(PathBuf),
}

#[derive(Clone, Debug, Default)]
pub struct OmniSearchBuilder {
    model: Option<ModelSelection>,
    runtime: RuntimeConfigBuilder,
}

impl OmniSearchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn model(&mut self, model: ModelConfig) -> &mut Self {
        self.model = Some(ModelSelection::Config(model));
        self
    }

    pub fn from_local_bundle(
        &mut self,
        family: ModelFamily,
        path: impl Into<PathBuf>,
    ) -> &mut Self {
        self.model(ModelConfig::from_local_bundle(family, path))
    }

    pub fn from_local_model_dir(&mut self, path: impl Into<PathBuf>) -> &mut Self {
        self.model = Some(ModelSelection::LocalModelDir(path.into()));
        self
    }

    pub fn runtime_config(&mut self, runtime: RuntimeConfig) -> &mut Self {
        self.runtime = RuntimeConfigBuilder::from_config(runtime);
        self
    }

    pub fn intra_threads(&mut self, val: usize) -> &mut Self {
        self.runtime.intra_threads(val);
        self
    }

    pub fn inter_threads(&mut self, val: usize) -> &mut Self {
        self.runtime.inter_threads(val);
        self
    }

    pub fn clear_inter_threads(&mut self) -> &mut Self {
        self.runtime.clear_inter_threads();
        self
    }

    pub fn fgclip_max_patches(&mut self, val: usize) -> &mut Self {
        self.runtime.fgclip_max_patches(val);
        self
    }

    pub fn clear_fgclip_max_patches(&mut self) -> &mut Self {
        self.runtime.clear_fgclip_max_patches();
        self
    }

    pub fn session_policy(&mut self, val: crate::config::SessionPolicy) -> &mut Self {
        self.runtime.session_policy(val);
        self
    }

    pub fn graph_optimization_level(
        &mut self,
        val: crate::config::GraphOptimizationLevel,
    ) -> &mut Self {
        self.runtime.graph_optimization_level(val);
        self
    }

    pub fn build(&mut self) -> Result<OmniSearch, Error> {
        let runtime = self.runtime.build()?;
        match self.model.clone() {
            Some(ModelSelection::Config(model)) => {
                OmniSearch::new(crate::config::OmniSearchConfig { model, runtime })
            }
            Some(ModelSelection::LocalModelDir(path)) => {
                OmniSearch::from_local_model_dir(path, runtime)
            }
            None => Err(Error::invalid_config(
                "omni search builder is missing a model source",
            )),
        }
    }
}

impl OmniSearch {
    pub fn builder() -> OmniSearchBuilder {
        OmniSearchBuilder::new()
    }

    pub fn new(config: crate::config::OmniSearchConfig) -> Result<Self, Error> {
        config.runtime.validate()?;
        let bundle = ModelBundle::load_for_config(&config.model)?;
        Self::from_loaded_bundle(bundle, config.runtime)
    }

    pub fn from_local_model_dir(
        path: impl AsRef<Path>,
        runtime: RuntimeConfig,
    ) -> Result<Self, Error> {
        runtime.validate()?;
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

#[cfg(test)]
mod tests {
    use super::OmniSearch;

    #[test]
    fn builder_requires_model_source() {
        let error = OmniSearch::builder().build().err().unwrap();
        assert!(
            error
                .to_string()
                .contains("omni search builder is missing a model source")
        );
    }
}
