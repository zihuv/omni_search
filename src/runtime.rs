use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Serialize;

use crate::backend::{EmbeddingBackend, create_backend};
use crate::bundle::{ModelBundle, ModelInfo};
use crate::config::{
    GraphOptimizationLevel, ModelConfig, ModelFamily, ProviderPolicy, RuntimeConfig,
    RuntimeConfigBuilder, RuntimeDevice, SessionPolicy,
};
use crate::embedding::Embedding;
use crate::error::Error;

#[derive(Clone, Debug, Default)]
pub struct RuntimeState {
    pub text_loaded: bool,
    pub image_loaded: bool,
    pub last_text_used_at: Option<Instant>,
    pub last_image_used_at: Option<Instant>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RuntimeSnapshot {
    pub config: RuntimeConfigSnapshot,
    pub summary: RuntimeSummary,
    pub text_session: SessionRuntimeSnapshot,
    pub image_session: SessionRuntimeSnapshot,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RuntimeConfigSnapshot {
    pub requested_device: RuntimeDevice,
    pub provider_policy: ProviderPolicy,
    pub intra_threads: usize,
    pub inter_threads: Option<usize>,
    pub fgclip_max_patches: Option<usize>,
    pub session_policy: SessionPolicy,
    pub graph_optimization_level: GraphOptimizationLevel,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RuntimeSummary {
    pub mode: RuntimeMode,
    pub effective_provider: Option<ExecutionProviderKind>,
    pub reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SessionRuntimeSnapshot {
    pub loaded: bool,
    pub planned_providers: Vec<ExecutionProviderKind>,
    pub provider_attempts: Vec<ProviderAttempt>,
    pub registered_providers: Vec<ExecutionProviderKind>,
    pub effective_provider: Option<ExecutionProviderKind>,
    pub mode: RuntimeMode,
    pub fallback_to_cpu: bool,
    pub last_error: Option<RuntimeIssue>,
    pub last_loaded_at_unix_ms: Option<u64>,
    pub last_used_at_unix_ms: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ProviderAttempt {
    pub provider: ExecutionProviderKind,
    pub state: ProviderAttemptState,
    pub detail: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderAttemptState {
    Registered,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RuntimeIssue {
    pub code: RuntimeIssueCode,
    pub message: String,
    pub at_unix_ms: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeIssueCode {
    GpuExecutionUnavailable,
    ProviderRegistrationFailed,
    SessionBuildFailed,
    InferenceFailed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeMode {
    Uninitialized,
    CpuOnly,
    GpuEnabled,
    Mixed,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionProviderKind {
    TensorRt,
    Cuda,
    DirectMl,
    CoreMl,
    Cpu,
}

impl ExecutionProviderKind {
    pub fn is_cpu(self) -> bool {
        matches!(self, Self::Cpu)
    }

    pub fn is_gpu(self) -> bool {
        !self.is_cpu()
    }
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

    pub fn device(&mut self, val: crate::config::RuntimeDevice) -> &mut Self {
        self.runtime.device(val);
        self
    }

    pub fn provider_policy(&mut self, val: crate::config::ProviderPolicy) -> &mut Self {
        self.runtime.provider_policy(val);
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

    pub fn runtime_snapshot(&self) -> RuntimeSnapshot {
        self.backend.runtime_snapshot()
    }
}

impl RuntimeConfigSnapshot {
    pub(crate) fn from_runtime_config(config: &RuntimeConfig) -> Self {
        Self {
            requested_device: config.device,
            provider_policy: config.provider_policy,
            intra_threads: config.intra_threads,
            inter_threads: config.inter_threads,
            fgclip_max_patches: config.fgclip_max_patches,
            session_policy: config.session_policy,
            graph_optimization_level: config.graph_optimization_level,
        }
    }
}

impl SessionRuntimeSnapshot {
    pub(crate) fn infer_mode(&self) -> RuntimeMode {
        if self.loaded {
            match self.effective_provider {
                Some(provider) if provider.is_cpu() => RuntimeMode::CpuOnly,
                Some(provider) if provider.is_gpu() => RuntimeMode::GpuEnabled,
                None => RuntimeMode::Unknown,
                _ => RuntimeMode::Unknown,
            }
        } else if self.last_error.is_some() || !self.provider_attempts.is_empty() {
            RuntimeMode::Unknown
        } else {
            RuntimeMode::Uninitialized
        }
    }
}

pub(crate) fn build_runtime_snapshot(
    config: &RuntimeConfig,
    text_session: SessionRuntimeSnapshot,
    image_session: SessionRuntimeSnapshot,
) -> RuntimeSnapshot {
    let summary = summarize_runtime(&text_session, &image_session);
    RuntimeSnapshot {
        config: RuntimeConfigSnapshot::from_runtime_config(config),
        summary,
        text_session,
        image_session,
    }
}

fn summarize_runtime(
    text_session: &SessionRuntimeSnapshot,
    image_session: &SessionRuntimeSnapshot,
) -> RuntimeSummary {
    let known_sessions = [text_session, image_session]
        .into_iter()
        .filter(|session| session.loaded)
        .collect::<Vec<_>>();
    if known_sessions.is_empty() {
        let any_error = [text_session, image_session]
            .into_iter()
            .find_map(|session| {
                session
                    .last_error
                    .as_ref()
                    .map(|issue| issue.message.clone())
            });
        return RuntimeSummary {
            mode: if any_error.is_some() {
                RuntimeMode::Unknown
            } else {
                RuntimeMode::Uninitialized
            },
            effective_provider: None,
            reason: any_error,
        };
    }

    let distinct_providers = known_sessions
        .iter()
        .filter_map(|session| session.effective_provider)
        .collect::<Vec<_>>();
    let mode = if distinct_providers.is_empty() {
        RuntimeMode::Unknown
    } else if distinct_providers.iter().all(|provider| provider.is_cpu()) {
        RuntimeMode::CpuOnly
    } else if distinct_providers
        .iter()
        .all(|provider| provider == &distinct_providers[0])
    {
        RuntimeMode::GpuEnabled
    } else {
        RuntimeMode::Mixed
    };
    let effective_provider = if distinct_providers.is_empty() {
        None
    } else if distinct_providers
        .iter()
        .all(|provider| provider == &distinct_providers[0])
    {
        Some(distinct_providers[0])
    } else {
        None
    };
    let reason = if mode == RuntimeMode::Mixed {
        Some(format!(
            "text_session={:?}, image_session={:?}",
            text_session.effective_provider, image_session.effective_provider
        ))
    } else {
        [text_session, image_session]
            .into_iter()
            .find_map(|session| {
                session
                    .last_error
                    .as_ref()
                    .map(|issue| issue.message.clone())
            })
    };
    RuntimeSummary {
        mode,
        effective_provider,
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ExecutionProviderKind, OmniSearch, ProviderAttempt, ProviderAttemptState, RuntimeIssue,
        RuntimeIssueCode, RuntimeMode, SessionRuntimeSnapshot, summarize_runtime,
    };

    #[test]
    fn builder_requires_model_source() {
        let error = OmniSearch::builder().build().err().unwrap();
        assert!(
            error
                .to_string()
                .contains("omni search builder is missing a model source")
        );
    }

    #[test]
    fn summarize_runtime_reports_mixed_when_sessions_use_different_providers() {
        let text = SessionRuntimeSnapshot {
            loaded: true,
            planned_providers: vec![ExecutionProviderKind::DirectMl, ExecutionProviderKind::Cpu],
            provider_attempts: vec![ProviderAttempt {
                provider: ExecutionProviderKind::DirectMl,
                state: ProviderAttemptState::Registered,
                detail: None,
            }],
            registered_providers: vec![ExecutionProviderKind::DirectMl],
            effective_provider: Some(ExecutionProviderKind::DirectMl),
            mode: RuntimeMode::GpuEnabled,
            fallback_to_cpu: false,
            last_error: None,
            last_loaded_at_unix_ms: Some(1),
            last_used_at_unix_ms: Some(2),
        };
        let image = SessionRuntimeSnapshot {
            loaded: true,
            planned_providers: vec![ExecutionProviderKind::DirectMl, ExecutionProviderKind::Cpu],
            provider_attempts: vec![ProviderAttempt {
                provider: ExecutionProviderKind::Cpu,
                state: ProviderAttemptState::Registered,
                detail: None,
            }],
            registered_providers: vec![ExecutionProviderKind::Cpu],
            effective_provider: Some(ExecutionProviderKind::Cpu),
            mode: RuntimeMode::CpuOnly,
            fallback_to_cpu: true,
            last_error: Some(RuntimeIssue {
                code: RuntimeIssueCode::ProviderRegistrationFailed,
                message: String::from("DirectML failed"),
                at_unix_ms: 3,
            }),
            last_loaded_at_unix_ms: Some(3),
            last_used_at_unix_ms: Some(4),
        };

        let summary = summarize_runtime(&text, &image);

        assert_eq!(summary.mode, RuntimeMode::Mixed);
        assert_eq!(summary.effective_provider, None);
        assert!(summary.reason.unwrap().contains("text_session"));
    }

    #[test]
    fn session_mode_is_unknown_after_failed_load() {
        let snapshot = SessionRuntimeSnapshot {
            loaded: false,
            planned_providers: vec![ExecutionProviderKind::DirectMl, ExecutionProviderKind::Cpu],
            provider_attempts: vec![ProviderAttempt {
                provider: ExecutionProviderKind::DirectMl,
                state: ProviderAttemptState::Failed,
                detail: Some(String::from("missing dependency")),
            }],
            registered_providers: Vec::new(),
            effective_provider: None,
            mode: RuntimeMode::Unknown,
            fallback_to_cpu: false,
            last_error: Some(RuntimeIssue {
                code: RuntimeIssueCode::SessionBuildFailed,
                message: String::from("session creation failed"),
                at_unix_ms: 10,
            }),
            last_loaded_at_unix_ms: None,
            last_used_at_unix_ms: None,
        };

        assert_eq!(snapshot.infer_mode(), RuntimeMode::Unknown);
    }
}
