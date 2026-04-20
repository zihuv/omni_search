mod chinese_clip;
mod fgclip;
mod openclip;

use std::env;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use ndarray::ArrayD;
use ort::ep::{self, ExecutionProvider};
use ort::session::{
    Session, builder::GraphOptimizationLevel as OrtGraphOptimizationLevel, builder::SessionBuilder,
};
use tokenizers::decoders::wordpiece::WordPiece as WordPieceDecoder;
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::bert::BertNormalizer;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::processors::bert::BertProcessing;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::bundle::{ModelBundle, ModelInfo};
use crate::config::{
    GraphOptimizationLevel, ModelFamily, ProviderPolicy, RuntimeConfig, RuntimeDevice,
    RuntimeLibraryConfig, RuntimeProfileKind,
};
use crate::embedding::Embedding;
use crate::error::Error;
use crate::runtime::{
    ExecutionProviderKind, ProviderAttempt, ProviderAttemptState, RuntimeIssue, RuntimeIssueCode,
    RuntimeLibraryConfigSnapshot, RuntimePlanProfileSnapshot, RuntimeSnapshot, RuntimeState,
    SessionRuntimeSnapshot, build_runtime_snapshot,
    looks_like_missing_dependency, looks_like_provider_library_incompatible,
    looks_like_provider_unsupported_by_runtime_library,
};
use crate::runtime_loader::prepare_runtime_libraries;

const FORCE_PROVIDER_ENV: &str = "OMNI_FORCE_PROVIDER";

pub(crate) use chinese_clip::ChineseClipBackend;
pub(crate) use fgclip::FgClipBackend;
pub(crate) use openclip::OpenClipBackend;

pub(crate) trait EmbeddingBackend: Send {
    fn embed_text(&self, text: &str) -> Result<Embedding, Error>;
    fn embed_texts(&self, texts: &[String]) -> Result<Vec<Embedding>, Error>;
    fn embed_image_path(&self, path: &Path) -> Result<Embedding, Error>;
    fn embed_image_bytes(&self, bytes: &[u8]) -> Result<Embedding, Error>;
    fn embed_image_paths(&self, paths: &[PathBuf]) -> Result<Vec<Embedding>, Error>;
    fn preload_text(&self) -> Result<(), Error>;
    fn preload_image(&self) -> Result<(), Error>;
    fn unload_text(&self) -> bool;
    fn unload_image(&self) -> bool;
    fn runtime_state(&self) -> RuntimeState;
    fn runtime_snapshot(&self) -> RuntimeSnapshot;
}

pub(crate) fn create_backend(
    bundle: ModelBundle,
    runtime: RuntimeConfig,
) -> Result<Box<dyn EmbeddingBackend + Send>, Error> {
    match bundle.info().model_family {
        ModelFamily::FgClip => Ok(Box::new(FgClipBackend::new(bundle, runtime)?)),
        ModelFamily::ChineseClip => Ok(Box::new(ChineseClipBackend::new(bundle, runtime)?)),
        ModelFamily::OpenClip => Ok(Box::new(OpenClipBackend::new(bundle, runtime)?)),
    }
}

pub(crate) struct LazySession {
    model_path: PathBuf,
    runtime: RuntimeConfig,
    state: Mutex<SessionState>,
}

struct SessionState {
    session: Option<Session>,
    selected_profile: Option<RuntimeProfileKind>,
    last_used_at: Option<Instant>,
    last_used_at_unix_ms: Option<u64>,
    last_loaded_at_unix_ms: Option<u64>,
    provider_attempts: Vec<ProviderAttempt>,
    registered_providers: Vec<ExecutionProviderKind>,
    issues: Vec<RuntimeIssue>,
    last_error: Option<RuntimeIssue>,
}

struct SessionLoadSuccess {
    session: Session,
    selected_profile: RuntimeProfileKind,
    provider_attempts: Vec<ProviderAttempt>,
    registered_providers: Vec<ExecutionProviderKind>,
    issues: Vec<RuntimeIssue>,
    loaded_at_unix_ms: u64,
}

struct SessionLoadFailure {
    error: Error,
    provider_attempts: Vec<ProviderAttempt>,
    registered_providers: Vec<ExecutionProviderKind>,
    issues: Vec<RuntimeIssue>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RuntimeProfilePlan {
    kind: RuntimeProfileKind,
    providers: Vec<ExecutionProviderKind>,
    library: RuntimeLibraryConfig,
}

impl LazySession {
    pub(crate) fn new(model_path: PathBuf, runtime: RuntimeConfig) -> Self {
        Self {
            model_path,
            runtime,
            state: Mutex::new(SessionState {
                session: None,
                selected_profile: None,
                last_used_at: None,
                last_used_at_unix_ms: None,
                last_loaded_at_unix_ms: None,
                provider_attempts: Vec::new(),
                registered_providers: Vec::new(),
                issues: Vec::new(),
                last_error: None,
            }),
        }
    }

    pub(crate) fn ensure_loaded(&self) -> Result<(), Error> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.session.is_none() {
            self.load_into_state(&mut state)?;
        }
        Ok(())
    }

    pub(crate) fn with_session<T>(
        &self,
        f: impl FnOnce(&mut Session) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.session.is_none() {
            self.load_into_state(&mut state)?;
        }
        match f(state.session.as_mut().expect("session must be loaded")) {
            Ok(result) => {
                state.last_used_at = Some(Instant::now());
                state.last_used_at_unix_ms = Some(now_unix_ms());
                Ok(result)
            }
            Err(error) => {
                let issue = RuntimeIssue {
                    code: RuntimeIssueCode::InferenceFailed,
                    message: error.to_string(),
                    at_unix_ms: now_unix_ms(),
                };
                state.issues.push(issue.clone());
                state.last_error = Some(issue);
                Err(error)
            }
        }
    }

    pub(crate) fn unload(&self) -> bool {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.session.take().is_some()
    }

    pub(crate) fn is_loaded(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .session
            .is_some()
    }

    pub(crate) fn last_used_at(&self) -> Option<Instant> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .last_used_at
    }

    pub(crate) fn runtime_snapshot(&self) -> SessionRuntimeSnapshot {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let compiled_providers = compiled_provider_kinds();
        let planned_providers = planned_provider_kinds(&self.runtime);
        let loaded = state.session.is_some();
        let effective_provider = if loaded {
            state.registered_providers.first().copied()
        } else {
            None
        };
        let mut snapshot = SessionRuntimeSnapshot {
            loaded,
            compiled_providers,
            planned_providers: planned_providers.clone(),
            selected_profile: state.selected_profile,
            provider_attempts: state.provider_attempts.clone(),
            registered_providers: state.registered_providers.clone(),
            effective_provider,
            mode: crate::runtime::RuntimeMode::Unknown,
            fallback_to_cpu: loaded
                && self.runtime.device != RuntimeDevice::Cpu
                && planned_providers.iter().any(|provider| provider.is_gpu())
                && effective_provider == Some(ExecutionProviderKind::Cpu),
            issues: state.issues.clone(),
            last_error: state.last_error.clone(),
            last_loaded_at_unix_ms: state.last_loaded_at_unix_ms,
            last_used_at_unix_ms: state.last_used_at_unix_ms,
        };
        snapshot.mode = snapshot.infer_mode();
        snapshot
    }

    fn load_into_state(&self, state: &mut SessionState) -> Result<(), Error> {
        match load_session(&self.model_path, &self.runtime) {
            Ok(loaded) => {
                state.session = Some(loaded.session);
                state.selected_profile = Some(loaded.selected_profile);
                state.provider_attempts = loaded.provider_attempts;
                state.registered_providers = loaded.registered_providers;
                state.issues = loaded.issues;
                state.last_loaded_at_unix_ms = Some(loaded.loaded_at_unix_ms);
                state.last_error = state.issues.last().cloned();
                Ok(())
            }
            Err(failure) => {
                state.session = None;
                state.selected_profile = None;
                state.provider_attempts = failure.provider_attempts;
                state.registered_providers = failure.registered_providers;
                state.issues = failure.issues;
                state.last_error = state.issues.last().cloned();
                Err(failure.error)
            }
        }
    }
}

pub(crate) fn runtime_snapshot_for_sessions(
    runtime: &RuntimeConfig,
    text_session: &LazySession,
    image_session: &LazySession,
) -> RuntimeSnapshot {
    build_runtime_snapshot(
        runtime,
        text_session.runtime_snapshot(),
        image_session.runtime_snapshot(),
    )
}

pub(crate) fn load_tokenizer(
    tokenizer_path: &Path,
    max_len: usize,
    fallback_pad_token: &str,
) -> Result<Tokenizer, Error> {
    let mut tokenizer = load_tokenizer_from_path(tokenizer_path)?;
    let pad_id = tokenizer.token_to_id(fallback_pad_token).unwrap_or(0);
    apply_tokenizer_truncation_and_padding(
        &mut tokenizer,
        max_len,
        pad_id,
        fallback_pad_token.to_owned(),
    )?;
    Ok(tokenizer)
}

pub(crate) fn load_tokenizer_with_pad_id(
    tokenizer_path: &Path,
    max_len: usize,
    pad_id: u32,
) -> Result<Tokenizer, Error> {
    let mut tokenizer = load_tokenizer_from_path(tokenizer_path)?;
    let pad_token = tokenizer.id_to_token(pad_id).ok_or_else(|| {
        Error::tokenizer(format!(
            "{} does not contain pad token id {pad_id}",
            tokenizer_path.display()
        ))
    })?;
    apply_tokenizer_truncation_and_padding(&mut tokenizer, max_len, pad_id, pad_token)?;
    Ok(tokenizer)
}

fn apply_tokenizer_truncation_and_padding(
    tokenizer: &mut Tokenizer,
    max_len: usize,
    pad_id: u32,
    pad_token: String,
) -> Result<(), Error> {
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: max_len,
            ..Default::default()
        }))
        .map_err(Error::from_tokenizer)?;

    let mut padding = tokenizer
        .get_padding()
        .cloned()
        .unwrap_or_else(|| PaddingParams {
            pad_id,
            pad_type_id: 0,
            pad_token: pad_token.clone(),
            ..Default::default()
        });
    padding.pad_id = pad_id;
    padding.pad_token = pad_token;
    padding.strategy = PaddingStrategy::Fixed(max_len);
    tokenizer.with_padding(Some(padding));
    Ok(())
}

fn load_tokenizer_from_path(tokenizer_path: &Path) -> Result<Tokenizer, Error> {
    if tokenizer_path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("txt"))
    {
        return build_bert_tokenizer_from_vocab(tokenizer_path);
    }

    Tokenizer::from_file(tokenizer_path)
        .map_err(Error::from_tokenizer)
        .map_err(|error| Error::tokenizer(format!("{}: {error}", tokenizer_path.display())))
}

fn build_bert_tokenizer_from_vocab(tokenizer_path: &Path) -> Result<Tokenizer, Error> {
    let tokenizer_path_str = tokenizer_path.to_str().ok_or_else(|| {
        Error::tokenizer(format!(
            "tokenizer path is not valid UTF-8: {}",
            tokenizer_path.display()
        ))
    })?;
    let wordpiece = WordPiece::from_file(tokenizer_path_str)
        .unk_token("[UNK]".to_owned())
        .build()
        .map_err(Error::from_tokenizer)
        .map_err(|error| Error::tokenizer(format!("{}: {error}", tokenizer_path.display())))?;
    let mut tokenizer = Tokenizer::new(wordpiece);
    let sep = tokenizer.token_to_id("[SEP]").ok_or_else(|| {
        Error::tokenizer(format!("{} is missing [SEP]", tokenizer_path.display()))
    })?;
    let cls = tokenizer.token_to_id("[CLS]").ok_or_else(|| {
        Error::tokenizer(format!("{} is missing [CLS]", tokenizer_path.display()))
    })?;
    tokenizer
        .with_normalizer(Some(BertNormalizer::default()))
        .with_pre_tokenizer(Some(BertPreTokenizer))
        .with_decoder(Some(WordPieceDecoder::default()))
        .with_post_processor(Some(BertProcessing::new(
            ("[SEP]".to_owned(), sep),
            ("[CLS]".to_owned(), cls),
        )));
    Ok(tokenizer)
}

pub(crate) fn embeddings_from_output(
    info: &ModelInfo,
    output: ArrayD<f32>,
    normalize_output: bool,
) -> Result<Vec<Embedding>, Error> {
    match output.ndim() {
        1 => {
            let mut values = output.iter().copied().collect::<Vec<_>>();
            if normalize_output {
                normalize_vector(&mut values)?;
            }
            Ok(vec![Embedding::from_vec(info, values)?])
        }
        2 => {
            let shape = output.shape().to_vec();
            let dims = shape[1];
            if dims != info.embedding_dim {
                return Err(Error::DimensionMismatch {
                    expected: info.embedding_dim,
                    actual: dims,
                });
            }

            let mut embeddings = Vec::with_capacity(shape[0]);
            let values = output.iter().copied().collect::<Vec<_>>();
            for row in values.chunks_exact(dims) {
                let mut row = row.to_vec();
                if normalize_output {
                    normalize_vector(&mut row)?;
                }
                embeddings.push(Embedding::from_vec(info, row)?);
            }
            Ok(embeddings)
        }
        ndim => Err(Error::ort(format!(
            "model output must be 1D or 2D, got {ndim}D tensor"
        ))),
    }
}

pub(crate) fn single_embedding(
    embeddings: Vec<Embedding>,
    label: &str,
) -> Result<Embedding, Error> {
    match embeddings.len() {
        1 => Ok(embeddings.into_iter().next().expect("len checked")),
        count => Err(Error::ort(format!(
            "{label} expected one embedding, got {count}"
        ))),
    }
}

fn normalize_vector(values: &mut [f32]) -> Result<(), Error> {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm <= f32::MIN_POSITIVE {
        return Err(Error::ort("model returned a zero-norm embedding"));
    }
    for value in values {
        *value /= norm;
    }
    Ok(())
}

fn load_session(
    model_path: &Path,
    runtime: &RuntimeConfig,
) -> Result<SessionLoadSuccess, SessionLoadFailure> {
    let profiles = resolve_runtime_profile_plans(runtime)
        .unwrap_or_else(|_| runtime_profile_plans(runtime.device, runtime.provider_policy, runtime));
    if runtime.device == RuntimeDevice::Gpu && profiles.is_empty() {
        return Err(session_load_failure(
            Error::ort(
                "gpu execution is not configured for this platform in the current build; enable the matching provider features (`directml`, `coreml`, or `nvidia`) for this target",
            ),
            Vec::new(),
            Vec::new(),
            RuntimeIssueCode::ProviderNotCompiled,
            Vec::new(),
        ));
    }

    let mut aggregated_issues = Vec::new();
    let mut last_failure: Option<SessionLoadFailure> = None;
    let mut profile_errors = Vec::new();

    for profile in profiles {
        match load_session_for_profile(model_path, runtime, &profile) {
            Ok(mut success) => {
                if !aggregated_issues.is_empty() {
                    let mut issues = aggregated_issues;
                    issues.extend(success.issues);
                    success.issues = issues;
                }
                return Ok(success);
            }
            Err(failure) => {
                aggregated_issues.extend(failure.issues.clone());
                profile_errors.push(format!("{}: {}", profile.kind, failure.error));
                last_failure = Some(failure);
            }
        }
    }

    if let Some(mut failure) = last_failure {
        failure.issues = aggregated_issues;
        failure.error = Error::ort(format!(
            "failed to load ONNX model {} with all planned runtime profiles: {}",
            model_path.display(),
            profile_errors.join("; ")
        ));
        return Err(failure);
    }

    Err(session_load_failure(
        Error::ort(format!(
            "failed to load ONNX model {}: no runtime profile could be planned",
            model_path.display()
        )),
        Vec::new(),
        Vec::new(),
        RuntimeIssueCode::ProviderNotCompiled,
        Vec::new(),
    ))
}

fn load_session_for_profile(
    model_path: &Path,
    runtime: &RuntimeConfig,
    profile: &RuntimeProfilePlan,
) -> Result<SessionLoadSuccess, SessionLoadFailure> {
    let runtime_library_issues =
        prepare_runtime_libraries(&profile.library, &profile.providers)
            .map_err(|failure| SessionLoadFailure {
                error: failure.error,
                provider_attempts: Vec::new(),
                registered_providers: Vec::new(),
                issues: annotate_profile_issues(profile.kind, failure.issues),
            })?
            .issues;
    let mut builder = Session::builder()
        .map_err(Error::from_ort)
        .map_err(|error| {
            annotate_profile_failure(
                profile.kind,
                session_load_failure(
                    error,
                    Vec::new(),
                    Vec::new(),
                    RuntimeIssueCode::SessionBuildFailed,
                    runtime_library_issues.clone(),
                ),
            )
        })?;
    builder = builder
        .with_no_environment_execution_providers()
        .map_err(Error::from_ort)
        .map_err(|error| {
            annotate_profile_failure(
                profile.kind,
                session_load_failure(
                    error,
                    Vec::new(),
                    Vec::new(),
                    RuntimeIssueCode::SessionBuildFailed,
                    runtime_library_issues.clone(),
                ),
            )
        })?;
    let provider_state =
        configure_execution_providers(&mut builder, runtime, profile, runtime_library_issues)?;
    builder = builder
        .with_optimization_level(map_graph_optimization_level(
            runtime.graph_optimization_level,
        ))
        .map_err(Error::from_ort)
        .map_err(|error| {
            annotate_profile_failure(
                profile.kind,
                session_load_failure(
                    error,
                    provider_state.provider_attempts.clone(),
                    provider_state.registered_providers.clone(),
                    RuntimeIssueCode::SessionBuildFailed,
                    provider_state.issues.clone(),
                ),
            )
        })?;
    builder = builder
        .with_intra_threads(runtime.intra_threads)
        .map_err(Error::from_ort)
        .map_err(|error| {
            annotate_profile_failure(
                profile.kind,
                session_load_failure(
                    error,
                    provider_state.provider_attempts.clone(),
                    provider_state.registered_providers.clone(),
                    RuntimeIssueCode::SessionBuildFailed,
                    provider_state.issues.clone(),
                ),
            )
        })?;
    if let Some(inter_threads) = runtime.inter_threads {
        builder = builder
            .with_inter_threads(inter_threads)
            .map_err(Error::from_ort)
            .map_err(|error| {
                annotate_profile_failure(
                    profile.kind,
                    session_load_failure(
                        error,
                        provider_state.provider_attempts.clone(),
                        provider_state.registered_providers.clone(),
                        RuntimeIssueCode::SessionBuildFailed,
                        provider_state.issues.clone(),
                    ),
                )
            })?;
    }
    let session = builder.commit_from_file(model_path).map_err(|error| {
        annotate_profile_failure(
            profile.kind,
            session_load_failure(
                Error::ort(format!(
                    "failed to load ONNX model {}: {error}",
                    model_path.display()
                )),
                provider_state.provider_attempts.clone(),
                provider_state.registered_providers.clone(),
                RuntimeIssueCode::SessionBuildFailed,
                provider_state.issues.clone(),
            ),
        )
    })?;
    Ok(SessionLoadSuccess {
        session,
        selected_profile: profile.kind,
        provider_attempts: provider_state.provider_attempts,
        registered_providers: provider_state.registered_providers,
        issues: provider_state.issues,
        loaded_at_unix_ms: now_unix_ms(),
    })
}

fn annotate_profile_issues(
    profile: RuntimeProfileKind,
    issues: Vec<RuntimeIssue>,
) -> Vec<RuntimeIssue> {
    issues
        .into_iter()
        .map(|mut issue| {
            issue.message = format!("[{profile}] {}", issue.message);
            issue
        })
        .collect()
}

fn annotate_profile_failure(
    profile: RuntimeProfileKind,
    mut failure: SessionLoadFailure,
) -> SessionLoadFailure {
    failure.issues = annotate_profile_issues(profile, failure.issues);
    failure
}

#[derive(Clone, Debug)]
struct ProviderState {
    provider_attempts: Vec<ProviderAttempt>,
    registered_providers: Vec<ExecutionProviderKind>,
    issues: Vec<RuntimeIssue>,
}

fn configure_execution_providers(
    builder: &mut SessionBuilder,
    runtime: &RuntimeConfig,
    profile: &RuntimeProfilePlan,
    mut issues: Vec<RuntimeIssue>,
) -> Result<ProviderState, SessionLoadFailure> {
    if runtime.device == RuntimeDevice::Gpu && profile.providers.is_empty() {
        return Err(annotate_profile_failure(
            profile.kind,
            session_load_failure(
                Error::ort(
                    "gpu execution is not configured for this platform in the current build; enable the matching provider features (`directml`, `coreml`, or `nvidia`) for this target",
                ),
                Vec::new(),
                Vec::new(),
                RuntimeIssueCode::ProviderNotCompiled,
                issues,
            ),
        ));
    }

    let mut provider_attempts = Vec::with_capacity(profile.providers.len());
    let mut registered_providers = Vec::new();
    for provider in &profile.providers {
        match register_provider(builder, *provider) {
            Ok(()) => {
                provider_attempts.push(ProviderAttempt {
                    provider: *provider,
                    state: ProviderAttemptState::Registered,
                    detail: None,
                });
                registered_providers.push(*provider);
            }
            Err(error) => {
                provider_attempts.push(ProviderAttempt {
                    provider: *provider,
                    state: ProviderAttemptState::Failed,
                    detail: Some(error.to_string()),
                });
                issues.push(provider_registration_issue(
                    runtime,
                    profile.kind,
                    *provider,
                    error.to_string(),
                ));
            }
        }
    }

    if runtime.device == RuntimeDevice::Gpu
        && !registered_providers
            .iter()
            .any(|provider| provider.is_gpu())
    {
        return Err(annotate_profile_failure(
            profile.kind,
            session_load_failure(
                Error::ort(format!(
                    "gpu execution requested, but no GPU execution provider could be registered: {}",
                    format_provider_attempts(&provider_attempts)
                )),
                provider_attempts,
                registered_providers,
                RuntimeIssueCode::GpuExecutionUnavailable,
                issues,
            ),
        ));
    }

    if registered_providers.is_empty() {
        return Err(annotate_profile_failure(
            profile.kind,
            session_load_failure(
                Error::ort(format!(
                    "no execution provider could be registered: {}",
                    format_provider_attempts(&provider_attempts)
                )),
                provider_attempts,
                registered_providers,
                RuntimeIssueCode::ProviderRegistrationFailed,
                issues,
            ),
        ));
    }

    Ok(ProviderState {
        provider_attempts,
        registered_providers,
        issues,
    })
}

pub(crate) fn planned_runtime_profiles_snapshot(
    runtime: &RuntimeConfig,
) -> Vec<RuntimePlanProfileSnapshot> {
    resolve_runtime_profile_plans(runtime)
        .unwrap_or_else(|_| runtime_profile_plans(runtime.device, runtime.provider_policy, runtime))
        .into_iter()
        .map(|profile| RuntimePlanProfileSnapshot {
            kind: profile.kind,
            providers: profile.providers,
            library: RuntimeLibraryConfigSnapshot::from_library_config(&profile.library),
        })
        .collect()
}

fn planned_provider_kinds(runtime: &RuntimeConfig) -> Vec<ExecutionProviderKind> {
    resolve_runtime_profile_plans(runtime)
        .unwrap_or_else(|_| runtime_profile_plans(runtime.device, runtime.provider_policy, runtime))
        .into_iter()
        .flat_map(|profile| profile.providers)
        .collect()
}

fn resolve_runtime_profile_plans(runtime: &RuntimeConfig) -> Result<Vec<RuntimeProfilePlan>, Error> {
    if let Some(provider) = forced_execution_provider_from_env()? {
        let profile = provider_runtime_profile(provider);
        return Ok(vec![RuntimeProfilePlan {
            kind: profile,
            providers: vec![provider],
            library: runtime.resolved_library_for_profile(profile),
        }]);
    }
    Ok(runtime_profile_plans(
        runtime.device,
        runtime.provider_policy,
        runtime,
    ))
}

fn runtime_profile_plans(
    device: RuntimeDevice,
    policy: ProviderPolicy,
    runtime: &RuntimeConfig,
) -> Vec<RuntimeProfilePlan> {
    let mut profiles = Vec::new();
    match device {
        RuntimeDevice::Cpu => {
            profiles.push(runtime_profile_plan(
                RuntimeProfileKind::Cpu,
                vec![ExecutionProviderKind::Cpu],
                runtime,
            ));
        }
        RuntimeDevice::Auto => {
            append_platform_gpu_profiles(&mut profiles, policy, runtime);
            profiles.push(runtime_profile_plan(
                RuntimeProfileKind::Cpu,
                vec![ExecutionProviderKind::Cpu],
                runtime,
            ));
        }
        RuntimeDevice::Gpu => {
            append_platform_gpu_profiles(&mut profiles, policy, runtime);
        }
    }
    profiles
}

fn runtime_profile_plan(
    kind: RuntimeProfileKind,
    providers: Vec<ExecutionProviderKind>,
    runtime: &RuntimeConfig,
) -> RuntimeProfilePlan {
    RuntimeProfilePlan {
        kind,
        providers,
        library: runtime.resolved_library_for_profile(kind),
    }
}

fn append_platform_gpu_profiles(
    profiles: &mut Vec<RuntimeProfilePlan>,
    policy: ProviderPolicy,
    runtime: &RuntimeConfig,
) {
    match policy {
        ProviderPolicy::Auto | ProviderPolicy::Interactive => {
            push_interactive_gpu_profiles(profiles, runtime);
        }
        ProviderPolicy::Service => {
            push_service_gpu_profiles(profiles, runtime);
        }
    }
}

fn push_service_gpu_profiles(profiles: &mut Vec<RuntimeProfilePlan>, runtime: &RuntimeConfig) {
    #[cfg(all(
        feature = "nvidia",
        target_arch = "x86_64",
        any(target_os = "windows", target_os = "linux")
    ))]
    {
        profiles.push(runtime_profile_plan(
            RuntimeProfileKind::Nvidia,
            vec![ExecutionProviderKind::TensorRt, ExecutionProviderKind::Cuda],
            runtime,
        ));
    }

    #[cfg(all(feature = "directml", target_os = "windows"))]
    {
        profiles.push(runtime_profile_plan(
            RuntimeProfileKind::DirectMl,
            vec![ExecutionProviderKind::DirectMl],
            runtime,
        ));
    }

    #[cfg(all(feature = "coreml", target_vendor = "apple"))]
    {
        profiles.push(runtime_profile_plan(
            RuntimeProfileKind::CoreMl,
            vec![ExecutionProviderKind::CoreMl],
            runtime,
        ));
    }
}

fn push_interactive_gpu_profiles(
    profiles: &mut Vec<RuntimeProfilePlan>,
    runtime: &RuntimeConfig,
) {
    #[cfg(all(
        feature = "nvidia",
        target_arch = "x86_64",
        any(target_os = "windows", target_os = "linux")
    ))]
    {
        profiles.push(runtime_profile_plan(
            RuntimeProfileKind::Nvidia,
            vec![ExecutionProviderKind::Cuda],
            runtime,
        ));
    }

    #[cfg(all(feature = "directml", target_os = "windows"))]
    {
        profiles.push(runtime_profile_plan(
            RuntimeProfileKind::DirectMl,
            vec![ExecutionProviderKind::DirectMl],
            runtime,
        ));
    }

    #[cfg(all(feature = "coreml", target_vendor = "apple"))]
    {
        profiles.push(runtime_profile_plan(
            RuntimeProfileKind::CoreMl,
            vec![ExecutionProviderKind::CoreMl],
            runtime,
        ));
    }
}

fn provider_runtime_profile(provider: ExecutionProviderKind) -> RuntimeProfileKind {
    match provider {
        ExecutionProviderKind::TensorRt | ExecutionProviderKind::Cuda => RuntimeProfileKind::Nvidia,
        ExecutionProviderKind::DirectMl => RuntimeProfileKind::DirectMl,
        ExecutionProviderKind::CoreMl => RuntimeProfileKind::CoreMl,
        ExecutionProviderKind::Cpu => RuntimeProfileKind::Cpu,
    }
}

fn forced_execution_provider_from_env() -> Result<Option<ExecutionProviderKind>, Error> {
    let Some(value) = env::var_os(FORCE_PROVIDER_ENV) else {
        return Ok(None);
    };
    let value = value
        .into_string()
        .map_err(|_| Error::invalid_config(format!("{FORCE_PROVIDER_ENV} must be valid UTF-8")))?;
    let value = value.trim().to_ascii_lowercase();
    if value.is_empty() {
        return Ok(None);
    }

    let provider = match value.as_str() {
        "tensorrt" | "tensor_rt" => ExecutionProviderKind::TensorRt,
        "cuda" => ExecutionProviderKind::Cuda,
        "directml" | "direct_ml" | "dml" => ExecutionProviderKind::DirectMl,
        "coreml" | "core_ml" => ExecutionProviderKind::CoreMl,
        "cpu" => ExecutionProviderKind::Cpu,
        _ => {
            return Err(Error::invalid_config(format!(
                "unsupported {FORCE_PROVIDER_ENV}='{value}', expected one of: tensorrt, cuda, directml, coreml, cpu"
            )));
        }
    };

    if !provider_supported_on_current_build(provider) {
        return Err(Error::invalid_config(format!(
            "{FORCE_PROVIDER_ENV}='{value}' is not supported by the current build/platform"
        )));
    }

    Ok(Some(provider))
}

fn provider_supported_on_current_build(provider: ExecutionProviderKind) -> bool {
    match provider {
        ExecutionProviderKind::TensorRt | ExecutionProviderKind::Cuda => cfg!(all(
            feature = "nvidia",
            target_arch = "x86_64",
            any(target_os = "windows", target_os = "linux")
        )),
        ExecutionProviderKind::DirectMl => cfg!(all(feature = "directml", target_os = "windows")),
        ExecutionProviderKind::CoreMl => {
            cfg!(all(feature = "coreml", target_vendor = "apple"))
        }
        ExecutionProviderKind::Cpu => true,
    }
}

fn compiled_provider_kinds() -> Vec<ExecutionProviderKind> {
    let mut providers = Vec::new();
    for provider in [
        ExecutionProviderKind::TensorRt,
        ExecutionProviderKind::Cuda,
        ExecutionProviderKind::DirectMl,
        ExecutionProviderKind::CoreMl,
        ExecutionProviderKind::Cpu,
    ] {
        if provider_supported_on_current_build(provider) {
            providers.push(provider);
        }
    }
    providers
}

fn register_provider(
    builder: &mut SessionBuilder,
    provider: ExecutionProviderKind,
) -> Result<(), Error> {
    match provider {
        ExecutionProviderKind::TensorRt => ep::TensorRT::default()
            .register(builder)
            .map_err(|error| Error::from_ort(error)),
        ExecutionProviderKind::Cuda => ep::CUDA::default()
            .register(builder)
            .map_err(|error| Error::from_ort(error)),
        ExecutionProviderKind::DirectMl => ep::DirectML::default()
            .register(builder)
            .map_err(|error| Error::from_ort(error)),
        ExecutionProviderKind::CoreMl => ep::CoreML::default()
            .register(builder)
            .map_err(|error| Error::from_ort(error)),
        ExecutionProviderKind::Cpu => ep::CPU::default()
            .register(builder)
            .map_err(|error| Error::from_ort(error)),
    }
}

fn provider_registration_issue(
    runtime: &RuntimeConfig,
    profile: RuntimeProfileKind,
    provider: ExecutionProviderKind,
    message: String,
) -> RuntimeIssue {
    let library = runtime.resolved_library_for_profile(profile);
    let code = if !provider_supported_on_current_build(provider) {
        RuntimeIssueCode::ProviderNotCompiled
    } else if provider_library_expected_but_missing(&library, provider) {
        RuntimeIssueCode::ProviderLibraryMissing
    } else if looks_like_provider_unsupported_by_runtime_library(&message) {
        RuntimeIssueCode::ProviderUnsupportedByRuntimeLibrary
    } else if looks_like_provider_library_incompatible(&message) {
        RuntimeIssueCode::ProviderLibraryIncompatible
    } else if looks_like_missing_dependency(&message) {
        RuntimeIssueCode::DependencyLibraryMissing
    } else {
        RuntimeIssueCode::ProviderRegistrationFailed
    };

    RuntimeIssue {
        code,
        message: format!("failed to register {provider:?} in {profile} profile: {message}"),
        at_unix_ms: now_unix_ms(),
    }
}

fn provider_library_expected_but_missing(
    library: &RuntimeLibraryConfig,
    provider: ExecutionProviderKind,
) -> bool {
    let Some(provider_dir) = library.provider_dir.as_ref() else {
        return false;
    };
    let Some(library_name) = provider_library_name(provider) else {
        return false;
    };
    !provider_dir.join(library_name).is_file()
}

fn provider_library_name(provider: ExecutionProviderKind) -> Option<&'static str> {
    match provider {
        #[cfg(target_os = "windows")]
        ExecutionProviderKind::TensorRt => Some("onnxruntime_providers_tensorrt.dll"),
        #[cfg(not(target_os = "windows"))]
        ExecutionProviderKind::TensorRt => Some("libonnxruntime_providers_tensorrt.so"),
        #[cfg(target_os = "windows")]
        ExecutionProviderKind::Cuda => Some("onnxruntime_providers_cuda.dll"),
        #[cfg(not(target_os = "windows"))]
        ExecutionProviderKind::Cuda => Some("libonnxruntime_providers_cuda.so"),
        ExecutionProviderKind::DirectMl
        | ExecutionProviderKind::CoreMl
        | ExecutionProviderKind::Cpu => None,
    }
}

fn format_provider_attempts(provider_attempts: &[ProviderAttempt]) -> String {
    if provider_attempts.is_empty() {
        return String::from("no providers were planned");
    }
    provider_attempts
        .iter()
        .map(|attempt| match &attempt.detail {
            Some(detail) => format!("{:?}: {}", attempt.provider, detail),
            None => format!("{:?}: {:?}", attempt.provider, attempt.state),
        })
        .collect::<Vec<_>>()
        .join("; ")
}

fn session_load_failure(
    error: Error,
    provider_attempts: Vec<ProviderAttempt>,
    registered_providers: Vec<ExecutionProviderKind>,
    code: RuntimeIssueCode,
    mut issues: Vec<RuntimeIssue>,
) -> SessionLoadFailure {
    issues.push(RuntimeIssue {
        code,
        message: error.to_string(),
        at_unix_ms: now_unix_ms(),
    });
    SessionLoadFailure {
        error,
        provider_attempts,
        registered_providers,
        issues,
    }
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn map_graph_optimization_level(level: GraphOptimizationLevel) -> OrtGraphOptimizationLevel {
    match level {
        GraphOptimizationLevel::Disabled => OrtGraphOptimizationLevel::Disable,
        GraphOptimizationLevel::Basic => OrtGraphOptimizationLevel::Level1,
        GraphOptimizationLevel::Extended => OrtGraphOptimizationLevel::Level2,
        GraphOptimizationLevel::All => OrtGraphOptimizationLevel::All,
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::fs;
    use std::sync::Mutex;

    use tempfile::tempdir;

    use super::{
        ExecutionProviderKind, FORCE_PROVIDER_ENV, load_tokenizer, planned_provider_kinds,
    };
    use crate::config::{RuntimeConfig, RuntimeDevice};

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct ScopedEnvVar {
        key: &'static str,
        original: Option<OsString>,
    }

    impl ScopedEnvVar {
        fn set(key: &'static str, value: Option<&str>) -> Self {
            let original = std::env::var_os(key);
            match value {
                Some(value) => unsafe {
                    std::env::set_var(key, value);
                },
                None => unsafe {
                    std::env::remove_var(key);
                },
            }
            Self { key, original }
        }
    }

    impl Drop for ScopedEnvVar {
        fn drop(&mut self) {
            match &self.original {
                Some(value) => unsafe {
                    std::env::set_var(self.key, value);
                },
                None => unsafe {
                    std::env::remove_var(self.key);
                },
            }
        }
    }

    fn with_force_provider_env<T>(value: Option<&str>, f: impl FnOnce() -> T) -> T {
        let _lock = ENV_LOCK.lock().unwrap();
        let _env = ScopedEnvVar::set(FORCE_PROVIDER_ENV, value);
        f()
    }

    #[test]
    fn loads_wordpiece_vocab_txt() {
        let dir = tempdir().unwrap();
        let vocab_path = dir.path().join("vocab.txt");
        fs::write(&vocab_path, "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n你\n好\n").unwrap();

        let tokenizer = load_tokenizer(&vocab_path, 6, "[PAD]").unwrap();
        let encoding = tokenizer.encode("你好", true).unwrap();

        assert_eq!(encoding.get_ids(), &[2, 5, 6, 3, 0, 0]);
        assert_eq!(encoding.get_attention_mask(), &[1, 1, 1, 1, 0, 0]);
        assert_eq!(encoding.get_type_ids(), &[0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn cpu_mode_only_plans_cpu_provider() {
        with_force_provider_env(None, || {
            assert_eq!(
                planned_provider_kinds(
                    &RuntimeConfig::builder()
                        .device(RuntimeDevice::Cpu)
                        .build()
                        .unwrap()
                ),
                vec![ExecutionProviderKind::Cpu]
            );
        });
    }

    #[cfg(all(target_os = "windows", feature = "directml"))]
    #[test]
    fn force_provider_env_overrides_default_plan() {
        with_force_provider_env(Some("directml"), || {
            let planned = planned_provider_kinds(&RuntimeConfig::default());
            assert_eq!(planned, vec![ExecutionProviderKind::DirectMl]);
        });
    }

    #[cfg(all(target_os = "windows", feature = "directml"))]
    #[test]
    fn windows_auto_plan_keeps_directml_before_cpu() {
        with_force_provider_env(None, || {
            let planned = planned_provider_kinds(&RuntimeConfig::default());
            assert_eq!(planned.last(), Some(&ExecutionProviderKind::Cpu));
            assert!(planned.contains(&ExecutionProviderKind::DirectMl));
        });
    }

    #[cfg(all(target_os = "windows", feature = "nvidia", feature = "directml"))]
    #[test]
    fn windows_interactive_plan_prefers_cuda_and_directml_before_cpu() {
        with_force_provider_env(None, || {
            let planned = planned_provider_kinds(
                &RuntimeConfig::builder()
                    .provider_policy(crate::config::ProviderPolicy::Interactive)
                    .build()
                    .unwrap(),
            );

            assert_eq!(
                planned,
                vec![
                    ExecutionProviderKind::Cuda,
                    ExecutionProviderKind::DirectMl,
                    ExecutionProviderKind::Cpu,
                ]
            );
        });
    }

    #[cfg(all(target_os = "windows", feature = "nvidia", feature = "directml"))]
    #[test]
    fn windows_auto_plan_matches_interactive_profile_order() {
        with_force_provider_env(None, || {
            let planned = planned_provider_kinds(&RuntimeConfig::default());

            assert_eq!(
                planned,
                vec![
                    ExecutionProviderKind::Cuda,
                    ExecutionProviderKind::DirectMl,
                    ExecutionProviderKind::Cpu,
                ]
            );
        });
    }

    #[cfg(all(target_os = "windows", feature = "nvidia", feature = "directml"))]
    #[test]
    fn windows_service_plan_prefers_tensorrt_before_cuda_and_directml() {
        with_force_provider_env(None, || {
            let planned = planned_provider_kinds(
                &RuntimeConfig::builder()
                    .provider_policy(crate::config::ProviderPolicy::Service)
                    .build()
                    .unwrap(),
            );

            assert_eq!(
                planned,
                vec![
                    ExecutionProviderKind::TensorRt,
                    ExecutionProviderKind::Cuda,
                    ExecutionProviderKind::DirectMl,
                    ExecutionProviderKind::Cpu,
                ]
            );
        });
    }

    #[cfg(all(target_vendor = "apple", feature = "coreml"))]
    #[test]
    fn apple_auto_plan_keeps_coreml_before_cpu() {
        with_force_provider_env(None, || {
            let planned = planned_provider_kinds(&RuntimeConfig::default());
            assert_eq!(planned.last(), Some(&ExecutionProviderKind::Cpu));
            assert!(planned.contains(&ExecutionProviderKind::CoreMl));
        });
    }

    #[test]
    fn provider_registration_issue_marks_runtime_unsupported_errors() {
        let issue = super::provider_registration_issue(
            &RuntimeConfig::default(),
            crate::config::RuntimeProfileKind::Cpu,
            ExecutionProviderKind::Cpu,
            String::from("Specified provider is not supported."),
        );

        assert_eq!(
            issue.code,
            super::RuntimeIssueCode::ProviderUnsupportedByRuntimeLibrary
        );
    }

    #[test]
    fn provider_registration_issue_marks_incompatible_provider_libraries() {
        let issue = super::provider_registration_issue(
            &RuntimeConfig::default(),
            crate::config::RuntimeProfileKind::Cpu,
            ExecutionProviderKind::Cpu,
            String::from("Failed to find symbol CreateEpFactories in library, error code: 127"),
        );

        assert_eq!(
            issue.code,
            super::RuntimeIssueCode::ProviderLibraryIncompatible
        );
    }
}
