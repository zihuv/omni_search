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
};
use crate::embedding::Embedding;
use crate::error::Error;
use crate::runtime::{
    ExecutionProviderKind, ProviderAttempt, ProviderAttemptState, RuntimeIssue, RuntimeIssueCode,
    RuntimeSnapshot, RuntimeState, SessionRuntimeSnapshot, build_runtime_snapshot,
};

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
    last_used_at: Option<Instant>,
    last_used_at_unix_ms: Option<u64>,
    last_loaded_at_unix_ms: Option<u64>,
    provider_attempts: Vec<ProviderAttempt>,
    registered_providers: Vec<ExecutionProviderKind>,
    last_error: Option<RuntimeIssue>,
}

struct SessionLoadSuccess {
    session: Session,
    provider_attempts: Vec<ProviderAttempt>,
    registered_providers: Vec<ExecutionProviderKind>,
    loaded_at_unix_ms: u64,
}

struct SessionLoadFailure {
    error: Error,
    provider_attempts: Vec<ProviderAttempt>,
    registered_providers: Vec<ExecutionProviderKind>,
    issue: RuntimeIssue,
}

impl LazySession {
    pub(crate) fn new(model_path: PathBuf, runtime: RuntimeConfig) -> Self {
        Self {
            model_path,
            runtime,
            state: Mutex::new(SessionState {
                session: None,
                last_used_at: None,
                last_used_at_unix_ms: None,
                last_loaded_at_unix_ms: None,
                provider_attempts: Vec::new(),
                registered_providers: Vec::new(),
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
                state.last_error = Some(RuntimeIssue {
                    code: RuntimeIssueCode::InferenceFailed,
                    message: error.to_string(),
                    at_unix_ms: now_unix_ms(),
                });
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
        let planned_providers = planned_provider_kinds(&self.runtime);
        let loaded = state.session.is_some();
        let effective_provider = if loaded {
            state.registered_providers.first().copied()
        } else {
            None
        };
        let mut snapshot = SessionRuntimeSnapshot {
            loaded,
            planned_providers: planned_providers.clone(),
            provider_attempts: state.provider_attempts.clone(),
            registered_providers: state.registered_providers.clone(),
            effective_provider,
            mode: crate::runtime::RuntimeMode::Unknown,
            fallback_to_cpu: loaded
                && self.runtime.device != RuntimeDevice::Cpu
                && planned_providers.iter().any(|provider| provider.is_gpu())
                && effective_provider == Some(ExecutionProviderKind::Cpu),
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
                state.provider_attempts = loaded.provider_attempts;
                state.registered_providers = loaded.registered_providers;
                state.last_loaded_at_unix_ms = Some(loaded.loaded_at_unix_ms);
                state.last_error = None;
                Ok(())
            }
            Err(failure) => {
                state.session = None;
                state.provider_attempts = failure.provider_attempts;
                state.registered_providers = failure.registered_providers;
                state.last_error = Some(failure.issue);
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
    let mut builder = Session::builder()
        .map_err(Error::from_ort)
        .map_err(|error| {
            session_load_failure(
                error,
                Vec::new(),
                Vec::new(),
                RuntimeIssueCode::SessionBuildFailed,
            )
        })?;
    builder = builder
        .with_no_environment_execution_providers()
        .map_err(Error::from_ort)
        .map_err(|error| {
            session_load_failure(
                error,
                Vec::new(),
                Vec::new(),
                RuntimeIssueCode::SessionBuildFailed,
            )
        })?;
    let provider_state = configure_execution_providers(&mut builder, runtime)?;
    builder = builder
        .with_optimization_level(map_graph_optimization_level(
            runtime.graph_optimization_level,
        ))
        .map_err(Error::from_ort)
        .map_err(|error| {
            session_load_failure(
                error,
                provider_state.provider_attempts.clone(),
                provider_state.registered_providers.clone(),
                RuntimeIssueCode::SessionBuildFailed,
            )
        })?;
    builder = builder
        .with_intra_threads(runtime.intra_threads)
        .map_err(Error::from_ort)
        .map_err(|error| {
            session_load_failure(
                error,
                provider_state.provider_attempts.clone(),
                provider_state.registered_providers.clone(),
                RuntimeIssueCode::SessionBuildFailed,
            )
        })?;
    if let Some(inter_threads) = runtime.inter_threads {
        builder = builder
            .with_inter_threads(inter_threads)
            .map_err(Error::from_ort)
            .map_err(|error| {
                session_load_failure(
                    error,
                    provider_state.provider_attempts.clone(),
                    provider_state.registered_providers.clone(),
                    RuntimeIssueCode::SessionBuildFailed,
                )
            })?;
    }
    let session = builder.commit_from_file(model_path).map_err(|error| {
        session_load_failure(
            Error::ort(format!(
                "failed to load ONNX model {}: {error}",
                model_path.display()
            )),
            provider_state.provider_attempts.clone(),
            provider_state.registered_providers.clone(),
            RuntimeIssueCode::SessionBuildFailed,
        )
    })?;
    Ok(SessionLoadSuccess {
        session,
        provider_attempts: provider_state.provider_attempts,
        registered_providers: provider_state.registered_providers,
        loaded_at_unix_ms: now_unix_ms(),
    })
}

#[derive(Clone, Debug)]
struct ProviderState {
    provider_attempts: Vec<ProviderAttempt>,
    registered_providers: Vec<ExecutionProviderKind>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ProviderSpec {
    kind: ExecutionProviderKind,
}

fn configure_execution_providers(
    builder: &mut SessionBuilder,
    runtime: &RuntimeConfig,
) -> Result<ProviderState, SessionLoadFailure> {
    let provider_specs = resolve_execution_provider_specs(runtime).map_err(|error| {
        session_load_failure(
            error,
            Vec::new(),
            Vec::new(),
            RuntimeIssueCode::ProviderRegistrationFailed,
        )
    })?;
    if runtime.device == RuntimeDevice::Gpu && provider_specs.is_empty() {
        return Err(session_load_failure(
            Error::ort(
                "gpu execution is not configured for this platform in the current build; enable the `nvidia` feature for TensorRT/CUDA on supported Windows/Linux x64 targets, use DirectML on Windows, or CoreML on Apple platforms",
            ),
            Vec::new(),
            Vec::new(),
            RuntimeIssueCode::GpuExecutionUnavailable,
        ));
    }

    let mut provider_attempts = Vec::with_capacity(provider_specs.len());
    let mut registered_providers = Vec::new();
    for provider in provider_specs {
        match register_provider(builder, provider.kind) {
            Ok(()) => {
                provider_attempts.push(ProviderAttempt {
                    provider: provider.kind,
                    state: ProviderAttemptState::Registered,
                    detail: None,
                });
                registered_providers.push(provider.kind);
            }
            Err(error) => {
                provider_attempts.push(ProviderAttempt {
                    provider: provider.kind,
                    state: ProviderAttemptState::Failed,
                    detail: Some(error.to_string()),
                });
            }
        }
    }

    if runtime.device == RuntimeDevice::Gpu
        && !registered_providers
            .iter()
            .any(|provider| provider.is_gpu())
    {
        return Err(session_load_failure(
            Error::ort(format!(
                "gpu execution requested, but no GPU execution provider could be registered: {}",
                format_provider_attempts(&provider_attempts)
            )),
            provider_attempts,
            registered_providers,
            RuntimeIssueCode::GpuExecutionUnavailable,
        ));
    }

    if registered_providers.is_empty() {
        return Err(session_load_failure(
            Error::ort(format!(
                "no execution provider could be registered: {}",
                format_provider_attempts(&provider_attempts)
            )),
            provider_attempts,
            registered_providers,
            RuntimeIssueCode::ProviderRegistrationFailed,
        ));
    }

    Ok(ProviderState {
        provider_attempts,
        registered_providers,
    })
}

fn execution_provider_specs(device: RuntimeDevice, policy: ProviderPolicy) -> Vec<ProviderSpec> {
    let mut providers = Vec::new();
    match device {
        RuntimeDevice::Cpu => {
            providers.push(ProviderSpec {
                kind: ExecutionProviderKind::Cpu,
            });
        }
        RuntimeDevice::Auto => {
            append_platform_gpu_providers(&mut providers, policy);
            providers.push(ProviderSpec {
                kind: ExecutionProviderKind::Cpu,
            });
        }
        RuntimeDevice::Gpu => {
            append_platform_gpu_providers(&mut providers, policy);
        }
    }
    providers
}

fn planned_provider_kinds(runtime: &RuntimeConfig) -> Vec<ExecutionProviderKind> {
    resolve_execution_provider_specs(runtime)
        .unwrap_or_else(|_| execution_provider_specs(runtime.device, runtime.provider_policy))
        .into_iter()
        .map(|provider| provider.kind)
        .collect()
}

fn resolve_execution_provider_specs(runtime: &RuntimeConfig) -> Result<Vec<ProviderSpec>, Error> {
    if let Some(provider) = forced_execution_provider_from_env()? {
        return Ok(vec![ProviderSpec { kind: provider }]);
    }
    Ok(execution_provider_specs(
        runtime.device,
        runtime.provider_policy,
    ))
}

fn append_platform_gpu_providers(providers: &mut Vec<ProviderSpec>, policy: ProviderPolicy) {
    for kind in platform_gpu_provider_kinds(policy) {
        providers.push(ProviderSpec { kind });
    }
}

fn platform_gpu_provider_kinds(policy: ProviderPolicy) -> Vec<ExecutionProviderKind> {
    let mut providers = Vec::new();

    match policy {
        ProviderPolicy::Auto | ProviderPolicy::Service => {
            push_service_gpu_providers(&mut providers);
        }
        ProviderPolicy::Interactive => {
            push_interactive_gpu_providers(&mut providers);
        }
    }

    providers
}

fn push_service_gpu_providers(providers: &mut Vec<ExecutionProviderKind>) {
    #[cfg(all(
        feature = "nvidia",
        target_arch = "x86_64",
        any(target_os = "windows", target_os = "linux")
    ))]
    {
        providers.push(ExecutionProviderKind::TensorRt);
        providers.push(ExecutionProviderKind::Cuda);
    }

    #[cfg(target_os = "windows")]
    {
        providers.push(ExecutionProviderKind::DirectMl);
    }

    #[cfg(target_vendor = "apple")]
    {
        providers.push(ExecutionProviderKind::CoreMl);
    }
}

fn push_interactive_gpu_providers(providers: &mut Vec<ExecutionProviderKind>) {
    #[cfg(all(
        feature = "nvidia",
        target_arch = "x86_64",
        any(target_os = "windows", target_os = "linux")
    ))]
    {
        providers.push(ExecutionProviderKind::Cuda);
    }

    #[cfg(target_os = "windows")]
    {
        providers.push(ExecutionProviderKind::DirectMl);
    }

    #[cfg(target_vendor = "apple")]
    {
        providers.push(ExecutionProviderKind::CoreMl);
    }

    #[cfg(all(
        feature = "nvidia",
        target_arch = "x86_64",
        any(target_os = "windows", target_os = "linux")
    ))]
    {
        providers.push(ExecutionProviderKind::TensorRt);
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
        ExecutionProviderKind::DirectMl => cfg!(target_os = "windows"),
        ExecutionProviderKind::CoreMl => cfg!(target_vendor = "apple"),
        ExecutionProviderKind::Cpu => true,
    }
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
) -> SessionLoadFailure {
    SessionLoadFailure {
        issue: RuntimeIssue {
            code,
            message: error.to_string(),
            at_unix_ms: now_unix_ms(),
        },
        error,
        provider_attempts,
        registered_providers,
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

    #[cfg(target_os = "windows")]
    #[test]
    fn force_provider_env_overrides_default_plan() {
        with_force_provider_env(Some("directml"), || {
            let planned = planned_provider_kinds(&RuntimeConfig::default());
            assert_eq!(planned, vec![ExecutionProviderKind::DirectMl]);
        });
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn windows_auto_plan_keeps_directml_before_cpu() {
        with_force_provider_env(None, || {
            let planned = planned_provider_kinds(&RuntimeConfig::default());
            assert_eq!(planned.last(), Some(&ExecutionProviderKind::Cpu));
            assert!(planned.contains(&ExecutionProviderKind::DirectMl));
        });
    }

    #[cfg(all(target_os = "windows", feature = "nvidia"))]
    #[test]
    fn windows_interactive_plan_prefers_cuda_and_directml_before_tensorrt() {
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
                    ExecutionProviderKind::TensorRt,
                    ExecutionProviderKind::Cpu,
                ]
            );
        });
    }

    #[cfg(all(target_os = "windows", feature = "nvidia"))]
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

    #[cfg(target_vendor = "apple")]
    #[test]
    fn apple_auto_plan_keeps_coreml_before_cpu() {
        with_force_provider_env(None, || {
            let planned = planned_provider_kinds(&RuntimeConfig::default());
            assert_eq!(planned.last(), Some(&ExecutionProviderKind::Cpu));
            assert!(planned.contains(&ExecutionProviderKind::CoreMl));
        });
    }
}
