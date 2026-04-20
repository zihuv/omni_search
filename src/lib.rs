mod backend;
mod bundle;
mod config;
mod embedding;
mod error;
mod manifest;
mod preprocess;
mod runtime;
mod runtime_env;
mod runtime_loader;
mod score;

pub use crate::bundle::{LocalModelDirProbe, ModelBundle, ModelInfo, probe_local_model_dir};
pub use crate::config::{
    GraphOptimizationLevel, ModelConfig, ModelFamily, ModelSource, ModelSourceKind,
    OmniSearchConfig, ProviderPolicy, RuntimeConfig, RuntimeConfigBuilder, RuntimeDevice,
    RuntimeLibraryConfig, RuntimeLibraryConfigBuilder, RuntimeLibraryConfigOverride,
    RuntimeLibraryConfigOverrideBuilder, RuntimeProfileKind, SessionPolicy,
};
pub use crate::embedding::Embedding;
pub use crate::error::Error;
pub use crate::runtime::{
    ExecutionProviderKind, OmniSearch, OmniSearchBuilder, ProviderAttempt, ProviderAttemptState,
    RuntimeConfigSnapshot, RuntimeIssue, RuntimeIssueCode, RuntimeLibraryConfigSnapshot,
    RuntimeMode, RuntimePlanProfileSnapshot, RuntimeSnapshot, RuntimeState, RuntimeSummary,
    SessionRuntimeSnapshot,
};
pub use crate::runtime_env::{
    default_intra_threads, env_bool, env_intra_threads, env_path, env_path_resolved,
    env_positive_usize, env_provider_policy, env_runtime_device, load_dotenv_from,
    logical_core_count, physical_core_count, runtime_config_from_env,
};
pub use crate::score::{Scored, cosine_similarity, score_embeddings, top_k};
