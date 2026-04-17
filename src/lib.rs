mod backend;
mod bundle;
mod config;
mod embedding;
mod error;
mod manifest;
mod preprocess;
mod runtime;
mod runtime_env;
mod score;

pub use crate::bundle::{LocalModelDirProbe, ModelBundle, ModelInfo, probe_local_model_dir};
pub use crate::config::{
    GraphOptimizationLevel, ModelConfig, ModelFamily, ModelSource, ModelSourceKind,
    OmniSearchConfig, RuntimeConfig, RuntimeConfigBuilder, RuntimeDevice, SessionPolicy,
};
pub use crate::embedding::Embedding;
pub use crate::error::Error;
pub use crate::runtime::{OmniSearch, OmniSearchBuilder, RuntimeState};
pub use crate::runtime_env::{
    default_intra_threads, env_intra_threads, env_path, env_path_resolved, env_positive_usize,
    env_runtime_device, load_dotenv_from, logical_core_count, physical_core_count,
    runtime_config_from_env,
};
pub use crate::score::{Scored, cosine_similarity, score_embeddings, top_k};
