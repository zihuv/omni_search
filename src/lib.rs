mod backend;
mod bundle;
mod config;
mod embedding;
mod error;
mod manifest;
mod preprocess;
mod runtime;
mod score;

pub use crate::bundle::{ModelBundle, ModelInfo};
pub use crate::config::{
    GraphOptimizationLevel, ModelConfig, ModelFamily, ModelSource, OmniSearchConfig, RuntimeConfig,
    SessionPolicy,
};
pub use crate::embedding::Embedding;
pub use crate::error::Error;
pub use crate::runtime::{OmniSearch, RuntimeState};
pub use crate::score::{Scored, cosine_similarity, score_embeddings, top_k};
