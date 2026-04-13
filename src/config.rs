use std::fmt;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFamily {
    FgClip,
    ChineseClip,
}

impl fmt::Display for ModelFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FgClip => f.write_str("fgclip"),
            Self::ChineseClip => f.write_str("chinese_clip"),
        }
    }
}

#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelSource {
    LocalBundleDir(PathBuf),
}

#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelConfig {
    pub family: ModelFamily,
    pub source: ModelSource,
}

impl ModelConfig {
    pub fn new(family: ModelFamily, source: ModelSource) -> Self {
        Self { family, source }
    }

    pub fn from_local_bundle(family: ModelFamily, path: impl Into<PathBuf>) -> Self {
        Self {
            family,
            source: ModelSource::LocalBundleDir(path.into()),
        }
    }
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionPolicy {
    SingleActive,
    KeepBothLoaded,
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphOptimizationLevel {
    Disabled,
    Basic,
    Extended,
    All,
}

#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeConfig {
    pub intra_threads: usize,
    pub inter_threads: Option<usize>,
    pub session_policy: SessionPolicy,
    pub graph_optimization_level: GraphOptimizationLevel,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            intra_threads: 4,
            inter_threads: None,
            session_policy: SessionPolicy::SingleActive,
            graph_optimization_level: GraphOptimizationLevel::All,
        }
    }
}

#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OmniSearchConfig {
    pub model: ModelConfig,
    pub runtime: RuntimeConfig,
}

impl OmniSearchConfig {
    pub fn new(model: ModelConfig, runtime: RuntimeConfig) -> Self {
        Self { model, runtime }
    }

    pub fn from_local_bundle(
        family: ModelFamily,
        path: impl Into<PathBuf>,
        runtime: RuntimeConfig,
    ) -> Self {
        Self {
            model: ModelConfig::from_local_bundle(family, path),
            runtime,
        }
    }
}
