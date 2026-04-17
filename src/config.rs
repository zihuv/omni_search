use std::fmt;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::Error;

#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFamily {
    FgClip,
    ChineseClip,
    OpenClip,
}

impl fmt::Display for ModelFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FgClip => f.write_str("fgclip"),
            Self::ChineseClip => f.write_str("chinese_clip"),
            Self::OpenClip => f.write_str("open_clip"),
        }
    }
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelSourceKind {
    LocalBundleDir,
}

impl fmt::Display for ModelSourceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LocalBundleDir => f.write_str("local_bundle_dir"),
        }
    }
}

#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelSource {
    LocalBundleDir(PathBuf),
}

impl ModelSource {
    pub fn kind(&self) -> ModelSourceKind {
        match self {
            Self::LocalBundleDir(_) => ModelSourceKind::LocalBundleDir,
        }
    }
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

#[derive(Clone, Debug)]
pub struct RuntimeConfigBuilder {
    config: RuntimeConfig,
}

impl Default for RuntimeConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: RuntimeConfig::default(),
        }
    }

    pub fn from_config(config: RuntimeConfig) -> Self {
        Self { config }
    }

    pub fn intra_threads(&mut self, val: usize) -> &mut Self {
        self.config.intra_threads = val;
        self
    }

    pub fn device(&mut self, val: RuntimeDevice) -> &mut Self {
        self.config.device = val;
        self
    }

    pub fn inter_threads(&mut self, val: usize) -> &mut Self {
        self.config.inter_threads = Some(val);
        self
    }

    pub fn clear_inter_threads(&mut self) -> &mut Self {
        self.config.inter_threads = None;
        self
    }

    pub fn fgclip_max_patches(&mut self, val: usize) -> &mut Self {
        self.config.fgclip_max_patches = Some(val);
        self
    }

    pub fn clear_fgclip_max_patches(&mut self) -> &mut Self {
        self.config.fgclip_max_patches = None;
        self
    }

    pub fn session_policy(&mut self, val: SessionPolicy) -> &mut Self {
        self.config.session_policy = val;
        self
    }

    pub fn graph_optimization_level(&mut self, val: GraphOptimizationLevel) -> &mut Self {
        self.config.graph_optimization_level = val;
        self
    }

    pub fn build(&mut self) -> Result<RuntimeConfig, Error> {
        self.config.validate()?;
        Ok(self.config.clone())
    }
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionPolicy {
    #[serde(alias = "SingleActive")]
    SingleActive,
    #[serde(alias = "KeepBothLoaded")]
    KeepBothLoaded,
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphOptimizationLevel {
    #[serde(alias = "Disabled")]
    Disabled,
    #[serde(alias = "Basic")]
    Basic,
    #[serde(alias = "Extended")]
    Extended,
    #[serde(alias = "All")]
    All,
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeDevice {
    #[serde(alias = "Auto")]
    Auto,
    #[serde(alias = "Cpu")]
    Cpu,
    #[serde(alias = "Gpu")]
    Gpu,
}

impl fmt::Display for RuntimeDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::Cpu => f.write_str("cpu"),
            Self::Gpu => f.write_str("gpu"),
        }
    }
}

#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeConfig {
    pub device: RuntimeDevice,
    pub intra_threads: usize,
    pub inter_threads: Option<usize>,
    pub fgclip_max_patches: Option<usize>,
    pub session_policy: SessionPolicy,
    pub graph_optimization_level: GraphOptimizationLevel,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            device: RuntimeDevice::Auto,
            intra_threads: std::thread::available_parallelism()
                .map(|parallelism| parallelism.get())
                .unwrap_or(4),
            inter_threads: None,
            fgclip_max_patches: None,
            session_policy: SessionPolicy::SingleActive,
            graph_optimization_level: GraphOptimizationLevel::All,
        }
    }
}

impl RuntimeConfig {
    pub fn builder() -> RuntimeConfigBuilder {
        RuntimeConfigBuilder::new()
    }

    pub fn validate(&self) -> Result<(), Error> {
        if self.intra_threads == 0 {
            return Err(Error::invalid_config(
                "runtime.intra_threads must be greater than 0",
            ));
        }
        if matches!(self.inter_threads, Some(0)) {
            return Err(Error::invalid_config(
                "runtime.inter_threads must be greater than 0 when set",
            ));
        }
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::{
        GraphOptimizationLevel, RuntimeConfig, RuntimeConfigBuilder, RuntimeDevice, SessionPolicy,
    };

    #[test]
    fn runtime_builder_uses_defaults_when_fields_are_not_overridden() {
        let expected = RuntimeConfig::default();
        let actual = RuntimeConfigBuilder::new().build().unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn runtime_builder_overrides_selected_fields_only() {
        let actual = RuntimeConfig::builder()
            .device(RuntimeDevice::Gpu)
            .intra_threads(2)
            .inter_threads(1)
            .fgclip_max_patches(256)
            .session_policy(SessionPolicy::KeepBothLoaded)
            .graph_optimization_level(GraphOptimizationLevel::Basic)
            .build()
            .unwrap();

        assert_eq!(actual.device, RuntimeDevice::Gpu);
        assert_eq!(actual.intra_threads, 2);
        assert_eq!(actual.inter_threads, Some(1));
        assert_eq!(actual.fgclip_max_patches, Some(256));
        assert_eq!(actual.session_policy, SessionPolicy::KeepBothLoaded);
        assert_eq!(
            actual.graph_optimization_level,
            GraphOptimizationLevel::Basic
        );
    }

    #[test]
    fn runtime_builder_can_clear_optional_overrides() {
        let actual = RuntimeConfig::builder()
            .inter_threads(2)
            .clear_inter_threads()
            .fgclip_max_patches(256)
            .clear_fgclip_max_patches()
            .build()
            .unwrap();

        assert_eq!(actual.inter_threads, None);
        assert_eq!(actual.fgclip_max_patches, None);
    }

    #[test]
    fn runtime_builder_rejects_invalid_values() {
        let error = RuntimeConfig::builder()
            .intra_threads(0)
            .build()
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("runtime.intra_threads must be greater than 0")
        );
    }

    #[test]
    fn session_policy_deserializes_snake_case_and_legacy_pascal_case() {
        let snake_case: SessionPolicy = serde_json::from_str(r#""keep_both_loaded""#).unwrap();
        let legacy_pascal_case: SessionPolicy =
            serde_json::from_str(r#""KeepBothLoaded""#).unwrap();

        assert_eq!(snake_case, SessionPolicy::KeepBothLoaded);
        assert_eq!(legacy_pascal_case, SessionPolicy::KeepBothLoaded);
    }

    #[test]
    fn graph_optimization_level_deserializes_snake_case_and_legacy_pascal_case() {
        let snake_case: GraphOptimizationLevel = serde_json::from_str(r#""basic""#).unwrap();
        let legacy_pascal_case: GraphOptimizationLevel =
            serde_json::from_str(r#""Basic""#).unwrap();

        assert_eq!(snake_case, GraphOptimizationLevel::Basic);
        assert_eq!(legacy_pascal_case, GraphOptimizationLevel::Basic);
    }

    #[test]
    fn runtime_device_deserializes_snake_case_and_legacy_pascal_case() {
        let snake_case: RuntimeDevice = serde_json::from_str(r#""gpu""#).unwrap();
        let legacy_pascal_case: RuntimeDevice = serde_json::from_str(r#""Gpu""#).unwrap();

        assert_eq!(snake_case, RuntimeDevice::Gpu);
        assert_eq!(legacy_pascal_case, RuntimeDevice::Gpu);
    }
}
