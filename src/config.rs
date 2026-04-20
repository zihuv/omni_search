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
pub struct RuntimeLibraryConfigBuilder {
    config: RuntimeLibraryConfig,
}

impl Default for RuntimeLibraryConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeLibraryConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: RuntimeLibraryConfig::default(),
        }
    }

    pub fn from_config(config: RuntimeLibraryConfig) -> Self {
        Self { config }
    }

    pub fn ort_dylib_path(&mut self, val: impl Into<PathBuf>) -> &mut Self {
        self.config.ort_dylib_path = Some(val.into());
        self
    }

    pub fn clear_ort_dylib_path(&mut self) -> &mut Self {
        self.config.ort_dylib_path = None;
        self
    }

    pub fn provider_dir(&mut self, val: impl Into<PathBuf>) -> &mut Self {
        self.config.provider_dir = Some(val.into());
        self
    }

    pub fn clear_provider_dir(&mut self) -> &mut Self {
        self.config.provider_dir = None;
        self
    }

    pub fn cuda_bin_dir(&mut self, val: impl Into<PathBuf>) -> &mut Self {
        self.config.cuda_bin_dir = Some(val.into());
        self
    }

    pub fn clear_cuda_bin_dir(&mut self) -> &mut Self {
        self.config.cuda_bin_dir = None;
        self
    }

    pub fn cudnn_bin_dir(&mut self, val: impl Into<PathBuf>) -> &mut Self {
        self.config.cudnn_bin_dir = Some(val.into());
        self
    }

    pub fn clear_cudnn_bin_dir(&mut self) -> &mut Self {
        self.config.cudnn_bin_dir = None;
        self
    }

    pub fn tensorrt_lib_dir(&mut self, val: impl Into<PathBuf>) -> &mut Self {
        self.config.tensorrt_lib_dir = Some(val.into());
        self
    }

    pub fn clear_tensorrt_lib_dir(&mut self) -> &mut Self {
        self.config.tensorrt_lib_dir = None;
        self
    }

    pub fn preload(&mut self, val: bool) -> &mut Self {
        self.config.preload = val;
        self
    }

    pub fn build(&mut self) -> Result<RuntimeLibraryConfig, Error> {
        self.config.validate()?;
        Ok(self.config.clone())
    }
}

#[derive(Clone, Debug)]
pub struct RuntimeLibraryConfigOverrideBuilder {
    config: RuntimeLibraryConfigOverride,
}

impl Default for RuntimeLibraryConfigOverrideBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeLibraryConfigOverrideBuilder {
    pub fn new() -> Self {
        Self {
            config: RuntimeLibraryConfigOverride::default(),
        }
    }

    pub fn from_config(config: RuntimeLibraryConfigOverride) -> Self {
        Self { config }
    }

    pub fn ort_dylib_path(&mut self, val: impl Into<PathBuf>) -> &mut Self {
        self.config.ort_dylib_path = Some(val.into());
        self
    }

    pub fn clear_ort_dylib_path(&mut self) -> &mut Self {
        self.config.ort_dylib_path = None;
        self
    }

    pub fn provider_dir(&mut self, val: impl Into<PathBuf>) -> &mut Self {
        self.config.provider_dir = Some(val.into());
        self
    }

    pub fn clear_provider_dir(&mut self) -> &mut Self {
        self.config.provider_dir = None;
        self
    }

    pub fn cuda_bin_dir(&mut self, val: impl Into<PathBuf>) -> &mut Self {
        self.config.cuda_bin_dir = Some(val.into());
        self
    }

    pub fn clear_cuda_bin_dir(&mut self) -> &mut Self {
        self.config.cuda_bin_dir = None;
        self
    }

    pub fn cudnn_bin_dir(&mut self, val: impl Into<PathBuf>) -> &mut Self {
        self.config.cudnn_bin_dir = Some(val.into());
        self
    }

    pub fn clear_cudnn_bin_dir(&mut self) -> &mut Self {
        self.config.cudnn_bin_dir = None;
        self
    }

    pub fn tensorrt_lib_dir(&mut self, val: impl Into<PathBuf>) -> &mut Self {
        self.config.tensorrt_lib_dir = Some(val.into());
        self
    }

    pub fn clear_tensorrt_lib_dir(&mut self) -> &mut Self {
        self.config.tensorrt_lib_dir = None;
        self
    }

    pub fn preload(&mut self, val: bool) -> &mut Self {
        self.config.preload = Some(val);
        self
    }

    pub fn clear_preload(&mut self) -> &mut Self {
        self.config.preload = None;
        self
    }

    pub fn build(&mut self) -> Result<RuntimeLibraryConfigOverride, Error> {
        self.config
            .validate("runtime.library_override")
            .map(|()| self.config.clone())
    }
}

#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeLibraryConfig {
    pub ort_dylib_path: Option<PathBuf>,
    pub provider_dir: Option<PathBuf>,
    pub cuda_bin_dir: Option<PathBuf>,
    pub cudnn_bin_dir: Option<PathBuf>,
    pub tensorrt_lib_dir: Option<PathBuf>,
    pub preload: bool,
}

impl Default for RuntimeLibraryConfig {
    fn default() -> Self {
        Self {
            ort_dylib_path: None,
            provider_dir: None,
            cuda_bin_dir: None,
            cudnn_bin_dir: None,
            tensorrt_lib_dir: None,
            preload: true,
        }
    }
}

impl RuntimeLibraryConfig {
    pub fn builder() -> RuntimeLibraryConfigBuilder {
        RuntimeLibraryConfigBuilder::new()
    }

    pub fn validate(&self) -> Result<(), Error> {
        validate_optional_path("runtime.library.ort_dylib_path", &self.ort_dylib_path)?;
        validate_optional_path("runtime.library.provider_dir", &self.provider_dir)?;
        validate_optional_path("runtime.library.cuda_bin_dir", &self.cuda_bin_dir)?;
        validate_optional_path("runtime.library.cudnn_bin_dir", &self.cudnn_bin_dir)?;
        validate_optional_path("runtime.library.tensorrt_lib_dir", &self.tensorrt_lib_dir)?;

        if self.ort_dylib_path.is_some() && !cfg!(feature = "runtime-dynamic") {
            return Err(Error::invalid_config(
                "runtime.library.ort_dylib_path requires the `runtime-dynamic` crate feature",
            ));
        }

        Ok(())
    }
}

#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RuntimeLibraryConfigOverride {
    pub ort_dylib_path: Option<PathBuf>,
    pub provider_dir: Option<PathBuf>,
    pub cuda_bin_dir: Option<PathBuf>,
    pub cudnn_bin_dir: Option<PathBuf>,
    pub tensorrt_lib_dir: Option<PathBuf>,
    pub preload: Option<bool>,
}

impl RuntimeLibraryConfigOverride {
    pub fn builder() -> RuntimeLibraryConfigOverrideBuilder {
        RuntimeLibraryConfigOverrideBuilder::new()
    }

    pub fn resolve(&self, base: &RuntimeLibraryConfig) -> RuntimeLibraryConfig {
        RuntimeLibraryConfig {
            ort_dylib_path: self
                .ort_dylib_path
                .clone()
                .or_else(|| base.ort_dylib_path.clone()),
            provider_dir: self
                .provider_dir
                .clone()
                .or_else(|| base.provider_dir.clone()),
            cuda_bin_dir: self
                .cuda_bin_dir
                .clone()
                .or_else(|| base.cuda_bin_dir.clone()),
            cudnn_bin_dir: self
                .cudnn_bin_dir
                .clone()
                .or_else(|| base.cudnn_bin_dir.clone()),
            tensorrt_lib_dir: self
                .tensorrt_lib_dir
                .clone()
                .or_else(|| base.tensorrt_lib_dir.clone()),
            preload: self.preload.unwrap_or(base.preload),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.ort_dylib_path.is_none()
            && self.provider_dir.is_none()
            && self.cuda_bin_dir.is_none()
            && self.cudnn_bin_dir.is_none()
            && self.tensorrt_lib_dir.is_none()
            && self.preload.is_none()
    }

    pub fn validate(&self, prefix: &str) -> Result<(), Error> {
        validate_optional_path(&format!("{prefix}.ort_dylib_path"), &self.ort_dylib_path)?;
        validate_optional_path(&format!("{prefix}.provider_dir"), &self.provider_dir)?;
        validate_optional_path(&format!("{prefix}.cuda_bin_dir"), &self.cuda_bin_dir)?;
        validate_optional_path(&format!("{prefix}.cudnn_bin_dir"), &self.cudnn_bin_dir)?;
        validate_optional_path(
            &format!("{prefix}.tensorrt_lib_dir"),
            &self.tensorrt_lib_dir,
        )?;

        if self.ort_dylib_path.is_some() && !cfg!(feature = "runtime-dynamic") {
            return Err(Error::invalid_config(format!(
                "{prefix}.ort_dylib_path requires the `runtime-dynamic` crate feature"
            )));
        }

        Ok(())
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

    pub fn runtime_library_config(&mut self, val: RuntimeLibraryConfig) -> &mut Self {
        self.config.library = val;
        self
    }

    pub fn nvidia_runtime_library_config(
        &mut self,
        val: RuntimeLibraryConfigOverride,
    ) -> &mut Self {
        self.config.nvidia_library = val;
        self
    }

    pub fn clear_nvidia_runtime_library_config(&mut self) -> &mut Self {
        self.config.nvidia_library = RuntimeLibraryConfigOverride::default();
        self
    }

    pub fn directml_runtime_library_config(
        &mut self,
        val: RuntimeLibraryConfigOverride,
    ) -> &mut Self {
        self.config.directml_library = val;
        self
    }

    pub fn clear_directml_runtime_library_config(&mut self) -> &mut Self {
        self.config.directml_library = RuntimeLibraryConfigOverride::default();
        self
    }

    pub fn coreml_runtime_library_config(
        &mut self,
        val: RuntimeLibraryConfigOverride,
    ) -> &mut Self {
        self.config.coreml_library = val;
        self
    }

    pub fn clear_coreml_runtime_library_config(&mut self) -> &mut Self {
        self.config.coreml_library = RuntimeLibraryConfigOverride::default();
        self
    }

    pub fn ort_dylib_path(&mut self, val: impl Into<PathBuf>) -> &mut Self {
        self.config.library.ort_dylib_path = Some(val.into());
        self
    }

    pub fn clear_ort_dylib_path(&mut self) -> &mut Self {
        self.config.library.ort_dylib_path = None;
        self
    }

    pub fn provider_dir(&mut self, val: impl Into<PathBuf>) -> &mut Self {
        self.config.library.provider_dir = Some(val.into());
        self
    }

    pub fn clear_provider_dir(&mut self) -> &mut Self {
        self.config.library.provider_dir = None;
        self
    }

    pub fn cuda_bin_dir(&mut self, val: impl Into<PathBuf>) -> &mut Self {
        self.config.library.cuda_bin_dir = Some(val.into());
        self
    }

    pub fn clear_cuda_bin_dir(&mut self) -> &mut Self {
        self.config.library.cuda_bin_dir = None;
        self
    }

    pub fn cudnn_bin_dir(&mut self, val: impl Into<PathBuf>) -> &mut Self {
        self.config.library.cudnn_bin_dir = Some(val.into());
        self
    }

    pub fn clear_cudnn_bin_dir(&mut self) -> &mut Self {
        self.config.library.cudnn_bin_dir = None;
        self
    }

    pub fn tensorrt_lib_dir(&mut self, val: impl Into<PathBuf>) -> &mut Self {
        self.config.library.tensorrt_lib_dir = Some(val.into());
        self
    }

    pub fn clear_tensorrt_lib_dir(&mut self) -> &mut Self {
        self.config.library.tensorrt_lib_dir = None;
        self
    }

    pub fn preload_runtime_libraries(&mut self, val: bool) -> &mut Self {
        self.config.library.preload = val;
        self
    }

    pub fn intra_threads(&mut self, val: usize) -> &mut Self {
        self.config.intra_threads = val;
        self
    }

    pub fn device(&mut self, val: RuntimeDevice) -> &mut Self {
        self.config.device = val;
        self
    }

    pub fn provider_policy(&mut self, val: ProviderPolicy) -> &mut Self {
        self.config.provider_policy = val;
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeProfileKind {
    #[serde(alias = "Nvidia")]
    Nvidia,
    #[serde(alias = "DirectMl")]
    DirectMl,
    #[serde(alias = "CoreMl")]
    CoreMl,
    #[serde(alias = "Cpu")]
    Cpu,
}

impl fmt::Display for RuntimeProfileKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Nvidia => f.write_str("nvidia"),
            Self::DirectMl => f.write_str("directml"),
            Self::CoreMl => f.write_str("coreml"),
            Self::Cpu => f.write_str("cpu"),
        }
    }
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderPolicy {
    #[serde(alias = "Auto")]
    Auto,
    #[serde(alias = "Interactive")]
    Interactive,
    #[serde(alias = "Service")]
    Service,
}

impl fmt::Display for ProviderPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::Interactive => f.write_str("interactive"),
            Self::Service => f.write_str("service"),
        }
    }
}

#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeConfig {
    pub device: RuntimeDevice,
    pub provider_policy: ProviderPolicy,
    pub library: RuntimeLibraryConfig,
    pub nvidia_library: RuntimeLibraryConfigOverride,
    pub directml_library: RuntimeLibraryConfigOverride,
    pub coreml_library: RuntimeLibraryConfigOverride,
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
            provider_policy: ProviderPolicy::Auto,
            library: RuntimeLibraryConfig::default(),
            nvidia_library: RuntimeLibraryConfigOverride::default(),
            directml_library: RuntimeLibraryConfigOverride::default(),
            coreml_library: RuntimeLibraryConfigOverride::default(),
            intra_threads: crate::runtime_env::default_intra_threads(),
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
        self.library.validate()?;
        self.nvidia_library.validate("runtime.nvidia_library")?;
        self.directml_library
            .validate("runtime.directml_library")?;
        self.coreml_library.validate("runtime.coreml_library")?;
        Ok(())
    }

    pub fn resolved_library_for_profile(&self, profile: RuntimeProfileKind) -> RuntimeLibraryConfig {
        match profile {
            RuntimeProfileKind::Nvidia => self.nvidia_library.resolve(&self.library),
            RuntimeProfileKind::DirectMl => self.directml_library.resolve(&self.library),
            RuntimeProfileKind::CoreMl => self.coreml_library.resolve(&self.library),
            RuntimeProfileKind::Cpu => self.library.clone(),
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

fn validate_optional_path(name: &str, value: &Option<PathBuf>) -> Result<(), Error> {
    if value
        .as_ref()
        .is_some_and(|path| path.as_os_str().is_empty())
    {
        return Err(Error::invalid_config(format!(
            "{name} must not be an empty path"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        GraphOptimizationLevel, ProviderPolicy, RuntimeConfig, RuntimeConfigBuilder, RuntimeDevice,
        RuntimeLibraryConfig, RuntimeLibraryConfigBuilder, RuntimeLibraryConfigOverride,
        RuntimeLibraryConfigOverrideBuilder, RuntimeProfileKind, SessionPolicy,
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
            .provider_policy(ProviderPolicy::Interactive)
            .provider_dir("runtime/providers")
            .intra_threads(2)
            .inter_threads(1)
            .fgclip_max_patches(256)
            .session_policy(SessionPolicy::KeepBothLoaded)
            .graph_optimization_level(GraphOptimizationLevel::Basic)
            .build()
            .unwrap();

        assert_eq!(actual.device, RuntimeDevice::Gpu);
        assert_eq!(actual.provider_policy, ProviderPolicy::Interactive);
        assert_eq!(
            actual.library.provider_dir,
            Some("runtime/providers".into())
        );
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
            .provider_dir("runtime/providers")
            .clear_provider_dir()
            .inter_threads(2)
            .clear_inter_threads()
            .fgclip_max_patches(256)
            .clear_fgclip_max_patches()
            .build()
            .unwrap();

        assert_eq!(actual.library.provider_dir, None);
        assert!(actual.nvidia_library.is_empty());
        assert_eq!(actual.inter_threads, None);
        assert_eq!(actual.fgclip_max_patches, None);
    }

    #[test]
    fn runtime_library_builder_overrides_selected_fields_only() {
        let actual = RuntimeLibraryConfig::builder()
            .provider_dir("runtime/providers")
            .cuda_bin_dir("cuda/bin")
            .preload(false)
            .build()
            .unwrap();

        assert_eq!(actual.provider_dir, Some("runtime/providers".into()));
        assert_eq!(actual.cuda_bin_dir, Some("cuda/bin".into()));
        assert!(!actual.preload);
        assert_eq!(actual.cudnn_bin_dir, None);
    }

    #[test]
    fn runtime_library_builder_rejects_empty_paths() {
        let error = RuntimeLibraryConfigBuilder::new()
            .provider_dir("")
            .build()
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("runtime.library.provider_dir must not be an empty path")
        );
    }

    #[test]
    fn runtime_library_override_resolves_against_base_config() {
        let base = RuntimeLibraryConfig::builder()
            .provider_dir("runtime/providers")
            .cuda_bin_dir("cuda/bin")
            .preload(false)
            .build()
            .unwrap();
        let override_config = RuntimeLibraryConfigOverride::builder()
            .ort_dylib_path("runtime/directml/onnxruntime.dll")
            .preload(true)
            .build()
            .unwrap();

        let resolved = override_config.resolve(&base);

        assert_eq!(
            resolved.ort_dylib_path,
            Some("runtime/directml/onnxruntime.dll".into())
        );
        assert_eq!(resolved.provider_dir, Some("runtime/providers".into()));
        assert_eq!(resolved.cuda_bin_dir, Some("cuda/bin".into()));
        assert!(resolved.preload);
    }

    #[test]
    fn runtime_library_override_builder_rejects_empty_paths() {
        let error = RuntimeLibraryConfigOverrideBuilder::new()
            .provider_dir("")
            .build()
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("runtime.library_override.provider_dir must not be an empty path")
        );
    }

    #[test]
    fn runtime_builder_can_set_family_specific_library_overrides() {
        let directml = RuntimeLibraryConfigOverride::builder()
            .ort_dylib_path("runtime/directml/onnxruntime.dll")
            .build()
            .unwrap();

        let actual = RuntimeConfig::builder()
            .provider_dir("runtime/providers")
            .directml_runtime_library_config(directml.clone())
            .build()
            .unwrap();

        assert_eq!(actual.directml_library, directml);
        assert_eq!(
            actual
                .resolved_library_for_profile(RuntimeProfileKind::DirectMl)
                .provider_dir,
            Some("runtime/providers".into())
        );
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

    #[test]
    fn provider_policy_deserializes_snake_case_and_legacy_pascal_case() {
        let snake_case: ProviderPolicy = serde_json::from_str(r#""interactive""#).unwrap();
        let legacy_pascal_case: ProviderPolicy = serde_json::from_str(r#""Interactive""#).unwrap();

        assert_eq!(snake_case, ProviderPolicy::Interactive);
        assert_eq!(legacy_pascal_case, ProviderPolicy::Interactive);
    }
}
