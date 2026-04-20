use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ort::environment::Environment;

use crate::config::RuntimeLibraryConfig;
use crate::error::Error;
use crate::runtime::{ExecutionProviderKind, RuntimeIssue, RuntimeIssueCode};

pub(crate) struct RuntimeLibraryPreparation {
    pub issues: Vec<RuntimeIssue>,
}

pub(crate) struct RuntimeLibraryError {
    pub error: Error,
    pub issues: Vec<RuntimeIssue>,
}

struct GlobalRuntimeLibraryState {
    config: Option<RuntimeLibraryConfig>,
    status: RuntimeLibraryStatus,
}

enum RuntimeLibraryStatus {
    Pending,
    Ready { issues: Vec<RuntimeIssue> },
}

static RUNTIME_LIBRARY_STATE: Mutex<GlobalRuntimeLibraryState> =
    Mutex::new(GlobalRuntimeLibraryState {
        config: None,
        status: RuntimeLibraryStatus::Pending,
    });

pub(crate) fn prepare_runtime_libraries(
    config: &RuntimeLibraryConfig,
    planned_providers: &[ExecutionProviderKind],
) -> Result<RuntimeLibraryPreparation, RuntimeLibraryError> {
    {
        let mut state = RUNTIME_LIBRARY_STATE
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(existing) = &state.config {
            if existing != config {
                return Err(conflicting_runtime_library_config(existing, config));
            }
        } else {
            state.config = Some(config.clone());
        }

        match &state.status {
            RuntimeLibraryStatus::Ready { issues } => {
                return Ok(RuntimeLibraryPreparation {
                    issues: issues.clone(),
                });
            }
            RuntimeLibraryStatus::Pending => {}
        }
    }

    let result = perform_prepare_runtime_libraries(config, planned_providers);

    let mut state = RUNTIME_LIBRARY_STATE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    match &result {
        Ok(prepared) => {
            state.status = RuntimeLibraryStatus::Ready {
                issues: prepared.issues.clone(),
            };
        }
        Err(error) => {
            let _ = error;
            state.config = None;
            state.status = RuntimeLibraryStatus::Pending;
        }
    }
    result
}

fn perform_prepare_runtime_libraries(
    config: &RuntimeLibraryConfig,
    planned_providers: &[ExecutionProviderKind],
) -> Result<RuntimeLibraryPreparation, RuntimeLibraryError> {
    if let Some(path) = &config.ort_dylib_path {
        apply_ort_dylib_path(path)?;
    }

    let mut issues = Vec::new();
    let planned_gpu_providers = unique_gpu_providers(planned_providers);

    if config.preload {
        preload_cuda_dependencies(config, &planned_gpu_providers, &mut issues);
        preload_tensorrt_dependencies(config, &planned_gpu_providers, &mut issues);
        preload_provider_shared_library(config, &planned_gpu_providers, &mut issues);
    }

    if config.provider_dir.is_some()
        && planned_gpu_providers.iter().any(|provider| {
            matches!(
                provider,
                ExecutionProviderKind::Cuda | ExecutionProviderKind::TensorRt
            )
        })
    {
        let env = Environment::current().map_err(|error| {
            fatal_runtime_library_error(
                RuntimeIssueCode::RuntimeLibraryMissing,
                format!("failed to initialize ONNX Runtime environment: {error}"),
            )
        })?;
        register_provider_libraries(&env, config, &planned_gpu_providers, &mut issues);
    }

    Ok(RuntimeLibraryPreparation { issues })
}

fn conflicting_runtime_library_config(
    existing: &RuntimeLibraryConfig,
    requested: &RuntimeLibraryConfig,
) -> RuntimeLibraryError {
    let message = format!(
        "runtime libraries are process-global and were already initialized with a different configuration: existing={existing:?}, requested={requested:?}"
    );
    RuntimeLibraryError {
        error: Error::invalid_config(message.clone()),
        issues: vec![runtime_issue(
            RuntimeIssueCode::RuntimeLibraryConfigurationUnsupported,
            message,
        )],
    }
}

fn apply_ort_dylib_path(path: &Path) -> Result<(), RuntimeLibraryError> {
    if !cfg!(feature = "runtime-dynamic") {
        return Err(fatal_runtime_library_error(
            RuntimeIssueCode::RuntimeLibraryConfigurationUnsupported,
            "runtime.library.ort_dylib_path requires the `runtime-dynamic` crate feature",
        ));
    }
    if !path.is_file() {
        return Err(fatal_runtime_library_error(
            RuntimeIssueCode::RuntimeLibraryMissing,
            format!(
                "configured ONNX Runtime library does not exist: {}",
                path.display()
            ),
        ));
    }

    unsafe {
        std::env::set_var("ORT_DYLIB_PATH", path);
    }
    Ok(())
}

fn preload_cuda_dependencies(
    config: &RuntimeLibraryConfig,
    planned_providers: &[ExecutionProviderKind],
    _issues: &mut Vec<RuntimeIssue>,
) {
    if !planned_providers.iter().any(|provider| {
        matches!(
            provider,
            ExecutionProviderKind::Cuda | ExecutionProviderKind::TensorRt
        )
    }) {
        return;
    }
    if config.cuda_bin_dir.is_none() && config.cudnn_bin_dir.is_none() {
        return;
    }

    if let Err(error) = ort::ep::cuda::preload_dylibs(
        config.cuda_bin_dir.as_deref(),
        config.cudnn_bin_dir.as_deref(),
    ) {
        let _ = error;
    }
}

fn preload_tensorrt_dependencies(
    config: &RuntimeLibraryConfig,
    planned_providers: &[ExecutionProviderKind],
    issues: &mut Vec<RuntimeIssue>,
) {
    if !planned_providers.contains(&ExecutionProviderKind::TensorRt) {
        return;
    }
    let Some(root) = config.tensorrt_lib_dir.as_ref() else {
        return;
    };
    if !root.is_dir() {
        issues.push(runtime_issue(
            RuntimeIssueCode::DependencyLibraryMissing,
            format!(
                "configured TensorRT library directory does not exist: {}",
                root.display()
            ),
        ));
        return;
    }

    let mut candidates = match tensorrt_dependency_candidates(root) {
        Ok(candidates) => candidates,
        Err(error) => {
            issues.push(runtime_issue(
                RuntimeIssueCode::DependencyLibraryMissing,
                format!(
                    "failed to enumerate TensorRT libraries under {}: {error}",
                    root.display()
                ),
            ));
            return;
        }
    };
    if candidates.is_empty() {
        issues.push(runtime_issue(
            RuntimeIssueCode::DependencyLibraryMissing,
            format!("no TensorRT libraries were found under {}", root.display()),
        ));
        return;
    }

    candidates.sort();
    for candidate in candidates {
        if let Err(error) = ort::util::preload_dylib(&candidate) {
            let _ = error;
        }
    }
}

fn preload_provider_shared_library(
    config: &RuntimeLibraryConfig,
    planned_providers: &[ExecutionProviderKind],
    issues: &mut Vec<RuntimeIssue>,
) {
    if !planned_providers.iter().any(|provider| {
        matches!(
            provider,
            ExecutionProviderKind::Cuda | ExecutionProviderKind::TensorRt
        )
    }) {
        return;
    }
    let Some(provider_dir) = config.provider_dir.as_ref() else {
        return;
    };

    let shared_path = provider_dir.join(shared_provider_library_name());
    if !shared_path.is_file() {
        issues.push(runtime_issue(
            RuntimeIssueCode::ProviderLibraryMissing,
            format!(
                "provider directory {} does not contain {}",
                provider_dir.display(),
                shared_provider_library_name()
            ),
        ));
        return;
    }

    if let Err(error) = ort::util::preload_dylib(&shared_path) {
        let _ = error;
    }
}

fn register_provider_libraries(
    env: &std::sync::Arc<Environment>,
    config: &RuntimeLibraryConfig,
    planned_providers: &[ExecutionProviderKind],
    issues: &mut Vec<RuntimeIssue>,
) {
    let Some(provider_dir) = config.provider_dir.as_ref() else {
        return;
    };

    for provider in planned_providers {
        let Some(library_name) = provider_library_name(*provider) else {
            continue;
        };
        let path = provider_dir.join(library_name);
        if !path.is_file() {
            issues.push(runtime_issue(
                RuntimeIssueCode::ProviderLibraryMissing,
                format!(
                    "provider directory {} does not contain the {} library {}",
                    provider_dir.display(),
                    provider_name(*provider),
                    library_name
                ),
            ));
            continue;
        }

        if config.preload {
            if let Err(error) = ort::util::preload_dylib(&path) {
                let _ = error;
            }
        }

        // ORT's bundled provider DLLs can still be usable even when the plugin-style
        // EP registration entrypoint is absent, so keep this registration best-effort.
        let _ = env.register_ep_library(provider_registration_name(*provider), &path);
    }
}

fn unique_gpu_providers(planned_providers: &[ExecutionProviderKind]) -> Vec<ExecutionProviderKind> {
    let mut unique = Vec::new();
    for provider in planned_providers {
        if provider.is_gpu() && !unique.contains(provider) {
            unique.push(*provider);
        }
    }
    unique
}

fn tensorrt_dependency_candidates(root: &Path) -> Result<Vec<PathBuf>, std::io::Error> {
    let mut paths = fs::read_dir(root)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(tensorrt_dependency_name_matches)
        })
        .collect::<Vec<_>>();
    paths.sort();
    Ok(paths)
}

fn tensorrt_dependency_name_matches(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    name.starts_with("nvinfer") || name.starts_with("nvonnxparser") || name.starts_with("nvparsers")
}

fn provider_library_name(provider: ExecutionProviderKind) -> Option<&'static str> {
    match provider {
        ExecutionProviderKind::TensorRt => Some(tensorrt_provider_library_name()),
        ExecutionProviderKind::Cuda => Some(cuda_provider_library_name()),
        ExecutionProviderKind::DirectMl
        | ExecutionProviderKind::CoreMl
        | ExecutionProviderKind::Cpu => None,
    }
}

fn provider_registration_name(provider: ExecutionProviderKind) -> &'static str {
    match provider {
        ExecutionProviderKind::TensorRt => "omni_search.tensorrt",
        ExecutionProviderKind::Cuda => "omni_search.cuda",
        ExecutionProviderKind::DirectMl => "omni_search.directml",
        ExecutionProviderKind::CoreMl => "omni_search.coreml",
        ExecutionProviderKind::Cpu => "omni_search.cpu",
    }
}

fn provider_name(provider: ExecutionProviderKind) -> &'static str {
    match provider {
        ExecutionProviderKind::TensorRt => "TensorRT",
        ExecutionProviderKind::Cuda => "CUDA",
        ExecutionProviderKind::DirectMl => "DirectML",
        ExecutionProviderKind::CoreMl => "CoreML",
        ExecutionProviderKind::Cpu => "CPU",
    }
}

#[cfg(target_os = "windows")]
fn shared_provider_library_name() -> &'static str {
    "onnxruntime_providers_shared.dll"
}

#[cfg(not(target_os = "windows"))]
fn shared_provider_library_name() -> &'static str {
    "libonnxruntime_providers_shared.so"
}

#[cfg(target_os = "windows")]
fn cuda_provider_library_name() -> &'static str {
    "onnxruntime_providers_cuda.dll"
}

#[cfg(not(target_os = "windows"))]
fn cuda_provider_library_name() -> &'static str {
    "libonnxruntime_providers_cuda.so"
}

#[cfg(target_os = "windows")]
fn tensorrt_provider_library_name() -> &'static str {
    "onnxruntime_providers_tensorrt.dll"
}

#[cfg(not(target_os = "windows"))]
fn tensorrt_provider_library_name() -> &'static str {
    "libonnxruntime_providers_tensorrt.so"
}

fn fatal_runtime_library_error(
    code: RuntimeIssueCode,
    message: impl Into<String>,
) -> RuntimeLibraryError {
    let message = message.into();
    RuntimeLibraryError {
        error: Error::ort(message.clone()),
        issues: vec![runtime_issue(code, message)],
    }
}

fn runtime_issue(code: RuntimeIssueCode, message: impl Into<String>) -> RuntimeIssue {
    RuntimeIssue {
        code,
        message: message.into(),
        at_unix_ms: now_unix_ms(),
    }
}

fn now_unix_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};

    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{apply_ort_dylib_path, tensorrt_dependency_name_matches};
    use crate::runtime::{RuntimeIssueCode, looks_like_missing_dependency};

    #[test]
    fn missing_dependency_classifier_matches_common_loader_errors() {
        assert!(looks_like_missing_dependency(
            "LoadLibrary failed with error 126: The specified module could not be found."
        ));
        assert!(looks_like_missing_dependency(
            "dlopen failed: image not found"
        ));
    }

    #[test]
    fn tensorrt_dependency_matcher_accepts_known_prefixes() {
        assert!(tensorrt_dependency_name_matches("nvinfer_10.dll"));
        assert!(tensorrt_dependency_name_matches("nvonnxparser_10.dll"));
        assert!(tensorrt_dependency_name_matches("nvparsers.dll"));
        assert!(!tensorrt_dependency_name_matches("DirectML.dll"));
    }

    #[cfg(not(feature = "runtime-dynamic"))]
    #[test]
    fn ort_dylib_path_requires_runtime_dynamic_at_loader_time() {
        let error = apply_ort_dylib_path(Path::new("D:/runtime/onnxruntime.dll")).unwrap_err();

        assert!(error.error.to_string().contains(
            "runtime.library.ort_dylib_path requires the `runtime-dynamic` crate feature"
        ));
        assert_eq!(
            error.issues.first().map(|issue| issue.code),
            Some(RuntimeIssueCode::RuntimeLibraryConfigurationUnsupported)
        );
    }
}
