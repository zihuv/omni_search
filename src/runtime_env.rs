use std::env;
use std::path::{Path, PathBuf};

use crate::config::{
    ProviderPolicy, RuntimeConfig, RuntimeDevice, RuntimeLibraryConfigOverride,
    RuntimeLibraryConfigOverrideBuilder,
};
use crate::error::Error;

const FALLBACK_THREAD_COUNT: usize = 4;

pub fn physical_core_count() -> Option<usize> {
    match num_cpus::get_physical() {
        0 => None,
        count => Some(count),
    }
}

pub fn logical_core_count() -> Option<usize> {
    std::thread::available_parallelism()
        .ok()
        .map(|parallelism| parallelism.get())
        .filter(|count| *count > 0)
}

pub fn default_intra_threads() -> usize {
    physical_core_count()
        .or_else(logical_core_count)
        .unwrap_or(FALLBACK_THREAD_COUNT)
}

pub fn load_dotenv_from(root: impl AsRef<Path>) -> Result<Option<PathBuf>, Error> {
    let path = root.as_ref().join(".env");
    if !path.is_file() {
        return Ok(None);
    }

    dotenvy::from_path(&path).map_err(|error| {
        Error::invalid_config(format!("failed to load {}: {error}", path.display()))
    })?;
    Ok(Some(path))
}

pub fn env_path(name: &str) -> Option<PathBuf> {
    env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

pub fn env_path_resolved(name: &str, root: impl AsRef<Path>) -> Option<PathBuf> {
    let root = root.as_ref();
    env_path(name).map(|path| {
        if path.is_absolute() {
            path
        } else {
            root.join(path)
        }
    })
}

pub fn env_runtime_device(name: &str) -> Result<Option<RuntimeDevice>, Error> {
    let Some(value) = env_string(name)? else {
        return Ok(None);
    };
    let device = match value.to_ascii_lowercase().as_str() {
        "auto" => RuntimeDevice::Auto,
        "cpu" => RuntimeDevice::Cpu,
        "gpu" => RuntimeDevice::Gpu,
        _ => {
            return Err(Error::invalid_config(format!(
                "unsupported {name}='{value}', expected one of: auto, cpu, gpu"
            )));
        }
    };
    Ok(Some(device))
}

pub fn env_provider_policy(name: &str) -> Result<Option<ProviderPolicy>, Error> {
    let Some(value) = env_string(name)? else {
        return Ok(None);
    };
    let policy = match value.to_ascii_lowercase().as_str() {
        "auto" => ProviderPolicy::Auto,
        "interactive" => ProviderPolicy::Interactive,
        "service" => ProviderPolicy::Service,
        _ => {
            return Err(Error::invalid_config(format!(
                "unsupported {name}='{value}', expected one of: auto, interactive, service"
            )));
        }
    };
    Ok(Some(policy))
}

pub fn env_intra_threads(name: &str) -> Result<Option<usize>, Error> {
    let Some(value) = env_string(name)? else {
        return Ok(None);
    };
    parse_intra_threads(name, &value).map(Some)
}

pub fn env_positive_usize(name: &str) -> Result<Option<usize>, Error> {
    let Some(value) = env_string(name)? else {
        return Ok(None);
    };
    parse_positive_usize(name, &value).map(Some)
}

pub fn env_bool(name: &str) -> Result<Option<bool>, Error> {
    let Some(value) = env_string(name)? else {
        return Ok(None);
    };
    parse_bool(name, &value).map(Some)
}

pub fn runtime_config_from_env() -> Result<RuntimeConfig, Error> {
    let mut builder = RuntimeConfig::builder();
    if let Some(device) = env_runtime_device("OMNI_DEVICE")? {
        builder.device(device);
    }
    if let Some(policy) = env_provider_policy("OMNI_PROVIDER_POLICY")? {
        builder.provider_policy(policy);
    }
    if let Some(intra_threads) = env_intra_threads("OMNI_INTRA_THREADS")? {
        builder.intra_threads(intra_threads);
    }
    if let Some(inter_threads) = env_positive_usize("OMNI_INTER_THREADS")? {
        builder.inter_threads(inter_threads);
    }
    if let Some(fgclip_max_patches) = env_positive_usize("OMNI_FGCLIP_MAX_PATCHES")? {
        builder.fgclip_max_patches(fgclip_max_patches);
    }
    if let Some(path) = env_path("OMNI_ORT_DYLIB_PATH").or_else(|| env_path("ORT_DYLIB_PATH")) {
        builder.ort_dylib_path(path);
    }
    if let Some(path) = env_path("OMNI_ORT_PROVIDER_DIR") {
        builder.provider_dir(path);
    }
    if let Some(path) = env_path("OMNI_CUDA_BIN_DIR") {
        builder.cuda_bin_dir(path);
    }
    if let Some(path) = env_path("OMNI_CUDNN_BIN_DIR") {
        builder.cudnn_bin_dir(path);
    }
    if let Some(path) = env_path("OMNI_TENSORRT_LIB_DIR") {
        builder.tensorrt_lib_dir(path);
    }
    if let Some(preload) = env_bool("OMNI_PRELOAD_RUNTIME_LIBRARIES")? {
        builder.preload_runtime_libraries(preload);
    }
    if let Some(config) =
        runtime_library_override_from_env("OMNI_NVIDIA", &["OMNI_NVIDIA"], true, true)?
    {
        builder.nvidia_runtime_library_config(config);
    }
    if let Some(config) = runtime_library_override_from_env(
        "OMNI_DIRECTML",
        &["OMNI_DIRECTML", "OMNI_DML"],
        false,
        false,
    )? {
        builder.directml_runtime_library_config(config);
    }
    if let Some(config) =
        runtime_library_override_from_env("OMNI_COREML", &["OMNI_COREML"], false, false)?
    {
        builder.coreml_runtime_library_config(config);
    }
    builder.build()
}

fn runtime_library_override_from_env(
    _label: &str,
    prefixes: &[&str],
    include_cuda: bool,
    include_tensorrt: bool,
) -> Result<Option<RuntimeLibraryConfigOverride>, Error> {
    let mut builder = RuntimeLibraryConfigOverrideBuilder::new();
    let mut configured = false;

    if let Some(path) = env_path_any(prefixes, "ORT_DYLIB_PATH") {
        builder.ort_dylib_path(path);
        configured = true;
    }
    if let Some(path) = env_path_any(prefixes, "ORT_PROVIDER_DIR") {
        builder.provider_dir(path);
        configured = true;
    }
    if include_cuda {
        if let Some(path) = env_path_any(prefixes, "CUDA_BIN_DIR") {
            builder.cuda_bin_dir(path);
            configured = true;
        }
        if let Some(path) = env_path_any(prefixes, "CUDNN_BIN_DIR") {
            builder.cudnn_bin_dir(path);
            configured = true;
        }
    }
    if include_tensorrt {
        if let Some(path) = env_path_any(prefixes, "TENSORRT_LIB_DIR") {
            builder.tensorrt_lib_dir(path);
            configured = true;
        }
    }
    if let Some(preload) = env_bool_any(prefixes, "PRELOAD_RUNTIME_LIBRARIES")? {
        builder.preload(preload);
        configured = true;
    }

    if !configured {
        return Ok(None);
    }

    Ok(Some(builder.build()?))
}

fn env_path_any(prefixes: &[&str], suffix: &str) -> Option<PathBuf> {
    prefixes
        .iter()
        .find_map(|prefix| env_path(&format!("{prefix}_{suffix}")))
}

fn env_bool_any(prefixes: &[&str], suffix: &str) -> Result<Option<bool>, Error> {
    for prefix in prefixes {
        let name = format!("{prefix}_{suffix}");
        if let Some(value) = env_bool(&name)? {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

fn env_string(name: &str) -> Result<Option<String>, Error> {
    let Some(value) = env::var_os(name) else {
        return Ok(None);
    };
    let value = value
        .into_string()
        .map_err(|_| Error::invalid_config(format!("{name} must be valid UTF-8")))?;
    let value = value.trim().to_owned();
    if value.is_empty() {
        return Ok(None);
    }
    Ok(Some(value))
}

fn parse_positive_usize(name: &str, value: &str) -> Result<usize, Error> {
    let parsed = value.parse::<usize>().map_err(|error| {
        Error::invalid_config(format!(
            "failed to parse {name}='{value}' as a positive integer: {error}"
        ))
    })?;
    if parsed == 0 {
        return Err(Error::invalid_config(format!(
            "{name} must be greater than 0"
        )));
    }
    Ok(parsed)
}

fn parse_intra_threads(name: &str, value: &str) -> Result<usize, Error> {
    if value.eq_ignore_ascii_case("auto") {
        return Ok(default_intra_threads());
    }
    parse_positive_usize(name, value)
}

fn parse_bool(name: &str, value: &str) -> Result<bool, Error> {
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(Error::invalid_config(format!(
            "unsupported {name}='{value}', expected one of: true, false, 1, 0, yes, no, on, off"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{
        default_intra_threads, env_bool, env_path_resolved, env_provider_policy,
        parse_intra_threads, parse_positive_usize, physical_core_count, runtime_config_from_env,
    };
    use crate::config::ProviderPolicy;

    #[test]
    fn default_intra_threads_is_always_positive() {
        assert!(default_intra_threads() > 0);
    }

    #[test]
    fn default_intra_threads_prefers_physical_cores_when_available() {
        if let Some(physical) = physical_core_count() {
            assert_eq!(default_intra_threads(), physical);
        }
    }

    #[test]
    fn parse_positive_usize_accepts_positive_values() {
        assert_eq!(parse_positive_usize("OMNI_THREADS", "12").unwrap(), 12);
    }

    #[test]
    fn parse_intra_threads_accepts_auto() {
        assert_eq!(
            parse_intra_threads("OMNI_INTRA_THREADS", "auto").unwrap(),
            default_intra_threads()
        );
    }

    #[test]
    fn env_provider_policy_accepts_interactive() {
        let parsed = {
            unsafe {
                std::env::set_var("OMNI_PROVIDER_POLICY_TEST", "interactive");
            }
            env_provider_policy("OMNI_PROVIDER_POLICY_TEST").unwrap()
        };
        assert_eq!(parsed, Some(ProviderPolicy::Interactive));
        unsafe {
            std::env::remove_var("OMNI_PROVIDER_POLICY_TEST");
        }
    }

    #[test]
    fn env_bool_accepts_false_values() {
        let parsed = {
            unsafe {
                std::env::set_var("OMNI_PRELOAD_RUNTIME_LIBRARIES_TEST", "off");
            }
            env_bool("OMNI_PRELOAD_RUNTIME_LIBRARIES_TEST").unwrap()
        };
        assert_eq!(parsed, Some(false));
        unsafe {
            std::env::remove_var("OMNI_PRELOAD_RUNTIME_LIBRARIES_TEST");
        }
    }

    #[test]
    fn parse_positive_usize_rejects_zero() {
        let error = parse_positive_usize("OMNI_THREADS", "0").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("OMNI_THREADS must be greater than 0")
        );
    }

    #[test]
    fn parse_positive_usize_rejects_non_numeric_values() {
        let error = parse_positive_usize("OMNI_THREADS", "auto").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("failed to parse OMNI_THREADS='auto' as a positive integer")
        );
    }

    #[test]
    fn env_path_resolved_uses_root_for_relative_paths() {
        let root = Path::new(r"D:\repo");
        let resolved = {
            unsafe {
                std::env::set_var("OMNI_TEST_PATH_RESOLVED", "models/fgclip2_flat");
            }
            env_path_resolved("OMNI_TEST_PATH_RESOLVED", root)
        };
        assert_eq!(resolved, Some(root.join("models/fgclip2_flat")));
        unsafe {
            std::env::remove_var("OMNI_TEST_PATH_RESOLVED");
        }
    }

    #[test]
    fn runtime_config_from_env_reads_nvidia_runtime_override() {
        unsafe {
            std::env::set_var(
                "OMNI_NVIDIA_ORT_DYLIB_PATH",
                r"D:\runtime\nvidia\onnxruntime.dll",
            );
            std::env::set_var("OMNI_NVIDIA_CUDA_BIN_DIR", r"D:\runtime\nvidia\cuda");
        }

        let config = runtime_config_from_env().unwrap();

        assert_eq!(
            config.nvidia_library.ort_dylib_path,
            Some(r"D:\runtime\nvidia\onnxruntime.dll".into())
        );
        assert_eq!(
            config.nvidia_library.cuda_bin_dir,
            Some(r"D:\runtime\nvidia\cuda".into())
        );

        unsafe {
            std::env::remove_var("OMNI_NVIDIA_ORT_DYLIB_PATH");
            std::env::remove_var("OMNI_NVIDIA_CUDA_BIN_DIR");
        }
    }

    #[test]
    fn runtime_config_from_env_reads_directml_runtime_override_alias() {
        unsafe {
            std::env::set_var("OMNI_DML_ORT_DYLIB_PATH", r"D:\runtime\dml\onnxruntime.dll");
            std::env::set_var("OMNI_DML_PRELOAD_RUNTIME_LIBRARIES", "false");
        }

        let config = runtime_config_from_env().unwrap();

        assert_eq!(
            config.directml_library.ort_dylib_path,
            Some(r"D:\runtime\dml\onnxruntime.dll".into())
        );
        assert_eq!(config.directml_library.preload, Some(false));

        unsafe {
            std::env::remove_var("OMNI_DML_ORT_DYLIB_PATH");
            std::env::remove_var("OMNI_DML_PRELOAD_RUNTIME_LIBRARIES");
        }
    }
}
