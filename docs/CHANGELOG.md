# Changelog

本文档记录所有值得用户关注的变更。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Added

### Changed

### Deprecated

### Removed

### Fixed

### Security

## [0.2.6] - 2026-04-20

### Added

- 新增可选 `avif` crate feature；启用后，CLI 样本扫描与图片路径嵌入可识别 `.avif` 文件。

### Fixed

- 默认构建不再因为缺少 `pkg-config` 或 `dav1d` 而在 Windows 上阻塞 `cargo test` 与发布流程。

## [0.2.5] - 2026-04-20

### Added

- 新增运行时库配置层，支持显式指定 ORT 主库、provider 目录、CUDA、cuDNN 与 TensorRT 路径，并控制是否在初始化时预加载动态库。
- 新增按 runtime family 区分的运行时动态库覆盖配置，可分别为 `nvidia`、`directml` 与 `coreml` 指定独立的 ORT/provider 路径与预加载策略。

### Changed

- 重构 crate feature 组合，显式区分 `runtime-bundled` / `runtime-dynamic` 运行时模式，以及 `directml`、`coreml`、`nvidia` provider 能力。
- 细化运行时 provider 诊断，新增区分“当前 ORT 运行时不支持该 provider”与“provider 动态库与当前 ORT 不兼容”的错误分类。
- 将 `auto` / `interactive` / `service` 升级为内置 runtime plan：`interactive` 默认优先 `nvidia(cuda) -> directml -> cpu`，`service` 默认优先 `nvidia(tensorrt -> cuda) -> directml -> cpu`。
- 扩展 `runtime_snapshot` 输出，新增 `planned_profiles` 与 `selected_profile`，便于确认当前策略计划与最终命中的 runtime family。

### Fixed

- 增强 `runtime_snapshot` 诊断信息，新增 `compiled_providers` 与 `issues` 输出，可区分 provider 未编译、provider 库缺失、依赖链缺失与注册失败等情况。
- 修复动态运行时预加载与 provider 预注册时的误报问题；当 CUDA 或 TensorRT 依赖库预处理失败但 provider 后续仍可正常启用时，不再提前标记为异常。
- 修复 GPU 实际已启用时 `runtime_summary.reason` 仍暴露次级 provider 失败原因的问题，避免 CUDA 已生效时被 TensorRT/DirectML 次要失败干扰判断。
- 修复运行时动态库初始化失败后会污染后续 profile 尝试的问题；当首个 runtime profile 因配置缺失或路径错误未能完成初始化时，后续 family 现在仍可继续探测。
