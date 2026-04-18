# SDK Runtime Update Note

本次更新主要补了运行时可观测性和 EP 选择语义。

## 新增

- `OmniSearch::runtime_snapshot()`
- 可序列化运行时状态：
  - `config.requested_device`
  - `summary.mode`
  - `summary.effective_provider`
  - `text_session` / `image_session`
  - `planned_providers`
  - `provider_attempts`
  - `registered_providers`
  - `last_error`

## 行为变化

- `RuntimeDevice::Gpu` 不再允许静默回退到 CPU。
  如果所有 GPU EP 都不可用，会直接返回错误。
- `RuntimeDevice::Auto` 仍会按 provider 链自动尝试，并在需要时回退到 CPU。

## Provider 策略

- 默认构建：
  - Windows：`DirectML -> CPU`
  - Apple：`CoreML -> CPU`
- 启用 `--features nvidia` 后：
  - Windows：`TensorRT -> CUDA -> DirectML -> CPU`
  - Linux x64：`TensorRT -> CUDA -> CPU`

## 打包说明

- Windows 发布包需要把输出目录里的 `DirectML.dll` 一起带上。
- 应用侧现在可以直接展示实际 provider，例如：
  `tensorrt`、`cuda`、`directml`、`coreml`、`cpu`。
