# Runtime Provider Policy Update

`omni_search` now exposes a formal runtime provider policy instead of relying only on platform defaults.

## What Changed

- `RuntimeConfig` and `OmniSearch::builder()` now support `provider_policy`
- environment loading now supports `OMNI_PROVIDER_POLICY`
- `runtime_snapshot().config` now includes the requested provider policy

## Policy Values

- `auto`: keep the default platform/provider order
- `interactive`: prefer lower warmup cost and compatibility
- `service`: prefer steady-state throughput for long-lived processes

On Windows with `cargo build --features nvidia`:

- `service`: `TensorRT -> CUDA -> DirectML -> CPU`
- `interactive`: `CUDA -> DirectML -> TensorRT -> CPU`

## Example

```dotenv
OMNI_DEVICE=auto
OMNI_PROVIDER_POLICY=service
```

```rust
use omni_search::{OmniSearch, ProviderPolicy, RuntimeDevice};

let sdk = OmniSearch::builder()
    .from_local_model_dir("D:/models/fgclip2_flat")
    .device(RuntimeDevice::Auto)
    .provider_policy(ProviderPolicy::Service)
    .build()?;
```

`OMNI_FORCE_PROVIDER` is still supported as a diagnostics-only override when you need to pin one execution provider for benchmarking or debugging.
