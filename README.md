# omni_search

`omni_search` is a Rust SDK for multimodal embedding and similarity search over local ONNX model directories.

Current scope:

- load a flat local model directory with root-level `model_config.json`;
- compute text embeddings;
- compute image embeddings;
- compare embeddings with cosine similarity;
- expose runtime snapshots with requested device, planned providers, registered providers, and effective provider;
- manually unload ONNX Runtime sessions.

Supported families:

- `chinese_clip`
- `fg_clip`
- `open_clip`

The published crate does not bundle ONNX models or sample images. Point it at your own local assets with `OMNI_BUNDLE_DIR` and `OMNI_SAMPLES_DIR`.

The CLI binaries automatically load a repo-root `.env` file when present. Existing shell environment variables still win, so `.env` works as a local default layer.

Quickstart:

- create a `.env` from `.env.example` or edit the existing local `.env` defaults when you want to pin a bundle, sample directory, or test fixtures;
- set `OMNI_BUNDLE_DIR` to a local model directory that contains `model_config.json` plus flat root-level assets;
- set `OMNI_SAMPLES_DIR` to a directory containing one or more `.jpg`, `.jpeg`, `.png`, `.webp`, or `.bmp` images;
- build SDK instances with `OmniSearch::builder()` when you only want to override part of the runtime config and keep the rest at defaults;
- run `cargo run --bin omni_search --release` to scan all images in `OMNI_SAMPLES_DIR` with the default query `"山"`;
- run `cargo run --bin omni_search --release -- "海边"` to scan all images with a custom query;
- run `cargo run --bin omni_search --release -- "海边" 20` to print a different top-k;
- run `cargo run --bin omni_search --release -- "海边" "/absolute/path/to/query.jpg"` to run `image_to_image` with a specific query image;
- set `OMNI_DEVICE` to `auto`, `cpu`, or `gpu` to control execution provider selection; default is `auto`;
- set `OMNI_PROVIDER_POLICY` to `auto`, `interactive`, or `service` when you want to change GPU provider priority without pinning a single provider;
- call `sdk.runtime_snapshot()` when you need to inspect or display the current execution provider state;
- the default `RuntimeConfig::intra_threads` value also resolves to the host physical core count;
- set `OMNI_INTRA_THREADS` to `auto` or a positive integer to override the ONNX Runtime intra-op thread count; `auto` resolves to the host physical core count;
- set `OMNI_INTER_THREADS` to a positive integer when you need to override the ONNX Runtime inter-op thread count while benchmarking or tuning;
- set `OMNI_FGCLIP_MAX_PATCHES` to cap FG-CLIP2 image preprocessing at a smaller bucket without changing the exported model directory;
- recommended `OMNI_FGCLIP_MAX_PATCHES` values are `128`, `256`, `576`, `784`, or `1024`;
- run `cargo test --test quickstart -- --ignored --nocapture` to execute the smoke test after setting `OMNI_TEST_BUNDLE_DIR` and `OMNI_TEST_SAMPLE_IMAGE`.

Example `.env`:

```dotenv
OMNI_DEVICE=auto
OMNI_PROVIDER_POLICY=auto
OMNI_BUNDLE_DIR=models/fgclip2_flat
OMNI_SAMPLES_DIR=samples
```

Device selection notes:

- all current model families load standard ONNX graphs, so GPU support is determined by the ONNX Runtime execution provider rather than by a model-specific code path;
- the default build keeps Windows `DirectML` and Apple `CoreML` enabled;
- enable `cargo build --features nvidia` when you want `TensorRT -> CUDA` ahead of the platform fallback provider on supported Windows/Linux x64 targets;
- `OMNI_PROVIDER_POLICY=service` prefers steady-state throughput and currently tries `TensorRT -> CUDA -> DirectML -> CPU` on Windows with `--features nvidia`;
- `OMNI_PROVIDER_POLICY=interactive` prefers lower warmup cost and currently tries `CUDA -> DirectML -> TensorRT -> CPU` on Windows with `--features nvidia`;
- `OMNI_FORCE_PROVIDER` remains available as a diagnostics-only override when you need to pin one execution provider for benchmarking or debugging;
- on Apple Silicon/macOS, `gpu` is wired to the CoreML execution provider;
- on Linux, the current crate build does not yet wire a GPU provider; AMD GPU support would require a dedicated ROCm or WebGPU path;
- `auto` first tries the configured GPU chain and falls back to CPU if acceleration is unavailable;
- `gpu` requires at least one GPU execution provider to register successfully; it does not silently fall back to CPU;
- the Windows build output includes `DirectML.dll`; application packaging should ship that file with the executable.

Legacy migration:

```powershell
python .\scripts\flatten_bundle_to_flat.py --input .\models\chinese_clip_bundle --mode hardlink
python .\scripts\flatten_bundle_to_flat.py --input .\models\fgclip2_bundle --mode hardlink
```

Direct exporters:

The exporter scripts live in `D:\code\vl-embedding-test` and default to writing flat model
directories into `D:\code\omni_search\models`.

```powershell
uv run D:\code\vl-embedding-test\export_openclip_flat.py --id timm/MobileCLIP2-S2-OpenCLIP --output D:\code\omni_search\models\mobileclip2 --force
uv run D:\code\vl-embedding-test\export_chinese_clip_flat.py --model-dir D:\models\chinese-clip-vit-base-patch16 --output D:\code\omni_search\models\chinese_clip_flat --force
uv run D:\code\vl-embedding-test\export_fgclip2_flat.py --model-dir D:\models\fg-clip2-base --output D:\code\omni_search\models\fgclip2_flat --force
```

Builder example:

```rust
use omni_search::{
    GraphOptimizationLevel, OmniSearch, ProviderPolicy, RuntimeDevice, SessionPolicy,
};

let sdk = OmniSearch::builder()
    .from_local_model_dir("D:/models/fgclip2_flat")
    .device(RuntimeDevice::Auto)
    .provider_policy(ProviderPolicy::Service)
    .intra_threads(4)
    .fgclip_max_patches(256)
    .session_policy(SessionPolicy::SingleActive)
    .graph_optimization_level(GraphOptimizationLevel::All)
    .build()?;

let snapshot = sdk.runtime_snapshot();
println!("mode: {:?}", snapshot.summary.mode);
println!("provider: {:?}", snapshot.summary.effective_provider);
```
