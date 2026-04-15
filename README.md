# omni_search

`omni_search` is a Rust SDK for multimodal embedding and similarity search over local ONNX model directories.

Current scope:

- load a flat local model directory with root-level `model_config.json`;
- compute text embeddings;
- compute image embeddings;
- compare embeddings with cosine similarity;
- manually unload ONNX Runtime sessions.

Supported families:

- `chinese_clip`
- `fg_clip`
- `open_clip`

The published crate does not bundle ONNX models or sample images. Point it at your own local assets with `OMNI_BUNDLE_DIR` and `OMNI_SAMPLES_DIR`.

Quickstart:

- set `OMNI_BUNDLE_DIR` to a local model directory that contains `model_config.json` plus flat root-level assets;
- set `OMNI_SAMPLES_DIR` to a directory containing one or more `.jpg`, `.jpeg`, `.png`, `.webp`, or `.bmp` images;
- build SDK instances with `OmniSearch::builder()` when you only want to override part of the runtime config and keep the rest at defaults;
- run `cargo run --bin omni_search --release` to scan all images in `OMNI_SAMPLES_DIR` with the default query `"山"`;
- run `cargo run --bin omni_search --release -- "海边"` to scan all images with a custom query;
- run `cargo run --bin omni_search --release -- "海边" 20` to print a different top-k;
- run `cargo run --bin omni_search --release -- "海边" "/absolute/path/to/query.jpg"` to run `image_to_image` with a specific query image;
- set `OMNI_INTRA_THREADS` / `OMNI_INTER_THREADS` to override ONNX Runtime thread counts when benchmarking or tuning;
- set `OMNI_FGCLIP_MAX_PATCHES` to cap FG-CLIP2 image preprocessing at a smaller bucket without changing the exported model directory;
- recommended `OMNI_FGCLIP_MAX_PATCHES` values are `128`, `256`, `576`, `784`, or `1024`;
- run `cargo test --test quickstart -- --ignored --nocapture` to execute the smoke test after setting `OMNI_TEST_BUNDLE_DIR` and `OMNI_TEST_SAMPLE_IMAGE`.

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
use omni_search::{GraphOptimizationLevel, OmniSearch, SessionPolicy};

let sdk = OmniSearch::builder()
    .from_local_model_dir("D:/models/fgclip2_flat")
    .intra_threads(4)
    .fgclip_max_patches(256)
    .session_policy(SessionPolicy::SingleActive)
    .graph_optimization_level(GraphOptimizationLevel::All)
    .build()?;
```
