# omni_search

`omni_search` is a Rust SDK for multimodal embedding and similarity search.

Current scope:

- load a local model bundle with `manifest.json`;
- compute text embeddings;
- compute image embeddings;
- compare embeddings with cosine similarity;
- manually unload ONNX Runtime sessions.

The SDK currently targets local ONNX bundles through the `ort` crate.

The published crate does not bundle ONNX models or sample images. Point it at your own local assets with `OMNI_BUNDLE_DIR` and `OMNI_SAMPLES_DIR`.

Quickstart:

- set `OMNI_BUNDLE_DIR` to a local bundle directory that contains `manifest.json`, tokenizer assets, and ONNX files;
- set `OMNI_SAMPLES_DIR` to a directory containing one or more `.jpg`, `.jpeg`, `.png`, `.webp`, or `.bmp` images;
- run `cargo run --bin omni_search --release` to scan all images in `OMNI_SAMPLES_DIR` with the default query `"山"`;
- `main.rs` prints both `text_to_image` and `image_to_image` top-k examples using the same public SDK API;
- run `cargo run --bin omni_search --release -- "海边"` to scan all images in `OMNI_SAMPLES_DIR` with a custom query;
- run `cargo run --bin omni_search --release -- "海边" 20` to scan all images in `OMNI_SAMPLES_DIR` and print top 20;
- run `cargo run --bin omni_search --release -- "海边" "/absolute/path/to/query.jpg"` to keep the default top-k and use a specific query image for `image_to_image`;
- run `cargo run --bin omni_search --release -- "海边" 20 "/absolute/path/to/query.jpg"` to set both top-k and query image;
- set `OMNI_INTRA_THREADS` / `OMNI_INTER_THREADS` to override ONNX Runtime thread counts when benchmarking or tuning;
- set `OMNI_FGCLIP_MAX_PATCHES` to cap FG-CLIP2 image preprocessing at a smaller bucket without changing the ONNX bundle;
- recommended `OMNI_FGCLIP_MAX_PATCHES` values are `128`, `256`, `576`, `784`, or `1024`; Chinese CLIP ignores this override because it uses fixed `224x224` inputs;
- see `docs/model-performance.md` for the current Chinese CLIP vs FGCLIP2 performance and memory comparison;
- run `cargo test --test quickstart -- --ignored --nocapture` to execute the smoke test after setting `OMNI_TEST_BUNDLE_DIR` and `OMNI_TEST_SAMPLE_IMAGE`.
