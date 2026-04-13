# omni_search

`omni_search` is a Rust SDK for multimodal embedding and similarity search.

Current scope:

- load a local model bundle with `manifest.json`;
- compute text embeddings;
- compute image embeddings;
- compare embeddings with cosine similarity;
- manually unload ONNX Runtime sessions.

The SDK currently targets local ONNX bundles through the `ort` crate.

Quickstart:

- keep a local FG-CLIP2 bundle under `models/fgclip2_bundle/`;
- keep one or more sample images under `samples/`;
- run `cargo run --release` to scan all images in `samples/` with the default query `"山"`;
- `main.rs` prints both `text_to_image` and `image_to_image` top-k examples using the same public SDK API;
- run `cargo run --release -- "海边"` to scan all images in `samples/` with a custom query;
- run `cargo run --release -- "海边" 20` to scan all images in `samples/` and print top 20;
- run `cargo run --release -- "海边" "samples/pic1.jpg"` to keep the default top-k and use a specific query image for `image_to_image`;
- run `cargo run --release -- "海边" 20 "samples/pic1.jpg"` to set both top-k and query image;
- set `OMNI_INTRA_THREADS` / `OMNI_INTER_THREADS` to override ONNX Runtime thread counts when benchmarking or tuning;
- set `OMNI_FGCLIP_MAX_PATCHES` to cap FG-CLIP2 image preprocessing at a smaller bucket without changing the ONNX bundle;
- recommended `OMNI_FGCLIP_MAX_PATCHES` values are `128`, `256`, `576`, `784`, or `1024`; Chinese CLIP ignores this override because it uses fixed `224x224` inputs;
- see `docs/model-performance.md` for the current Chinese CLIP vs FGCLIP2 performance and memory comparison;
- run `cargo test --test quickstart -- --nocapture` for the matching smoke test.
