# Model Performance Notes

This document summarizes the current CPU-side performance and memory measurements for the local `Chinese CLIP` and `FGCLIP2` bundles in this workspace.

## Method

- Platform: Windows desktop, CPU execution provider, `cargo run --release`
- Runtime: `OMNI_INTRA_THREADS=16`, `OMNI_INTER_THREADS` unset
- Warm latency: average over 30 repeated calls after preloading the relevant session
- Batch latency: 3 sample images under `samples/`, reported both as total batch time and per-image average
- Memory: separate end-to-end process runs, reported as peak `Working Set` and peak `Private Bytes`
- Text probes: `山`, `海边`, `灯笼`

## Chinese CLIP vs FGCLIP2

| Model | Image input form | Warm text avg | Warm image avg | Warm batch avg | Warm batch avg / image | Single-image peak memory | 3-image peak memory | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| Chinese CLIP | Fixed `224x224` tensor | `23.98 ms` | `110.98 ms` | `1795.54 ms` | `598.51 ms` | `433.5 MB WS / 421.3 MB Private` | `616.7 MB WS / 604.9 MB Private` | No `max_patches`; image path is fixed-size ViT-B/16 style preprocessing |
| FGCLIP2 (`max_patches=1024`) | Dynamic patch tokens | `31.43 ms` | `518.08 ms` | `1984.73 ms` | `661.58 ms` | `711.1 MB WS / 758.2 MB Private` | `961.6 MB WS / 1195.4 MB Private` | Higher image cost, but runtime can trade speed for quality by lowering patch budget |

## FGCLIP2 Patch Budget Tradeoff

`FGCLIP2` can reuse the same ONNX bundle while changing the preprocessing cap through `OMNI_FGCLIP_MAX_PATCHES`. Lower values reduce latency and memory, but they also remove image detail before the model sees it.

| `max_patches` | Warm image avg | Warm batch avg / image | Single-image peak memory | Cosine vs `1024` baseline | Recommendation |
| ---: | ---: | ---: | --- | --- | --- |
| `1024` | `518.08 ms` | `661.58 ms` | `711.1 MB WS / 758.2 MB Private` | baseline | Best quality, heaviest runtime |
| `576` | `272.92 ms` | `391.22 ms` | `552.4 MB WS / 587.1 MB Private` | avg `0.9645`, min `0.9403` | Best default tradeoff |
| `256` | `123.77 ms` | `224.24 ms` | `490.2 MB WS / 487.3 MB Private` | avg `0.9514`, min `0.9349` | Resource-constrained mode |
| `128` | `74.33 ms` | `172.28 ms` | `471.5 MB WS / 466.8 MB Private` | avg `0.8112`, min `0.7654` | Aggressive cap, visible quality risk |

## Practical Takeaways

- `Chinese CLIP` is the lighter default if you care about lower CPU cost and lower memory footprint.
- `FGCLIP2` is substantially heavier on image inference because it uses a dynamic patch-token image tower instead of a fixed `224x224` image tensor.
- `FGCLIP2` does not need a separate export per patch budget. The current runtime override only changes preprocessing, so the same ONNX bundle can be reused.
- If you want a single recommended `FGCLIP2` setting, use `OMNI_FGCLIP_MAX_PATCHES=576`.
- `Chinese CLIP` ignores `OMNI_FGCLIP_MAX_PATCHES` because its image path is fixed-size and has no dynamic patch budget.

## Related Runtime Knobs

- `OMNI_INTRA_THREADS`: override ORT intra-op thread count
- `OMNI_INTER_THREADS`: override ORT inter-op thread count
- `OMNI_FGCLIP_MAX_PATCHES`: cap `FGCLIP2` image preprocessing to `128`, `256`, `576`, `784`, or `1024`
