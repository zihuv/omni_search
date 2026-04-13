use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use image::{DynamicImage, GenericImageView};
use memmap2::Mmap;
use ndarray::{Array, ArrayD, IxDyn};
use ort::value::TensorRef;
use tokenizers::{Encoding, Tokenizer};

use crate::backend::{
    EmbeddingBackend, LazySession, embeddings_from_output, load_tokenizer, single_embedding,
};
use crate::bundle::{ModelBundle, ModelInfo};
use crate::config::{RuntimeConfig, SessionPolicy};
use crate::embedding::Embedding;
use crate::error::Error;
use crate::manifest::{ImagePreprocessConfig, TensorDtype};
use crate::preprocess::fgclip::{
    build_positional_embedding, determine_max_patches, preprocess_image, read_f32_file,
    stack_attention_masks, stack_f32_batches, stack_pixel_values,
};
use crate::runtime::RuntimeState;

pub(crate) struct FgClipBackend {
    info: ModelInfo,
    normalize_output: bool,
    session_policy: SessionPolicy,
    tokenizer: Tokenizer,
    context_length: usize,
    text_output_name: String,
    token_embedding_table: Mmap,
    token_embedding_rows: usize,
    token_embedding_dtype: TensorDtype,
    token_embedding_dim: usize,
    image_output_name: String,
    patch_size: usize,
    default_max_patches: usize,
    base_pos_embedding: Vec<f32>,
    base_grid_height: usize,
    base_grid_width: usize,
    text_session: LazySession,
    image_session: LazySession,
}

impl FgClipBackend {
    pub(crate) fn new(bundle: ModelBundle, runtime: RuntimeConfig) -> Result<Self, Error> {
        let fgclip_max_patches = runtime.fgclip_max_patches;
        let tokenizer = load_tokenizer(
            bundle.tokenizer_path(),
            bundle.manifest().text.context_length,
            "<pad>",
        )?;
        let token_embedding = bundle
            .manifest()
            .text
            .token_embedding
            .as_ref()
            .ok_or_else(|| {
                Error::invalid_bundle("fgclip bundle is missing text.token_embedding")
            })?;
        let token_embedding_path = bundle
            .token_embedding_path()
            .ok_or_else(|| Error::invalid_bundle("fgclip bundle is missing token embedding asset"))?
            .to_path_buf();
        let token_embedding_row_bytes =
            token_embedding.dtype.bytes_per_value() * token_embedding.embedding_dim;
        let token_embedding_file = fs::File::open(&token_embedding_path)?;
        let token_embedding_len = token_embedding_file.metadata()?.len() as usize;
        if token_embedding_len % token_embedding_row_bytes != 0 {
            return Err(Error::invalid_bundle(format!(
                "token embedding file {} has invalid byte length {}",
                token_embedding_path.display(),
                token_embedding_len
            )));
        }
        let token_embedding_table = unsafe { Mmap::map(&token_embedding_file)? };
        let token_embedding_rows = token_embedding_len / token_embedding_row_bytes;

        let (patch_size, manifest_default_max_patches, vision_pos_embedding_path) =
            match &bundle.manifest().image.preprocess {
                ImagePreprocessConfig::FgclipPatchTokens {
                    patch_size,
                    default_max_patches,
                    ..
                } => (
                    *patch_size,
                    *default_max_patches,
                    bundle
                        .vision_pos_embedding_path()
                        .ok_or_else(|| {
                            Error::invalid_bundle(
                                "fgclip bundle is missing image.preprocess.vision_pos_embedding",
                            )
                        })?
                        .to_path_buf(),
                ),
                _ => {
                    return Err(Error::invalid_bundle(
                        "fgclip bundle must use fgclip_patch_tokens preprocess",
                    ));
                }
            };
        let default_max_patches =
            resolve_default_max_patches(manifest_default_max_patches, fgclip_max_patches)?;

        let base_pos_embedding = read_f32_file(&vision_pos_embedding_path)?;
        let token_count = base_pos_embedding.len() / bundle.info().embedding_dim;
        let side = (token_count as f64).sqrt() as usize;
        if side == 0 || side * side != token_count {
            return Err(Error::invalid_bundle(format!(
                "fgclip vision_pos_embedding length {} is not a square grid for embedding dim {}",
                base_pos_embedding.len(),
                bundle.info().embedding_dim
            )));
        }

        Ok(Self {
            info: bundle.info().clone(),
            normalize_output: bundle.info().normalize_output,
            session_policy: runtime.session_policy,
            tokenizer,
            context_length: bundle.manifest().text.context_length,
            text_output_name: bundle.manifest().text.output_name.clone(),
            token_embedding_table,
            token_embedding_rows,
            token_embedding_dtype: token_embedding.dtype,
            token_embedding_dim: token_embedding.embedding_dim,
            image_output_name: bundle.manifest().image.output_name.clone(),
            patch_size,
            default_max_patches,
            base_pos_embedding,
            base_grid_height: side,
            base_grid_width: side,
            text_session: LazySession::new(bundle.text_onnx_path().to_path_buf(), runtime.clone()),
            image_session: LazySession::new(bundle.image_onnx_path().to_path_buf(), runtime),
        })
    }

    fn maybe_unload_image(&self) {
        if self.session_policy == SessionPolicy::SingleActive {
            self.image_session.unload();
        }
    }

    fn maybe_unload_text(&self) {
        if self.session_policy == SessionPolicy::SingleActive {
            self.text_session.unload();
        }
    }

    fn encode_texts_internal(&self, texts: &[String]) -> Result<Vec<Embedding>, Error> {
        self.text_session.with_session(|session| {
            let mut all_embeddings = Vec::with_capacity(texts.len());
            for text in texts {
                let encoding = self
                    .tokenizer
                    .encode(text.to_lowercase(), true)
                    .map_err(Error::from_tokenizer)?;
                let input_ids =
                    encodings_to_i64_array(&[encoding], self.context_length, |encoding| {
                        encoding
                            .get_ids()
                            .iter()
                            .map(|id| i64::from(*id))
                            .collect::<Vec<_>>()
                    })?;
                let token_embeds = self.gather_text_token_embeddings(&input_ids)?;
                let token_embeds =
                    TensorRef::from_array_view(token_embeds.view()).map_err(Error::from_ort)?;
                let outputs = session
                    .run(ort::inputs!["token_embeds" => token_embeds])
                    .map_err(Error::from_ort)?;
                let output = outputs.get(self.text_output_name.as_str()).ok_or_else(|| {
                    Error::ort(format!(
                        "text output '{}' not found in fgclip session",
                        self.text_output_name
                    ))
                })?;
                let output = output
                    .try_extract_array::<f32>()
                    .map_err(Error::from_ort)?
                    .to_owned()
                    .into_dyn();
                let mut embeddings =
                    embeddings_from_output(&self.info, output, self.normalize_output)?;
                all_embeddings.append(&mut embeddings);
            }
            Ok(all_embeddings)
        })
    }

    fn encode_images_internal(&self, images: &[DynamicImage]) -> Result<Vec<Embedding>, Error> {
        let mut image_groups = BTreeMap::<usize, Vec<usize>>::new();
        for (index, image) in images.iter().enumerate() {
            let (width, height) = image.dimensions();
            let max_patches =
                determine_max_patches(width, height, self.patch_size, self.default_max_patches);
            image_groups.entry(max_patches).or_default().push(index);
        }
        if image_groups.is_empty() {
            return Err(Error::image_preprocess("cannot embed an empty image batch"));
        }

        let mut all_embeddings = vec![None; images.len()];
        self.image_session.with_session(|session| {
            for (max_patches, indices) in &image_groups {
                let mut image_inputs = Vec::with_capacity(indices.len());
                let mut pos_embeds = Vec::with_capacity(indices.len());
                for &index in indices {
                    let image = &images[index];
                    let encoded = preprocess_image(image, self.patch_size, *max_patches)?;
                    let pos_embed = build_positional_embedding(
                        &self.base_pos_embedding,
                        self.base_grid_height,
                        self.base_grid_width,
                        encoded.spatial_height,
                        encoded.spatial_width,
                        *max_patches,
                        self.info.embedding_dim,
                    )?;
                    image_inputs.push(encoded);
                    pos_embeds.push(pos_embed);
                }

                let pixel_values = stack_pixel_values(&image_inputs)?;
                let pixel_attention_mask = stack_attention_masks(&image_inputs)?;
                let pos_embed = stack_f32_batches(
                    &pos_embeds,
                    [indices.len(), *max_patches, self.info.embedding_dim],
                )?;

                let pixel_values =
                    TensorRef::from_array_view(pixel_values.view()).map_err(Error::from_ort)?;
                let pixel_attention_mask = TensorRef::from_array_view(pixel_attention_mask.view())
                    .map_err(Error::from_ort)?;
                let pos_embed =
                    TensorRef::from_array_view(pos_embed.view()).map_err(Error::from_ort)?;
                let outputs = session
                    .run(ort::inputs![
                        "pixel_values" => pixel_values,
                        "pixel_attention_mask" => pixel_attention_mask,
                        "pos_embed" => pos_embed,
                    ])
                    .map_err(Error::from_ort)?;
                let output = outputs
                    .get(self.image_output_name.as_str())
                    .ok_or_else(|| {
                        Error::ort(format!(
                            "image output '{}' not found in fgclip session",
                            self.image_output_name
                        ))
                    })?;
                let output = output
                    .try_extract_array::<f32>()
                    .map_err(Error::from_ort)?
                    .to_owned()
                    .into_dyn();
                let batch_embeddings =
                    embeddings_from_output(&self.info, output, self.normalize_output)?;
                for (index, embedding) in indices.iter().copied().zip(batch_embeddings) {
                    all_embeddings[index] = Some(embedding);
                }
            }
            Ok(())
        })?;

        all_embeddings
            .into_iter()
            .enumerate()
            .map(|(index, embedding)| {
                embedding.ok_or_else(|| {
                    Error::image_preprocess(format!(
                        "fgclip image batch did not produce an embedding for index {index}"
                    ))
                })
            })
            .collect()
    }

    fn gather_text_token_embeddings(&self, input_ids: &ArrayD<i64>) -> Result<ArrayD<f32>, Error> {
        let shape = input_ids.shape();
        if shape.len() != 2 {
            return Err(Error::tokenizer(format!(
                "fgclip input_ids must have shape [B,S], got {:?}",
                shape
            )));
        }

        let input_ids = input_ids
            .as_slice()
            .ok_or_else(|| Error::tokenizer("input_ids are not contiguous"))?;
        let values = gather_token_embedding_rows(
            &self.token_embedding_table,
            self.token_embedding_rows,
            self.token_embedding_dtype,
            self.token_embedding_dim,
            input_ids,
        )?;

        Array::from_shape_vec(
            IxDyn(&[shape[0], shape[1], self.token_embedding_dim]),
            values,
        )
        .map_err(|error| Error::tokenizer(error.to_string()))
    }
}

fn resolve_default_max_patches(
    manifest_default_max_patches: usize,
    runtime_override: Option<usize>,
) -> Result<usize, Error> {
    const SUPPORTED_BUCKETS: [usize; 5] = [128, 256, 576, 784, 1024];

    let Some(runtime_override) = runtime_override else {
        return Ok(manifest_default_max_patches);
    };
    if !SUPPORTED_BUCKETS.contains(&runtime_override) {
        return Err(Error::invalid_config(format!(
            "fgclip_max_patches must be one of {:?}, got {runtime_override}",
            SUPPORTED_BUCKETS
        )));
    }
    if runtime_override > manifest_default_max_patches {
        return Err(Error::invalid_config(format!(
            "fgclip_max_patches {runtime_override} exceeds bundle default_max_patches {manifest_default_max_patches}"
        )));
    }
    Ok(runtime_override)
}

impl EmbeddingBackend for FgClipBackend {
    fn embed_text(&self, text: &str) -> Result<Embedding, Error> {
        let batch = vec![text.to_owned()];
        self.maybe_unload_image();
        single_embedding(self.encode_texts_internal(&batch)?, "fgclip text")
    }

    fn embed_texts(&self, texts: &[String]) -> Result<Vec<Embedding>, Error> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        self.maybe_unload_image();
        self.encode_texts_internal(texts)
    }

    fn embed_image_path(&self, path: &Path) -> Result<Embedding, Error> {
        self.maybe_unload_text();
        let image = image::open(path)
            .map_err(|error| Error::image_preprocess(format!("{}: {error}", path.display())))?;
        single_embedding(self.encode_images_internal(&[image])?, "fgclip image")
    }

    fn embed_image_bytes(&self, bytes: &[u8]) -> Result<Embedding, Error> {
        self.maybe_unload_text();
        let image = image::load_from_memory(bytes)
            .map_err(|error| Error::image_preprocess(error.to_string()))?;
        single_embedding(self.encode_images_internal(&[image])?, "fgclip image")
    }

    fn embed_image_paths(&self, paths: &[PathBuf]) -> Result<Vec<Embedding>, Error> {
        if paths.is_empty() {
            return Ok(Vec::new());
        }
        self.maybe_unload_text();
        let images = paths
            .iter()
            .map(|path| {
                image::open(path).map_err(|error| {
                    Error::image_preprocess(format!("{}: {error}", path.display()))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.encode_images_internal(&images)
    }

    fn preload_text(&self) -> Result<(), Error> {
        self.maybe_unload_image();
        self.text_session.ensure_loaded()
    }

    fn preload_image(&self) -> Result<(), Error> {
        self.maybe_unload_text();
        self.image_session.ensure_loaded()
    }

    fn unload_text(&self) -> bool {
        self.text_session.unload()
    }

    fn unload_image(&self) -> bool {
        self.image_session.unload()
    }

    fn runtime_state(&self) -> RuntimeState {
        RuntimeState {
            text_loaded: self.text_session.is_loaded(),
            image_loaded: self.image_session.is_loaded(),
            last_text_used_at: self.text_session.last_used_at(),
            last_image_used_at: self.image_session.last_used_at(),
        }
    }
}

fn encodings_to_i64_array(
    encodings: &[Encoding],
    context_length: usize,
    extractor: impl Fn(&Encoding) -> Vec<i64>,
) -> Result<ArrayD<i64>, Error> {
    let mut values = Vec::with_capacity(encodings.len() * context_length);
    for encoding in encodings {
        let row = extractor(encoding);
        if row.len() != context_length {
            return Err(Error::tokenizer(format!(
                "tokenized length {} does not match configured context length {}",
                row.len(),
                context_length
            )));
        }
        values.extend(row);
    }
    Array::from_shape_vec(IxDyn(&[encodings.len(), context_length]), values)
        .map_err(|error| Error::tokenizer(error.to_string()))
}

fn gather_token_embedding_rows(
    bytes: &[u8],
    token_count: usize,
    dtype: TensorDtype,
    token_embedding_dim: usize,
    input_ids: &[i64],
) -> Result<Vec<f32>, Error> {
    let row_bytes = dtype.bytes_per_value() * token_embedding_dim;
    let mut values = vec![0.0f32; input_ids.len() * token_embedding_dim];

    for (token_index, token_id) in input_ids.iter().enumerate() {
        if *token_id < 0 || *token_id as usize >= token_count {
            return Err(Error::tokenizer(format!(
                "token id {token_id} is outside embedding table with {token_count} rows"
            )));
        }
        let row_start = *token_id as usize * row_bytes;
        let row = &bytes[row_start..row_start + row_bytes];
        let output =
            &mut values[token_index * token_embedding_dim..(token_index + 1) * token_embedding_dim];
        match dtype {
            TensorDtype::F16 => {
                for (value, bytes) in output.iter_mut().zip(row.chunks_exact(2)) {
                    *value = f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
                }
            }
            TensorDtype::F32 => {
                for (value, bytes) in output.iter_mut().zip(row.chunks_exact(4)) {
                    *value = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                }
            }
        }
    }

    Ok(values)
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = bits & 0x03ff;

    let f32_bits = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let mut fraction = fraction as u32;
            let mut exponent = -14i32;
            while fraction & 0x0400 == 0 {
                fraction <<= 1;
                exponent -= 1;
            }
            fraction &= 0x03ff;
            sign | (((exponent + 127) as u32) << 23) | (fraction << 13)
        }
        0x1f => sign | 0x7f80_0000 | ((fraction as u32) << 13),
        _ => sign | (((exponent as u32) + 112) << 23) | ((fraction as u32) << 13),
    };
    f32::from_bits(f32_bits)
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, GenericImageView, Rgb, RgbImage};

    use super::{
        TensorDtype, determine_max_patches, gather_token_embedding_rows,
        resolve_default_max_patches,
    };

    #[test]
    fn assigns_patch_buckets_per_image() {
        let images = [
            DynamicImage::ImageRgb8(RgbImage::from_pixel(853, 1280, Rgb([0, 0, 0]))),
            DynamicImage::ImageRgb8(RgbImage::from_pixel(240, 160, Rgb([0, 0, 0]))),
            DynamicImage::ImageRgb8(RgbImage::from_pixel(5152, 7728, Rgb([0, 0, 0]))),
        ];
        let buckets = images
            .iter()
            .map(|image| {
                let (width, height) = image.dimensions();
                determine_max_patches(width, height, 16, 1024)
            })
            .collect::<Vec<_>>();
        assert_eq!(buckets, vec![1024, 256, 1024]);
    }

    #[test]
    fn gathers_f16_token_embeddings_from_mapped_bytes() {
        let bytes = [
            0x00, 0x00, 0x00, 0x3c, 0x00, 0x40, 0x00, 0x42, 0x00, 0x44, 0x00, 0x45,
        ];
        let values = gather_token_embedding_rows(&bytes, 3, TensorDtype::F16, 2, &[2, 0]).unwrap();
        assert_eq!(values, vec![4.0, 5.0, 0.0, 1.0]);
    }

    #[test]
    fn accepts_supported_fgclip_patch_overrides() {
        assert_eq!(resolve_default_max_patches(1024, None).unwrap(), 1024);
        assert_eq!(resolve_default_max_patches(1024, Some(576)).unwrap(), 576);
    }

    #[test]
    fn rejects_invalid_fgclip_patch_overrides() {
        assert!(resolve_default_max_patches(1024, Some(300)).is_err());
        assert!(resolve_default_max_patches(576, Some(784)).is_err());
    }
}
