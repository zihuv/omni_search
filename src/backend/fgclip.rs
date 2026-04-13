use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use image::{DynamicImage, GenericImageView};
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
    token_embedding_path: PathBuf,
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

        let (patch_size, default_max_patches, vision_pos_embedding_path) =
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
            token_embedding_path,
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
        let max_patches = images
            .iter()
            .map(|image| {
                let (width, height) = image.dimensions();
                determine_max_patches(width, height, self.patch_size, self.default_max_patches)
            })
            .max()
            .ok_or_else(|| Error::image_preprocess("cannot embed an empty image batch"))?;

        let mut image_inputs = Vec::with_capacity(images.len());
        let mut pos_embeds = Vec::with_capacity(images.len());
        for image in images {
            let encoded = preprocess_image(image, self.patch_size, max_patches)?;
            let pos_embed = build_positional_embedding(
                &self.base_pos_embedding,
                self.base_grid_height,
                self.base_grid_width,
                encoded.spatial_height,
                encoded.spatial_width,
                max_patches,
                self.info.embedding_dim,
            )?;
            image_inputs.push(encoded);
            pos_embeds.push(pos_embed);
        }

        let pixel_values = stack_pixel_values(&image_inputs)?;
        let pixel_attention_mask = stack_attention_masks(&image_inputs)?;
        let pos_embed = stack_f32_batches(
            &pos_embeds,
            [images.len(), max_patches, self.info.embedding_dim],
        )?;

        self.image_session.with_session(|session| {
            let pixel_values =
                TensorRef::from_array_view(pixel_values.view()).map_err(Error::from_ort)?;
            let pixel_attention_mask =
                TensorRef::from_array_view(pixel_attention_mask.view()).map_err(Error::from_ort)?;
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
            embeddings_from_output(&self.info, output, self.normalize_output)
        })
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
        let row_bytes = self.token_embedding_dtype.bytes_per_value() * self.token_embedding_dim;
        let metadata = fs::metadata(&self.token_embedding_path)?;
        if metadata.len() % row_bytes as u64 != 0 {
            return Err(Error::invalid_bundle(format!(
                "token embedding file {} has invalid byte length {}",
                self.token_embedding_path.display(),
                metadata.len()
            )));
        }
        let token_count = metadata.len() / row_bytes as u64;
        let mut file = fs::File::open(&self.token_embedding_path)?;
        let mut row_buffer = vec![0u8; row_bytes];
        let mut values = vec![0.0f32; input_ids.len() * self.token_embedding_dim];

        for (token_index, token_id) in input_ids.iter().enumerate() {
            if *token_id < 0 || *token_id as u64 >= token_count {
                return Err(Error::tokenizer(format!(
                    "token id {token_id} is outside embedding table with {token_count} rows"
                )));
            }
            file.seek(SeekFrom::Start(*token_id as u64 * row_bytes as u64))?;
            file.read_exact(&mut row_buffer)?;
            let output = &mut values[token_index * self.token_embedding_dim
                ..(token_index + 1) * self.token_embedding_dim];
            match self.token_embedding_dtype {
                TensorDtype::F16 => {
                    for (value, bytes) in output.iter_mut().zip(row_buffer.chunks_exact(2)) {
                        *value = f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
                    }
                }
                TensorDtype::F32 => {
                    for (value, bytes) in output.iter_mut().zip(row_buffer.chunks_exact(4)) {
                        *value = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                    }
                }
            }
        }

        Array::from_shape_vec(
            IxDyn(&[shape[0], shape[1], self.token_embedding_dim]),
            values,
        )
        .map_err(|error| Error::tokenizer(error.to_string()))
    }
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
