use std::path::{Path, PathBuf};

use image::DynamicImage;
use ndarray::{Array, ArrayD, IxDyn};
use ort::value::TensorRef;
use tokenizers::{Encoding, Tokenizer};

use crate::backend::{
    EmbeddingBackend, LazySession, embeddings_from_output, load_tokenizer_with_pad_id,
    runtime_snapshot_for_sessions, single_embedding,
};
use crate::bundle::{ModelBundle, ModelInfo};
use crate::config::{RuntimeConfig, SessionPolicy};
use crate::embedding::Embedding;
use crate::error::Error;
use crate::manifest::{ImagePreprocessConfig, TextInputConfig};
use crate::preprocess::clip_image::{
    ClipImagePreprocessConfig, preprocess_image, stack_image_batches,
};
use crate::runtime::{RuntimeSnapshot, RuntimeState};

pub(crate) struct OpenClipBackend {
    info: ModelInfo,
    runtime: RuntimeConfig,
    normalize_output: bool,
    session_policy: SessionPolicy,
    tokenizer: Tokenizer,
    context_length: usize,
    lower_case: bool,
    text_output_name: String,
    input_ids_name: String,
    image_output_name: String,
    image_preprocess: ClipImagePreprocessConfig,
    text_session: LazySession,
    image_session: LazySession,
}

impl OpenClipBackend {
    pub(crate) fn new(bundle: ModelBundle, runtime: RuntimeConfig) -> Result<Self, Error> {
        let (input_ids_name, lower_case, pad_id) = match &bundle.manifest().text.input {
            TextInputConfig::InputIds {
                input_ids_name,
                lower_case,
                pad_id,
            } => (input_ids_name.clone(), *lower_case, pad_id.unwrap_or(0)),
            _ => {
                return Err(Error::invalid_bundle(
                    "open clip bundle must use input_ids text input",
                ));
            }
        };
        let tokenizer = load_tokenizer_with_pad_id(
            bundle.tokenizer_path(),
            bundle.manifest().text.context_length,
            pad_id,
        )?;
        let image_preprocess = match &bundle.manifest().image.preprocess {
            ImagePreprocessConfig::ClipImage {
                image_size,
                resize_shortest_edge,
                crop,
                mean,
                std,
            } => ClipImagePreprocessConfig {
                image_size: *image_size,
                resize_shortest_edge: *resize_shortest_edge,
                crop: *crop,
                mean: [mean[0], mean[1], mean[2]],
                std: [std[0], std[1], std[2]],
            },
            _ => {
                return Err(Error::invalid_bundle(
                    "open clip bundle must use clip_image preprocess",
                ));
            }
        };

        Ok(Self {
            info: bundle.info().clone(),
            runtime: runtime.clone(),
            normalize_output: bundle.info().normalize_output,
            session_policy: runtime.session_policy,
            tokenizer,
            context_length: bundle.manifest().text.context_length,
            lower_case,
            text_output_name: bundle.manifest().text.output_name.clone(),
            input_ids_name,
            image_output_name: bundle.manifest().image.output_name.clone(),
            image_preprocess,
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
        let inputs = texts
            .iter()
            .map(|text| {
                if self.lower_case {
                    text.to_lowercase()
                } else {
                    text.clone()
                }
            })
            .collect::<Vec<_>>();
        let encodings = self
            .tokenizer
            .encode_batch(inputs, true)
            .map_err(Error::from_tokenizer)?;
        let input_ids = encodings_to_i64_array(&encodings, self.context_length, |encoding| {
            encoding
                .get_ids()
                .iter()
                .map(|id| i64::from(*id))
                .collect::<Vec<_>>()
        })?;

        self.text_session.with_session(|session| {
            let input_ids =
                TensorRef::from_array_view(input_ids.view()).map_err(Error::from_ort)?;
            let outputs = session
                .run(ort::inputs![self.input_ids_name.as_str() => input_ids])
                .map_err(Error::from_ort)?;
            let output = outputs.get(self.text_output_name.as_str()).ok_or_else(|| {
                Error::ort(format!(
                    "text output '{}' not found in open clip session",
                    self.text_output_name
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

    fn encode_images_internal(&self, images: &[DynamicImage]) -> Result<Vec<Embedding>, Error> {
        let tensors = images
            .iter()
            .map(|image| preprocess_image(image, &self.image_preprocess))
            .collect::<Result<Vec<_>, _>>()?;
        let input = stack_image_batches(&tensors, self.image_preprocess.image_size)?;

        self.image_session.with_session(|session| {
            let pixel_values = TensorRef::from_array_view(input.view()).map_err(Error::from_ort)?;
            let outputs = session
                .run(ort::inputs!["pixel_values" => pixel_values])
                .map_err(Error::from_ort)?;
            let output = outputs
                .get(self.image_output_name.as_str())
                .ok_or_else(|| {
                    Error::ort(format!(
                        "image output '{}' not found in open clip session",
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
}

impl EmbeddingBackend for OpenClipBackend {
    fn embed_text(&self, text: &str) -> Result<Embedding, Error> {
        let batch = vec![text.to_owned()];
        self.maybe_unload_image();
        single_embedding(self.encode_texts_internal(&batch)?, "open clip text")
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
        single_embedding(self.encode_images_internal(&[image])?, "open clip image")
    }

    fn embed_image_bytes(&self, bytes: &[u8]) -> Result<Embedding, Error> {
        self.maybe_unload_text();
        let image = image::load_from_memory(bytes)
            .map_err(|error| Error::image_preprocess(error.to_string()))?;
        single_embedding(self.encode_images_internal(&[image])?, "open clip image")
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

    fn runtime_snapshot(&self) -> RuntimeSnapshot {
        runtime_snapshot_for_sessions(&self.runtime, &self.text_session, &self.image_session)
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
