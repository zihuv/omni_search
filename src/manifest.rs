use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::config::ModelFamily;
use crate::error::Error;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct ModelManifest {
    pub schema_version: u32,
    pub family: ModelFamily,
    pub model_id: String,
    pub model_revision: String,
    pub embedding_dim: usize,
    pub normalize_output: bool,
    pub text: TextConfig,
    pub image: ImageConfig,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct TextConfig {
    pub onnx: PathBuf,
    pub output_name: String,
    pub tokenizer: PathBuf,
    pub context_length: usize,
    pub input: TextInputConfig,
    #[serde(default)]
    pub token_embedding: Option<TokenEmbeddingConfig>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct TokenEmbeddingConfig {
    pub file: PathBuf,
    pub dtype: TensorDtype,
    pub embedding_dim: usize,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TensorDtype {
    F16,
    F32,
}

impl TensorDtype {
    pub fn bytes_per_value(self) -> usize {
        match self {
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum TextInputConfig {
    TokenEmbeds,
    BertLike {
        input_ids_name: String,
        attention_mask_name: String,
        #[serde(default)]
        token_type_ids_name: Option<String>,
    },
    InputIds {
        input_ids_name: String,
        #[serde(default)]
        lower_case: bool,
        #[serde(default)]
        pad_id: Option<u32>,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct ImageConfig {
    pub onnx: PathBuf,
    pub output_name: String,
    pub preprocess: ImagePreprocessConfig,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum ImagePreprocessConfig {
    FgclipPatchTokens {
        patch_size: usize,
        default_max_patches: usize,
        vision_pos_embedding: PathBuf,
    },
    ClipImage {
        image_size: usize,
        resize_shortest_edge: usize,
        crop: CropMode,
        mean: Vec<f32>,
        std: Vec<f32>,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CropMode {
    Center,
    None,
}

impl ModelManifest {
    pub fn validate(&self) -> Result<(), Error> {
        if self.schema_version != 1 {
            return Err(Error::invalid_bundle(format!(
                "unsupported schema_version {}",
                self.schema_version
            )));
        }
        if self.model_id.trim().is_empty() {
            return Err(Error::invalid_bundle("model_id cannot be empty"));
        }
        if self.model_revision.trim().is_empty() {
            return Err(Error::invalid_bundle("model_revision cannot be empty"));
        }
        if self.embedding_dim == 0 {
            return Err(Error::invalid_bundle(
                "embedding_dim must be greater than 0",
            ));
        }
        if self.text.output_name.trim().is_empty() {
            return Err(Error::invalid_bundle("text.output_name cannot be empty"));
        }
        if self.image.output_name.trim().is_empty() {
            return Err(Error::invalid_bundle("image.output_name cannot be empty"));
        }
        if self.text.context_length == 0 {
            return Err(Error::invalid_bundle(
                "text.context_length must be greater than 0",
            ));
        }

        match (&self.family, &self.text.input, &self.image.preprocess) {
            (
                ModelFamily::FgClip,
                TextInputConfig::TokenEmbeds,
                ImagePreprocessConfig::FgclipPatchTokens {
                    patch_size,
                    default_max_patches,
                    ..
                },
            ) => {
                if *patch_size == 0 {
                    return Err(Error::invalid_bundle(
                        "image.preprocess.patch_size must be greater than 0",
                    ));
                }
                if *default_max_patches == 0 {
                    return Err(Error::invalid_bundle(
                        "image.preprocess.default_max_patches must be greater than 0",
                    ));
                }
                let token_embedding = self.text.token_embedding.as_ref().ok_or_else(|| {
                    Error::invalid_bundle("fgclip manifest requires text.token_embedding")
                })?;
                if token_embedding.embedding_dim != self.embedding_dim {
                    return Err(Error::invalid_bundle(format!(
                        "text.token_embedding.embedding_dim {} does not match manifest embedding_dim {}",
                        token_embedding.embedding_dim, self.embedding_dim
                    )));
                }
            }
            (
                ModelFamily::ChineseClip,
                TextInputConfig::BertLike {
                    input_ids_name,
                    attention_mask_name,
                    ..
                },
                ImagePreprocessConfig::ClipImage {
                    image_size,
                    resize_shortest_edge,
                    crop: _,
                    mean,
                    std,
                },
            ) => {
                if input_ids_name.trim().is_empty() || attention_mask_name.trim().is_empty() {
                    return Err(Error::invalid_bundle(
                        "bert_like input names cannot be empty",
                    ));
                }
                if *image_size == 0 || *resize_shortest_edge == 0 {
                    return Err(Error::invalid_bundle(
                        "clip_image sizes must be greater than 0",
                    ));
                }
                if mean.len() != 3 || std.len() != 3 {
                    return Err(Error::invalid_bundle(
                        "clip_image mean/std must contain exactly 3 values",
                    ));
                }
                if std.iter().any(|value| value.abs() <= f32::EPSILON) {
                    return Err(Error::invalid_bundle(
                        "clip_image std values must be non-zero",
                    ));
                }
                if self.text.token_embedding.is_some() {
                    return Err(Error::invalid_bundle(
                        "chinese clip manifest must not define text.token_embedding",
                    ));
                }
            }
            (
                ModelFamily::OpenClip,
                TextInputConfig::InputIds { input_ids_name, .. },
                ImagePreprocessConfig::ClipImage {
                    image_size,
                    resize_shortest_edge,
                    crop: _,
                    mean,
                    std,
                },
            ) => {
                if input_ids_name.trim().is_empty() {
                    return Err(Error::invalid_bundle(
                        "input_ids input name cannot be empty",
                    ));
                }
                if *image_size == 0 || *resize_shortest_edge == 0 {
                    return Err(Error::invalid_bundle(
                        "clip_image sizes must be greater than 0",
                    ));
                }
                if mean.len() != 3 || std.len() != 3 {
                    return Err(Error::invalid_bundle(
                        "clip_image mean/std must contain exactly 3 values",
                    ));
                }
                if std.iter().any(|value| value.abs() <= f32::EPSILON) {
                    return Err(Error::invalid_bundle(
                        "clip_image std values must be non-zero",
                    ));
                }
                if self.text.token_embedding.is_some() {
                    return Err(Error::invalid_bundle(
                        "open clip manifest must not define text.token_embedding",
                    ));
                }
            }
            (ModelFamily::FgClip, _, _) => {
                return Err(Error::invalid_bundle(
                    "fgclip manifests require token_embeds text input and fgclip_patch_tokens image preprocess",
                ));
            }
            (ModelFamily::ChineseClip, _, _) => {
                return Err(Error::invalid_bundle(
                    "chinese_clip manifests require bert_like text input and clip_image image preprocess",
                ));
            }
            (ModelFamily::OpenClip, _, _) => {
                return Err(Error::invalid_bundle(
                    "open_clip manifests require input_ids text input and clip_image image preprocess",
                ));
            }
        }

        Ok(())
    }
}
