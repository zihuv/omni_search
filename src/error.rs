use std::fmt::Display;

use thiserror::Error;

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid config: {message}")]
    InvalidConfig { message: String },

    #[error("invalid bundle: {message}")]
    InvalidBundle { message: String },

    #[error("unsupported model family: {family}")]
    UnsupportedModelFamily { family: String },

    #[error("tokenizer error: {message}")]
    Tokenizer { message: String },

    #[error("image preprocess error: {message}")]
    ImagePreprocess { message: String },

    #[error("onnx runtime error: {message}")]
    Ort { message: String },

    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("embedding model mismatch: left={left_model_id}, right={right_model_id}")]
    EmbeddingModelMismatch {
        left_model_id: String,
        right_model_id: String,
    },

    #[error("feature not implemented yet: {feature}")]
    FeatureNotImplemented { feature: String },
}

impl Error {
    pub(crate) fn invalid_config(message: impl Into<String>) -> Self {
        Self::InvalidConfig {
            message: message.into(),
        }
    }

    pub(crate) fn invalid_bundle(message: impl Into<String>) -> Self {
        Self::InvalidBundle {
            message: message.into(),
        }
    }

    pub(crate) fn tokenizer(message: impl Into<String>) -> Self {
        Self::Tokenizer {
            message: message.into(),
        }
    }

    pub(crate) fn image_preprocess(message: impl Into<String>) -> Self {
        Self::ImagePreprocess {
            message: message.into(),
        }
    }

    pub(crate) fn ort(message: impl Into<String>) -> Self {
        Self::Ort {
            message: message.into(),
        }
    }

    pub(crate) fn from_tokenizer(error: impl Display) -> Self {
        Self::tokenizer(error.to_string())
    }

    pub(crate) fn from_ort(error: impl Display) -> Self {
        Self::ort(error.to_string())
    }
}
