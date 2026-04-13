use std::sync::Arc;

use crate::bundle::ModelInfo;
use crate::config::ModelFamily;
use crate::error::Error;

#[derive(Clone, Debug)]
pub struct Embedding {
    pub model_family: ModelFamily,
    pub model_id: String,
    pub dims: usize,
    values: Arc<[f32]>,
}

impl Embedding {
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    pub(crate) fn from_vec(info: &ModelInfo, values: Vec<f32>) -> Result<Self, Error> {
        if values.len() != info.embedding_dim {
            return Err(Error::DimensionMismatch {
                expected: info.embedding_dim,
                actual: values.len(),
            });
        }
        Ok(Self {
            model_family: info.model_family.clone(),
            model_id: info.model_id.clone(),
            dims: values.len(),
            values: Arc::<[f32]>::from(values),
        })
    }
}

impl AsRef<[f32]> for Embedding {
    fn as_ref(&self) -> &[f32] {
        self.as_slice()
    }
}
