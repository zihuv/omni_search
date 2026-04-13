mod chinese_clip;
mod fgclip;

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use ndarray::ArrayD;
use ort::session::{Session, builder::GraphOptimizationLevel as OrtGraphOptimizationLevel};
use tokenizers::decoders::wordpiece::WordPiece as WordPieceDecoder;
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::bert::BertNormalizer;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::processors::bert::BertProcessing;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::bundle::{ModelBundle, ModelInfo};
use crate::config::{GraphOptimizationLevel, ModelFamily, RuntimeConfig};
use crate::embedding::Embedding;
use crate::error::Error;
use crate::runtime::RuntimeState;

pub(crate) use chinese_clip::ChineseClipBackend;
pub(crate) use fgclip::FgClipBackend;

pub(crate) trait EmbeddingBackend {
    fn embed_text(&self, text: &str) -> Result<Embedding, Error>;
    fn embed_texts(&self, texts: &[String]) -> Result<Vec<Embedding>, Error>;
    fn embed_image_path(&self, path: &Path) -> Result<Embedding, Error>;
    fn embed_image_bytes(&self, bytes: &[u8]) -> Result<Embedding, Error>;
    fn embed_image_paths(&self, paths: &[PathBuf]) -> Result<Vec<Embedding>, Error>;
    fn preload_text(&self) -> Result<(), Error>;
    fn preload_image(&self) -> Result<(), Error>;
    fn unload_text(&self) -> bool;
    fn unload_image(&self) -> bool;
    fn runtime_state(&self) -> RuntimeState;
}

pub(crate) fn create_backend(
    bundle: ModelBundle,
    runtime: RuntimeConfig,
) -> Result<Box<dyn EmbeddingBackend>, Error> {
    match bundle.info().model_family {
        ModelFamily::FgClip => Ok(Box::new(FgClipBackend::new(bundle, runtime)?)),
        ModelFamily::ChineseClip => Ok(Box::new(ChineseClipBackend::new(bundle, runtime)?)),
    }
}

pub(crate) struct LazySession {
    model_path: PathBuf,
    runtime: RuntimeConfig,
    state: Mutex<SessionState>,
}

struct SessionState {
    session: Option<Session>,
    last_used_at: Option<Instant>,
}

impl LazySession {
    pub(crate) fn new(model_path: PathBuf, runtime: RuntimeConfig) -> Self {
        Self {
            model_path,
            runtime,
            state: Mutex::new(SessionState {
                session: None,
                last_used_at: None,
            }),
        }
    }

    pub(crate) fn ensure_loaded(&self) -> Result<(), Error> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.session.is_none() {
            state.session = Some(load_session(&self.model_path, &self.runtime)?);
        }
        Ok(())
    }

    pub(crate) fn with_session<T>(
        &self,
        f: impl FnOnce(&mut Session) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.session.is_none() {
            state.session = Some(load_session(&self.model_path, &self.runtime)?);
        }
        let result = f(state.session.as_mut().expect("session must be loaded"))?;
        state.last_used_at = Some(Instant::now());
        Ok(result)
    }

    pub(crate) fn unload(&self) -> bool {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.session.take().is_some()
    }

    pub(crate) fn is_loaded(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .session
            .is_some()
    }

    pub(crate) fn last_used_at(&self) -> Option<Instant> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .last_used_at
    }
}

pub(crate) fn load_tokenizer(
    tokenizer_path: &Path,
    max_len: usize,
    fallback_pad_token: &str,
) -> Result<Tokenizer, Error> {
    let mut tokenizer = load_tokenizer_from_path(tokenizer_path)?;
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: max_len,
            ..Default::default()
        }))
        .map_err(Error::from_tokenizer)?;

    let pad_id = tokenizer.token_to_id(fallback_pad_token).unwrap_or(0);
    let mut padding = tokenizer
        .get_padding()
        .cloned()
        .unwrap_or_else(|| PaddingParams {
            pad_id,
            pad_type_id: 0,
            pad_token: fallback_pad_token.to_owned(),
            ..Default::default()
        });
    padding.strategy = PaddingStrategy::Fixed(max_len);
    tokenizer.with_padding(Some(padding));
    Ok(tokenizer)
}

fn load_tokenizer_from_path(tokenizer_path: &Path) -> Result<Tokenizer, Error> {
    if tokenizer_path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("txt"))
    {
        return build_bert_tokenizer_from_vocab(tokenizer_path);
    }

    Tokenizer::from_file(tokenizer_path)
        .map_err(Error::from_tokenizer)
        .map_err(|error| Error::tokenizer(format!("{}: {error}", tokenizer_path.display())))
}

fn build_bert_tokenizer_from_vocab(tokenizer_path: &Path) -> Result<Tokenizer, Error> {
    let tokenizer_path_str = tokenizer_path.to_str().ok_or_else(|| {
        Error::tokenizer(format!(
            "tokenizer path is not valid UTF-8: {}",
            tokenizer_path.display()
        ))
    })?;
    let wordpiece = WordPiece::from_file(tokenizer_path_str)
        .unk_token("[UNK]".to_owned())
        .build()
        .map_err(Error::from_tokenizer)
        .map_err(|error| Error::tokenizer(format!("{}: {error}", tokenizer_path.display())))?;
    let mut tokenizer = Tokenizer::new(wordpiece);
    let sep = tokenizer.token_to_id("[SEP]").ok_or_else(|| {
        Error::tokenizer(format!("{} is missing [SEP]", tokenizer_path.display()))
    })?;
    let cls = tokenizer.token_to_id("[CLS]").ok_or_else(|| {
        Error::tokenizer(format!("{} is missing [CLS]", tokenizer_path.display()))
    })?;
    tokenizer
        .with_normalizer(Some(BertNormalizer::default()))
        .with_pre_tokenizer(Some(BertPreTokenizer))
        .with_decoder(Some(WordPieceDecoder::default()))
        .with_post_processor(Some(BertProcessing::new(
            ("[SEP]".to_owned(), sep),
            ("[CLS]".to_owned(), cls),
        )));
    Ok(tokenizer)
}

pub(crate) fn embeddings_from_output(
    info: &ModelInfo,
    output: ArrayD<f32>,
    normalize_output: bool,
) -> Result<Vec<Embedding>, Error> {
    match output.ndim() {
        1 => {
            let mut values = output.iter().copied().collect::<Vec<_>>();
            if normalize_output {
                normalize_vector(&mut values)?;
            }
            Ok(vec![Embedding::from_vec(info, values)?])
        }
        2 => {
            let shape = output.shape().to_vec();
            let dims = shape[1];
            if dims != info.embedding_dim {
                return Err(Error::DimensionMismatch {
                    expected: info.embedding_dim,
                    actual: dims,
                });
            }

            let mut embeddings = Vec::with_capacity(shape[0]);
            let values = output.iter().copied().collect::<Vec<_>>();
            for row in values.chunks_exact(dims) {
                let mut row = row.to_vec();
                if normalize_output {
                    normalize_vector(&mut row)?;
                }
                embeddings.push(Embedding::from_vec(info, row)?);
            }
            Ok(embeddings)
        }
        ndim => Err(Error::ort(format!(
            "model output must be 1D or 2D, got {ndim}D tensor"
        ))),
    }
}

pub(crate) fn single_embedding(
    embeddings: Vec<Embedding>,
    label: &str,
) -> Result<Embedding, Error> {
    match embeddings.len() {
        1 => Ok(embeddings.into_iter().next().expect("len checked")),
        count => Err(Error::ort(format!(
            "{label} expected one embedding, got {count}"
        ))),
    }
}

fn normalize_vector(values: &mut [f32]) -> Result<(), Error> {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm <= f32::MIN_POSITIVE {
        return Err(Error::ort("model returned a zero-norm embedding"));
    }
    for value in values {
        *value /= norm;
    }
    Ok(())
}

fn load_session(model_path: &Path, runtime: &RuntimeConfig) -> Result<Session, Error> {
    let mut builder = Session::builder().map_err(Error::from_ort)?;
    builder = builder
        .with_optimization_level(map_graph_optimization_level(
            runtime.graph_optimization_level,
        ))
        .map_err(Error::from_ort)?;
    builder = builder
        .with_intra_threads(runtime.intra_threads)
        .map_err(Error::from_ort)?;
    if let Some(inter_threads) = runtime.inter_threads {
        builder = builder
            .with_inter_threads(inter_threads)
            .map_err(Error::from_ort)?;
    }
    builder.commit_from_file(model_path).map_err(|error| {
        Error::ort(format!(
            "failed to load ONNX model {}: {error}",
            model_path.display()
        ))
    })
}

fn map_graph_optimization_level(level: GraphOptimizationLevel) -> OrtGraphOptimizationLevel {
    match level {
        GraphOptimizationLevel::Disabled => OrtGraphOptimizationLevel::Disable,
        GraphOptimizationLevel::Basic => OrtGraphOptimizationLevel::Level1,
        GraphOptimizationLevel::Extended => OrtGraphOptimizationLevel::Level2,
        GraphOptimizationLevel::All => OrtGraphOptimizationLevel::All,
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::load_tokenizer;

    #[test]
    fn loads_wordpiece_vocab_txt() {
        let dir = tempdir().unwrap();
        let vocab_path = dir.path().join("vocab.txt");
        fs::write(
            &vocab_path,
            "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n你\n好\n",
        )
        .unwrap();

        let tokenizer = load_tokenizer(&vocab_path, 6, "[PAD]").unwrap();
        let encoding = tokenizer.encode("你好", true).unwrap();

        assert_eq!(encoding.get_ids(), &[2, 5, 6, 3, 0, 0]);
        assert_eq!(encoding.get_attention_mask(), &[1, 1, 1, 1, 0, 0]);
        assert_eq!(encoding.get_type_ids(), &[0, 0, 0, 0, 0, 0]);
    }
}
