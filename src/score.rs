use crate::embedding::Embedding;
use crate::error::Error;

#[derive(Clone, Debug, PartialEq)]
pub struct Scored<T> {
    pub item: T,
    pub score: f32,
}

pub fn cosine_similarity(lhs: &[f32], rhs: &[f32]) -> Result<f32, Error> {
    if lhs.len() != rhs.len() {
        return Err(Error::DimensionMismatch {
            expected: lhs.len(),
            actual: rhs.len(),
        });
    }
    if lhs.is_empty() {
        return Err(Error::invalid_config("cannot score empty vectors"));
    }

    let lhs_norm = lhs.iter().map(|value| value * value).sum::<f32>().sqrt();
    let rhs_norm = rhs.iter().map(|value| value * value).sum::<f32>().sqrt();
    if lhs_norm <= f32::MIN_POSITIVE || rhs_norm <= f32::MIN_POSITIVE {
        return Err(Error::invalid_config(
            "cannot score vectors with zero magnitude",
        ));
    }

    let dot = lhs
        .iter()
        .zip(rhs)
        .map(|(left, right)| left * right)
        .sum::<f32>();
    Ok(dot / (lhs_norm * rhs_norm))
}

pub fn score_embeddings(lhs: &Embedding, rhs: &Embedding) -> Result<f32, Error> {
    if lhs.model_family != rhs.model_family || lhs.model_id != rhs.model_id {
        return Err(Error::EmbeddingModelMismatch {
            left_model_id: lhs.model_id.clone(),
            right_model_id: rhs.model_id.clone(),
        });
    }
    cosine_similarity(lhs.as_slice(), rhs.as_slice())
}

pub fn top_k<T, V>(
    query: &[f32],
    candidates: impl IntoIterator<Item = (T, V)>,
    k: usize,
) -> Result<Vec<Scored<T>>, Error>
where
    V: AsRef<[f32]>,
{
    if k == 0 {
        return Ok(Vec::new());
    }

    let mut scored = candidates
        .into_iter()
        .map(|(item, values)| {
            cosine_similarity(query, values.as_ref()).map(|score| Scored { item, score })
        })
        .collect::<Result<Vec<_>, _>>()?;
    scored.sort_by(|left, right| right.score.total_cmp(&left.score));
    scored.truncate(scored.len().min(k));
    Ok(scored)
}

#[cfg(test)]
mod tests {
    use crate::bundle::ModelInfo;
    use crate::config::ModelFamily;
    use crate::embedding::Embedding;
    use crate::score::{cosine_similarity, score_embeddings, top_k};

    #[test]
    fn scores_cosine_similarity() {
        let score = cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]).unwrap();
        assert_eq!(score, 1.0);
    }

    #[test]
    fn checks_embedding_model_identity() {
        let info = ModelInfo {
            model_family: ModelFamily::FgClip,
            model_id: "fgclip".to_owned(),
            model_revision: "1".to_owned(),
            embedding_dim: 2,
            normalize_output: true,
        };
        let left = Embedding::from_vec(&info, vec![1.0, 0.0]).unwrap();
        let mut other_info = info.clone();
        other_info.model_id = "other".to_owned();
        let right = Embedding::from_vec(&other_info, vec![1.0, 0.0]).unwrap();
        assert!(score_embeddings(&left, &right).is_err());
    }

    #[test]
    fn sorts_top_k_descending() {
        let results = top_k(&[1.0, 0.0], [("a", [1.0, 0.0]), ("b", [0.0, 1.0])], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].item, "a");
    }
}
