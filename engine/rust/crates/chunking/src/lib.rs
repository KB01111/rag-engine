use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ChunkingError {
    #[error("empty text provided")]
    EmptyText,
    #[error("chunk size must be greater than overlap")]
    InvalidChunkSize,
    #[error("invalid UTF-8 text")]
    InvalidUtf8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub text: String,
    pub index: usize,
    pub start_char: usize,
    pub end_char: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfig {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub min_chunk_size: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            chunk_overlap: 50,
            min_chunk_size: 50,
        }
    }
}

pub fn chunk_text(text: &str, config: &ChunkingConfig) -> Result<Vec<Chunk>, ChunkingError> {
    if text.is_empty() {
        return Err(ChunkingError::EmptyText);
    }

    if config.chunk_size <= config.chunk_overlap {
        return Err(ChunkingError::InvalidChunkSize);
    }

    let chars: Vec<char> = text.chars().collect();
    let total_len = chars.len();

    if total_len < config.min_chunk_size {
        return Ok(vec![Chunk {
            text: text.to_string(),
            index: 0,
            start_char: 0,
            end_char: total_len,
        }]);
    }

    let mut chunks = Vec::new();
    let mut index = 0;
    let mut position = 0;

    while position < total_len {
        let end = (position + config.chunk_size).min(total_len);

        let mut chunk_end = end;
        if end < total_len {
            if let Some(last_space) = chars[position..end]
                .iter()
                .rposition(|&c| c == ' ' || c == '\n' || c == '\t')
            {
                chunk_end = position + last_space;
            }
        }

        let chunk_text: String = chars[position..chunk_end].iter().collect();

        if chunk_text.len() >= config.min_chunk_size || chunks.is_empty() {
            chunks.push(Chunk {
                text: chunk_text,
                index,
                start_char: position,
                end_char: chunk_end,
            });
            index += 1;
        }

        if chunk_end >= total_len {
            break;
        }

        let advance = config.chunk_size - config.chunk_overlap;
        position = (chunk_end - advance).max(position + 1);

        if position >= total_len {
            break;
        }
    }

    Ok(chunks)
}

pub fn chunk_text_simple(text: &str, chunk_size: usize, overlap: usize) -> Vec<Chunk> {
    let config = ChunkingConfig {
        chunk_size,
        chunk_overlap: overlap,
        min_chunk_size: 50,
    };

    chunk_text(text, &config).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_text_basic() {
        let text =
            "This is a test document. It has multiple sentences. We want to chunk it properly.";
        let config = ChunkingConfig {
            chunk_size: 30,
            chunk_overlap: 10,
            min_chunk_size: 10,
        };

        let chunks = chunk_text(text, &config).unwrap();
        assert!(!chunks.is_empty());
        assert!(chunks.iter().all(|c| c.text.len() <= config.chunk_size));
    }

    #[test]
    fn test_empty_text() {
        let text = "";
        let config = ChunkingConfig::default();

        let result = chunk_text(text, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_small_text() {
        let text = "Short text.";
        let config = ChunkingConfig::default();

        let chunks = chunk_text(text, &config).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Short text.");
    }
}
