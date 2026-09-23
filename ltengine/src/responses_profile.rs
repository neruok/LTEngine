//! The `--responses-api` compatibility profile (`RD-40`, `RD-41`).
//!
//! The profile selects the behavior at each point where the OpenAI Responses
//! API, the DeepSeek Responses API, and the Open Responses specification
//! disagree. `responses-api-ph8.md` section 11 holds the divergence table.
//!
//! The option is required and has no default (`RD-40`). The profile sets the
//! default for each divergence point that `PH-8a` implements; the remaining
//! points belong to `PH-8b`.

use clap::ValueEnum;

use crate::llm::{DEFAULT_TEMPERATURE, DEFAULT_TOP_P, Generation};

/// The Responses API compatibility profile (`RD-40`).
///
/// `clap` derives the command-line values `openai`, `deepseek`, and
/// `open-responses` from the variant names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum ResponsesApi {
    Openai,
    Deepseek,
    OpenResponses,
}

impl ResponsesApi {
    /// `RD-40` row 4: `openai` and `open-responses` append the `RD-30`
    /// directive; `deepseek` accepts `text.verbosity` with no effect.
    pub fn applies_verbosity_directive(self) -> bool {
        !matches!(self, Self::Deepseek)
    }

    /// `RD-40` row 8 and `RD-41`: only `deepseek` ignores an unsupported
    /// request field. An unsupported tool and an unsupported modality keep the
    /// error in every profile.
    pub fn ignores_unsupported_fields(self) -> bool {
        matches!(self, Self::Deepseek)
    }

    /// `RD-40` row 3: the value that reaches the chat template for an accepted
    /// `reasoning.effort` value. The accepted set is the same in every profile;
    /// `deepseek` maps `minimal` to `low` and `medium`/`xhigh` to `high`.
    pub fn map_effort(self, value: &str) -> String {
        match (self, value) {
            (Self::Deepseek, "minimal") => "low".to_string(),
            (Self::Deepseek, "medium") | (Self::Deepseek, "xhigh") => "high".to_string(),
            _ => value.to_string(),
        }
    }

    /// `RD-40` row 5: under `deepseek` and thinking mode, `temperature` has no
    /// effect and `top_p` is clamped to at least `0.95`. Every other profile
    /// applies the requested values.
    pub fn sampling(self, thinking: bool, generation: Generation) -> Generation {
        if self == Self::Deepseek && thinking {
            return Generation {
                temperature: DEFAULT_TEMPERATURE,
                top_p: generation.top_p.max(DEFAULT_TOP_P),
                max_output_tokens: generation.max_output_tokens,
            };
        }
        generation
    }
}

#[cfg(test)]
mod tests {
    use super::ResponsesApi;
    use crate::llm::Generation;

    /// `PH8-29`: `deepseek` maps the effort values of `RD-40` row 3; the other
    /// profiles pass every accepted value through.
    #[test]
    fn ph8_29_deepseek_maps_effort() {
        for profile in [ResponsesApi::Openai, ResponsesApi::OpenResponses] {
            for value in ["minimal", "medium", "xhigh", "low"] {
                assert_eq!(profile.map_effort(value), value);
            }
        }
        assert_eq!(ResponsesApi::Deepseek.map_effort("minimal"), "low");
        assert_eq!(ResponsesApi::Deepseek.map_effort("medium"), "high");
        assert_eq!(ResponsesApi::Deepseek.map_effort("xhigh"), "high");
        assert_eq!(ResponsesApi::Deepseek.map_effort("low"), "low");
        assert_eq!(ResponsesApi::Deepseek.map_effort("none"), "none");
    }

    /// `PH8-29`: `deepseek` and thinking mode ignore `temperature` and clamp
    /// `top_p`; every other case is unchanged.
    #[test]
    fn ph8_29_deepseek_clamps_thinking_sampling() {
        let requested = Generation {
            temperature: 1.5,
            top_p: 0.4,
            max_output_tokens: Some(9),
        };
        let clamped = ResponsesApi::Deepseek.sampling(true, requested);
        assert_eq!(clamped.temperature, 0.0);
        assert_eq!(clamped.top_p, 0.95);
        assert_eq!(clamped.max_output_tokens, Some(9));
        // Thinking off keeps the requested values.
        assert_eq!(ResponsesApi::Deepseek.sampling(false, requested), requested);
        // The other profiles keep the requested values in thinking mode.
        assert_eq!(ResponsesApi::Openai.sampling(true, requested), requested);
        assert_eq!(
            ResponsesApi::OpenResponses.sampling(true, requested),
            requested
        );
        // A `top_p` already above the clamp is unchanged.
        let high = Generation {
            top_p: 0.99,
            ..requested
        };
        assert_eq!(ResponsesApi::Deepseek.sampling(true, high).top_p, 0.99);
    }

    /// `PH8-29`: only `deepseek` suppresses the verbosity directive.
    #[test]
    fn ph8_29_only_deepseek_skips_the_directive() {
        assert!(ResponsesApi::Openai.applies_verbosity_directive());
        assert!(ResponsesApi::OpenResponses.applies_verbosity_directive());
        assert!(!ResponsesApi::Deepseek.applies_verbosity_directive());
    }

    /// `PH8-26`: only `deepseek` ignores an unsupported request field.
    #[test]
    fn ph8_26_only_deepseek_ignores_unsupported_fields() {
        assert!(ResponsesApi::Deepseek.ignores_unsupported_fields());
        assert!(!ResponsesApi::Openai.ignores_unsupported_fields());
        assert!(!ResponsesApi::OpenResponses.ignores_unsupported_fields());
    }
}
