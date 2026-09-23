//! Generation-parameter request handling for `POST /v1/responses` (`RD-29`).
//!
//! `temperature`, `top_p`, and `max_output_tokens` are the OpenAI generation
//! controls. Each is optional. An absent or `null` value keeps the value that
//! `PH-1` through `PH-7` used, so a request without them keeps the current
//! sampler, context size, and response body (`INVARIANT PH8-1`).
//!
//! The sampler values are f32 because the llama.cpp sampler takes f32. The echo
//! keeps the request's f64 value, so the body reports the number the client
//! sent.

use crate::llm::{DEFAULT_TEMPERATURE, DEFAULT_TOP_P, Generation};
use crate::responses::CreateRequest;

/// The request echo of the generation controls. A field is `Some` only when the
/// request carried it, so an absent field stays absent in the response body
/// (`RD-29` 3.4, `INVARIANT PH8-1`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GenerationEcho {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub max_output_tokens: Option<u32>,
}

/// The decoded generation controls of one request.
#[derive(Clone, Copy, Debug, Default)]
pub struct GenerationEffect {
    /// The values that reach the sampler and the context sizing (`RD-29`).
    pub generation: Generation,
    /// The values that the response echoes.
    pub echo: GenerationEcho,
}

/// Decode `temperature`, `top_p`, and `max_output_tokens`. An error message
/// names the rejected field, so the handler can return the OpenAI-shaped 400
/// (`SP-MUST-011`).
pub fn derive_generation(request: &CreateRequest) -> Result<GenerationEffect, String> {
    let temperature = match request.temperature.as_ref() {
        None => None,
        Some(value) => Some(number_in_range(value, "temperature", 0.0, 2.0)?),
    };
    let top_p = match request.top_p.as_ref() {
        None => None,
        Some(value) => Some(number_in_range(value, "top_p", 0.0, 1.0)?),
    };
    let max_output_tokens = match request.max_output_tokens.as_ref() {
        None => None,
        Some(value) => Some(integer_at_least_one(value, "max_output_tokens")?),
    };
    Ok(GenerationEffect {
        generation: Generation {
            temperature: temperature.unwrap_or(f64::from(DEFAULT_TEMPERATURE)) as f32,
            top_p: top_p.unwrap_or(f64::from(DEFAULT_TOP_P)) as f32,
            max_output_tokens,
        },
        echo: GenerationEcho {
            temperature,
            top_p,
            max_output_tokens,
        },
    })
}

/// A JSON number inside the inclusive range. A string, a boolean, or `null`
/// (already decoded to `None`) is not a number.
fn number_in_range(
    value: &serde_json::Value,
    field: &str,
    min: f64,
    max: f64,
) -> Result<f64, String> {
    let Some(number) = value.as_f64() else {
        return Err(format!("`{field}` must be a number between {min} and {max}"));
    };
    if number < min || number > max {
        return Err(format!("`{field}` must be between {min} and {max}"));
    }
    Ok(number)
}

/// A JSON integer in the inclusive range 1–`i32::MAX`. A float with a
/// fractional part is not an integer.
fn integer_at_least_one(value: &serde_json::Value, field: &str) -> Result<u32, String> {
    let Some(number) = value.as_i64() else {
        return Err(format!(
            "`{field}` must be an integer between 1 and {}",
            i32::MAX
        ));
    };
    if number < 1 || number > i64::from(i32::MAX) {
        return Err(format!(
            "`{field}` must be an integer between 1 and {}",
            i32::MAX
        ));
    }
    Ok(number as u32)
}

#[cfg(test)]
mod tests {
    use super::{derive_generation, number_in_range};
    use serde_json::json;

    #[test]
    fn range_rejects_a_non_number_and_a_boundary_violation() {
        assert!(number_in_range(&json!(1.5), "x", 0.0, 2.0).is_ok());
        assert!(number_in_range(&json!("1"), "x", 0.0, 2.0).is_err());
        assert!(number_in_range(&json!(-0.1), "x", 0.0, 2.0).is_err());
    }

    #[test]
    fn an_absent_control_is_the_pre_rd29_default() {
        let request: crate::responses::CreateRequest = serde_json::from_str(r#"{"input":"hi"}"#)
            .expect("request should parse");
        let effect = derive_generation(&request).expect("no control is valid");
        assert_eq!(effect.generation.temperature, 0.0);
        assert_eq!(effect.generation.top_p, 0.95);
        assert_eq!(effect.generation.max_output_tokens, None);
        assert_eq!(effect.echo, super::GenerationEcho::default());
    }
}
