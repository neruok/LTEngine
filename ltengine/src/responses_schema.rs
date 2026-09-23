//! Structured-output request handling for `POST /v1/responses` (PH-4a, RD-19).
//!
//! `text.format` follows the OpenAI Responses API contract. `strict: true`
//! selects constrained decoding: the schema becomes a GBNF grammar through
//! `llama_cpp_common::json_schema_to_grammar`, and the sampler is constrained with
//! it. No second decode path and no new dependency (`CC-3`, `CC-4`).
//!
//! `PH-4a` emits no `refusal` content part. A refusal is model behavior and
//! stays outside the permanent limits (`SP-NEVER-001`).
//!
//! `text.verbosity` is a best-effort, LTEngine-defined prompt directive
//! (`RD-30`). It is not OpenAI behavior, and a model can ignore it.

use serde::Deserialize;

/// OpenAI `text` request object. `format` selects structured output;
/// `verbosity` selects the `RD-30` prompt directive.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResponseTextConfig {
    #[serde(default)]
    pub format: Option<TextFormat>,
    /// OpenAI `text.verbosity`. A non-string or an unknown value is rejected
    /// by `derive_text`, naming the field (`SP-NEVER-010`).
    #[serde(default)]
    pub verbosity: Option<serde_json::Value>,
}

/// OpenAI `text.verbosity` (`RD-30`). `medium` is the default and appends no
/// directive.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Verbosity {
    Low,
    #[default]
    Medium,
    High,
}

impl Verbosity {
    /// The accepted values, in the order an error message lists them.
    pub const VALUES: [&'static str; 3] = ["low", "medium", "high"];

    /// The request value of this verbosity.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }

    /// The best-effort system directive of `RD-30` 4.2, or `None` for
    /// `medium`.
    pub fn directive(self) -> Option<&'static str> {
        match self {
            Self::Low => {
                Some("Be concise. Give the shortest complete answer and omit explanation.")
            }
            Self::High => Some(
                "Be thorough. Give a complete answer with explanation and supporting detail.",
            ),
            Self::Medium => None,
        }
    }

    fn decode(value: &serde_json::Value) -> Result<Self, String> {
        let Some(text) = value.as_str() else {
            return Err(format!(
                "`text.verbosity` must be one of {}",
                Self::VALUES.join(", ")
            ));
        };
        match text {
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            other => Err(format!(
                "`text.verbosity` `{other}` is not supported; use {}",
                Self::VALUES.join(", ")
            )),
        }
    }
}

/// The decoded consequence of the `text` request object.
#[derive(Debug, Default)]
pub struct TextEffect {
    /// The `text.format` effect (`PH-4a`).
    pub format: StructuredFormat,
    /// The effective `text.verbosity` (`RD-30`). `medium` when absent.
    pub verbosity: Verbosity,
}

/// Decode the `text` request object into its format and verbosity effects. An
/// error message names the rejected field, so the handler can return the
/// OpenAI-shaped 400 (`SP-MUST-011`).
pub fn derive_text(text: Option<&ResponseTextConfig>) -> Result<TextEffect, String> {
    let verbosity = match text.and_then(|text| text.verbosity.as_ref()) {
        None => Verbosity::Medium,
        Some(value) => Verbosity::decode(value)?,
    };
    let format = derive_format(text)?;
    Ok(TextEffect { format, verbosity })
}

/// OpenAI `text.format` object. `type` selects the format; `name`,
/// `description`, `schema`, and `strict` belong to `json_schema`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextFormat {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub schema: Option<serde_json::Value>,
    #[serde(default)]
    pub strict: Option<bool>,
}

/// The decoded consequence of `text.format` for one request.
#[derive(Debug, Default)]
pub struct StructuredFormat {
    /// The generated text must parse as JSON (`json_object` and `json_schema`).
    pub json_required: bool,
    /// The GBNF grammar that constrains decoding, when `strict` is true.
    pub grammar: Option<String>,
}

/// `text.format.name` is 1 to 64 characters of `a-z`, `A-Z`, `0-9`, `_`, `-`.
const FORMAT_NAME_MAX_CHARS: usize = 64;

/// Decode `text.format` into its effect. An error message names the rejected
/// field, so the handler can return the OpenAI-shaped 400 (`SP-MUST-011`).
pub fn derive_format(text: Option<&ResponseTextConfig>) -> Result<StructuredFormat, String> {
    let Some(text) = text else {
        return Ok(StructuredFormat::default());
    };
    let Some(format) = text.format.as_ref() else {
        return Ok(StructuredFormat::default());
    };
    match format.kind.as_str() {
        "text" => {
            reject_schema_members(format, "text")?;
            Ok(StructuredFormat::default())
        }
        "json_object" => {
            reject_schema_members(format, "json_object")?;
            // The OpenAI contract guarantees valid JSON in JSON mode, so the
            // decode is constrained to any JSON value, not just best effort.
            let grammar = llama_cpp_common::json_schema_to_grammar(r#"{"type":"object"}"#)
                .map_err(|err| format!("`text.format.type` `json_object` failed: {err}"))?;
            Ok(StructuredFormat {
                json_required: true,
                grammar: Some(grammar),
            })
        }
        "json_schema" => decode_json_schema(format),
        other => Err(format!(
            "`text.format.type` `{other}` is not supported; use `text`, `json_object`, or `json_schema`"
        )),
    }
}

/// `text` and `json_object` carry no schema members.
fn reject_schema_members(format: &TextFormat, kind: &str) -> Result<(), String> {
    if format.name.is_some()
        || format.description.is_some()
        || format.schema.is_some()
        || format.strict.is_some()
    {
        return Err(format!(
            "`text.format` `{kind}` does not accept `name`, `description`, `schema`, or `strict`"
        ));
    }
    Ok(())
}

/// `json_schema` requires `name` and `schema`. `strict` defaults to false, as
/// recorded in `responses-api-ph4.md` section 8.
fn decode_json_schema(format: &TextFormat) -> Result<StructuredFormat, String> {
    let Some(name) = format.name.as_deref() else {
        return Err("`text.format` requires `name` for type `json_schema`".to_string());
    };
    validate_name(name)?;
    let Some(schema) = format.schema.as_ref() else {
        return Err("`text.format` requires `schema` for type `json_schema`".to_string());
    };
    if !schema.is_object() {
        return Err("`text.format.schema` must be a JSON object".to_string());
    }
    let grammar = if format.strict.unwrap_or(false) {
        Some(
            llama_cpp_common::json_schema_to_grammar(&schema.to_string()).map_err(|err| {
                format!("`text.format.schema` is not a supported JSON schema: {err}")
            })?,
        )
    } else {
        None
    };
    Ok(StructuredFormat {
        json_required: true,
        grammar,
    })
}

fn validate_name(name: &str) -> Result<(), String> {
    let allowed = !name.is_empty()
        && name.chars().count() <= FORMAT_NAME_MAX_CHARS
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    if allowed {
        Ok(())
    } else {
        Err(format!(
            "`text.format.name` must be 1 to {FORMAT_NAME_MAX_CHARS} characters of a-z, A-Z, 0-9, `_`, or `-`"
        ))
    }
}




