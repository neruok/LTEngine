//! OpenAI `reasoning` request handling for `POST /v1/responses` (`RD-28`).
//!
//! `effort` is the one implemented member (`SP-PLANNED-013`). The remaining
//! members are part of the OpenAI contract and are rejected by name rather
//! than ignored (`SP-NEVER-010`).
//!
//! The effort value reaches the loaded model's chat template as its
//! `reasoning_effort` variable. `none` is the only value that turns the
//! template's `enable_thinking` variable off; every other value turns it on,
//! as does an object with no `effort`.

use serde::Deserialize;

use crate::llm::Reasoning;

/// The `reasoning.effort` values of the pinned SDKs (`RD-8`).
pub const EFFORT_VALUES: [&str; 7] =
    ["none", "minimal", "low", "medium", "high", "xhigh", "max"];

/// OpenAI `reasoning` request object.
///
/// Every member but `effort` is accepted by the contract and not implemented.
/// A `null` member is the same as an absent one.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResponseReasoning {
    #[serde(default)]
    pub context: Option<serde_json::Value>,
    #[serde(default)]
    pub effort: Option<serde_json::Value>,
    #[serde(default)]
    pub generate_summary: Option<serde_json::Value>,
    #[serde(default)]
    pub mode: Option<serde_json::Value>,
    #[serde(default)]
    pub summary: Option<serde_json::Value>,
}

/// The decoded consequence of `reasoning` for one request.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ReasoningEffect {
    /// The template's `enable_thinking` variable.
    pub thinking: bool,
    /// The `reasoning_effort` template variable.
    pub effort: Option<String>,
}

impl ReasoningEffect {
    /// Borrow as the generation-layer value.
    pub fn as_reasoning(&self) -> Reasoning<'_> {
        Reasoning {
            thinking: self.thinking,
            effort: self.effort.as_deref(),
        }
    }
}

/// Decode `reasoning` into its effect. An error message names the rejected
/// field, so the handler can return the OpenAI-shaped 400 (`SP-MUST-011`).
pub fn derive_reasoning(
    reasoning: Option<&ResponseReasoning>,
) -> Result<ReasoningEffect, String> {
    let Some(reasoning) = reasoning else {
        // An absent object keeps the behavior that predates `RD-28`, which is
        // the behavior of every request that does not ask for reasoning.
        return Ok(ReasoningEffect::default());
    };
    reject_unimplemented(reasoning)?;
    let effort = match reasoning.effort.as_ref() {
        None => None,
        Some(value) => Some(validate_effort(value)?),
    };
    let thinking = effort.as_deref() != Some("none");
    Ok(ReasoningEffect { thinking, effort })
}

/// The contract members this route does not implement (`SP-NEVER-010`).
fn reject_unimplemented(reasoning: &ResponseReasoning) -> Result<(), String> {
    for (present, field) in [
        (reasoning.context.is_some(), "reasoning.context"),
        (
            reasoning.generate_summary.is_some(),
            "reasoning.generate_summary",
        ),
        (reasoning.mode.is_some(), "reasoning.mode"),
        (reasoning.summary.is_some(), "reasoning.summary"),
    ] {
        if present {
            return Err(format!("`{field}` is not implemented on this route"));
        }
    }
    Ok(())
}

/// `reasoning.effort` is a string from the pinned SDK's value set.
fn validate_effort(value: &serde_json::Value) -> Result<String, String> {
    let Some(effort) = value.as_str() else {
        return Err("`reasoning.effort` must be a string".to_string());
    };
    if !EFFORT_VALUES.contains(&effort) {
        return Err(format!(
            "`reasoning.effort` `{effort}` is not supported; use one of {}",
            EFFORT_VALUES.join(", ")
        ));
    }
    Ok(effort.to_string())
}

#[cfg(test)]
mod tests {
    use super::{EFFORT_VALUES, ResponseReasoning, derive_reasoning};
    use serde_json::json;

    fn parse(value: serde_json::Value) -> Option<ResponseReasoning> {
        serde_json::from_value(value).ok()
    }

    /// `RD-28`, changed behavior: `reasoning` is accepted, and an absent object
    /// keeps the pre-`RD-28` behavior.
    #[test]
    fn accepts_an_absent_object_and_an_empty_one() {
        assert_eq!(
            derive_reasoning(None).expect("absent reasoning"),
            super::ReasoningEffect::default()
        );
        let empty = parse(json!({})).expect("empty object parses");
        assert_eq!(
            derive_reasoning(Some(&empty)).expect("empty reasoning"),
            super::ReasoningEffect {
                thinking: true,
                effort: None,
            }
        );
    }

    /// `RD-28`: every pinned-SDK value is accepted, and `none` is the only one
    /// that turns thinking off.
    #[test]
    fn accepts_every_pinned_effort_value() {
        for effort in EFFORT_VALUES {
            let reasoning = parse(json!({ "effort": effort })).expect("effort parses");
            let effect = derive_reasoning(Some(&reasoning)).expect("effort is valid");
            assert_eq!(effect.effort.as_deref(), Some(effort));
            assert_eq!(effect.thinking, effort != "none", "effort {effort}");
        }
    }

    /// `RD-28`: a value outside the contract names `reasoning.effort`.
    #[test]
    fn rejects_an_unsupported_effort_value() {
        let reasoning = parse(json!({ "effort": "enormous" })).expect("parses");
        let message = derive_reasoning(Some(&reasoning)).expect_err("unsupported effort");
        assert!(message.contains("reasoning.effort"), "{message}");
        assert!(message.contains("enormous"), "{message}");
    }

    /// `RD-28`: `effort` that is not a string names `reasoning.effort`.
    #[test]
    fn rejects_a_non_string_effort() {
        let reasoning = parse(json!({ "effort": 3 })).expect("parses");
        let message = derive_reasoning(Some(&reasoning)).expect_err("non-string effort");
        assert!(message.contains("reasoning.effort"), "{message}");
    }

    /// `SP-NEVER-010`: a contract member this route does not implement is
    /// rejected by name instead of being ignored.
    #[test]
    fn rejects_each_unimplemented_member() {
        for member in ["context", "generate_summary", "mode", "summary"] {
            let reasoning = parse(json!({ member: "auto" })).expect("parses");
            let message = derive_reasoning(Some(&reasoning)).expect_err("unimplemented");
            assert!(message.contains(member), "{member}: {message}");
        }
    }

    /// `SP-NEVER-010`: a member outside the contract is rejected by serde.
    #[test]
    fn rejects_an_unknown_member() {
        assert!(parse(json!({ "temperature": 0.5 })).is_none());
    }

    /// The contract allows an explicit `null`, which is the absent case.
    #[test]
    fn treats_null_members_as_absent() {
        let reasoning =
            parse(json!({ "effort": null, "summary": null })).expect("null members parse");
        assert_eq!(
            derive_reasoning(Some(&reasoning)).expect("null is accepted"),
            super::ReasoningEffect {
                thinking: true,
                effort: None,
            }
        );
    }
}
