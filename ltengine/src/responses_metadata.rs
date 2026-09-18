//! `metadata` request-field validation for `POST /v1/responses` (RD-3).
//!
//! The OpenAI limits are at most 16 entries, each key at most 64 characters,
//! and each value at most 512 characters. `CreateRequest` validates a value at
//! parse time with `deserialize_metadata`, so the OpenAI-shaped 400 body names
//! `metadata` (`SP-MUST-011`).

use serde::Deserialize;

/// OpenAI `metadata` limits: at most 16 entries, key at most 64 characters,
/// value at most 512 characters. Every bound is inclusive.
pub(crate) const METADATA_MAX_ENTRIES: usize = 16;
pub(crate) const METADATA_MAX_KEY_CHARS: usize = 64;
pub(crate) const METADATA_MAX_VALUE_CHARS: usize = 512;

/// Reject an invalid `metadata` at parse time, so the OpenAI-shaped 400 body
/// carries a message that names `metadata` (SP-MUST-011).
pub(crate) fn deserialize_metadata<'de, D>(deserializer: D) -> Result<Option<serde_json::Value>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let metadata = Option::<serde_json::Value>::deserialize(deserializer)?;
    validate_metadata(metadata.as_ref()).map_err(serde::de::Error::custom)?;
    Ok(metadata)
}

fn validate_metadata(metadata: Option<&serde_json::Value>) -> Result<(), String> {
    let Some(metadata) = metadata.filter(|value| !value.is_null()) else {
        return Ok(());
    };
    let Some(entries) = metadata.as_object() else {
        return Err("metadata must be an object of string values".to_string());
    };
    if entries.len() > METADATA_MAX_ENTRIES {
        return Err(format!(
            "metadata must have at most {METADATA_MAX_ENTRIES} entries, got {}",
            entries.len()
        ));
    }
    for (key, value) in entries {
        if key.chars().count() > METADATA_MAX_KEY_CHARS {
            return Err(format!(
                "metadata key `{key}` is too long: at most {METADATA_MAX_KEY_CHARS} characters"
            ));
        }
        let Some(value) = value.as_str() else {
            return Err(format!("metadata value for key `{key}` must be a string"));
        };
        if value.chars().count() > METADATA_MAX_VALUE_CHARS {
            return Err(format!(
                "metadata value for key `{key}` is too long: at most {METADATA_MAX_VALUE_CHARS} characters"
            ));
        }
    }
    Ok(())
}
