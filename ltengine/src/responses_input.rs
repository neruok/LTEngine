//! `input` request handling for `POST /v1/responses` (PH-1, PH-4b, PH-5).
//!
//! An `input` array holds messages and, since `PH-4b`, the tool items
//! `function_call` and `function_call_output`. A local text model has no native
//! tool template on this route, so the mapping folds both tool items into the
//! prompt text.
//!
//! Since `PH-8b` (`RD-39`), an `input` array also accepts a `reasoning` item.
//! Its plain-text `content` parts fold into the prompt text, so a client can
//! round-trip a verbatim trace. Its `summary` and `encrypted_content` members
//! are accepted and not merged.
//!
//! `PH-5` also normalizes an input to JSON items for storage and listing
//! (`input_items`), converts a stored `output` array back to input items for
//! `previous_response_id` (`output_items_as_input`), and rebuilds the typed
//! input (`parse_input_items`).

use serde::{Deserialize, Serialize};

/// `input` is a plain string or a message array.
#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ResponseInput {
    Text(String),
    Messages(Vec<InputItem>),
}

/// One `input` array entry. The OpenAI Responses API allows a message and the
/// tool items (`function_call` and `function_call_output`) in the same array
/// (PH-4b). The variant order selects the first shape that matches.
#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum InputItem {
    Message(Message),
    FunctionCall(FunctionCallItem),
    FunctionCallOutput(FunctionCallOutputItem),
    Reasoning(ReasoningItem),
}

/// An assistant tool call sent back as input. The route folds it into the
/// prompt text, because the local decode path is text-only. `id` and `status`
/// exist so an OpenAI SDK can echo the item back unchanged.
#[allow(dead_code)]
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionCallItem {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    pub call_id: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// A tool result sent back as input (PH-4b). `id` and `status` exist so an
/// OpenAI SDK can echo the item back unchanged.
#[allow(dead_code)]
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionCallOutputItem {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    pub call_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<serde_json::Value>,
}

/// An input `reasoning` item (`RD-39`). Only the plain-text `content` parts
/// reach the prompt; `summary` and `encrypted_content` are accepted so a client
/// can echo a response item back unchanged. `id` and `status` exist for the
/// same reason.
#[allow(dead_code)]
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReasoningItem {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<ContentPart>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encrypted_content: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Message {
    pub role: String,
    pub content: MessageContent,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

/// `extra` captures every undeclared field, and `message_text` rejects a
/// captured field, so an unknown field is never silently ignored
/// (SP-NEVER-010). The capture is what lets a native modality payload (for
/// example `input_image` with `image_url`) reach the clear unsupported-modality
/// error instead of a generic unknown-field error (SP-MUST-011).
#[derive(Debug, Deserialize, Serialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

const SYSTEM_ROLES: [&str; 2] = ["system", "developer"];
const TEXT_ROLES: [&str; 4] = ["system", "developer", "user", "assistant"];

/// Normalize a request `input` to a JSON item array for storage and listing.
///
/// A string `input` becomes one `user` message with an `input_text` part, the
/// shape that `PH5-06` returns. An item of a message array keeps its shape.
pub fn input_items(input: &ResponseInput) -> Vec<serde_json::Value> {
    match input {
        ResponseInput::Text(text) => vec![serde_json::json!({
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        })],
        ResponseInput::Messages(items) => items
            .iter()
            .map(|item| serde_json::to_value(item).expect("an input item is serializable"))
            .collect(),
    }
}

/// Convert a stored `output` array to input items for `previous_response_id`
/// (`§3.5`). A `message` item becomes an assistant message with one
/// `output_text` part, and a `function_call` item keeps its tool fields. An
/// item of another type is skipped.
pub fn output_items_as_input(output: &serde_json::Value) -> Vec<serde_json::Value> {
    let Some(items) = output.as_array() else {
        return Vec::new();
    };
    items.iter().filter_map(output_item_as_input).collect()
}

fn output_item_as_input(item: &serde_json::Value) -> Option<serde_json::Value> {
    match item["type"].as_str() {
        Some("message") => {
            let text = item["content"]
                .as_array()
                .and_then(|parts| parts.iter().find_map(|part| part["text"].as_str()))
                .unwrap_or_default();
            Some(serde_json::json!({
                "role": item["role"].as_str().unwrap_or("assistant"),
                "content": [{"type": "output_text", "text": text}],
            }))
        }
        Some("function_call") => Some(serde_json::json!({
            "type": "function_call",
            "call_id": item["call_id"],
            "name": item["name"],
            "arguments": item["arguments"],
        })),
        // A stored reasoning item round-trips as input (`RD-39`).
        Some("reasoning") => Some(item.clone()),
        _ => None,
    }
}

/// Rebuild the typed input from a JSON item array. Every stored item was
/// accepted once, so a failure here is an internal error, not a request error.
pub fn parse_input_items(items: Vec<serde_json::Value>) -> Result<ResponseInput, String> {
    serde_json::from_value::<Vec<InputItem>>(serde_json::Value::Array(items))
        .map(ResponseInput::Messages)
        .map_err(|err| format!("invalid input item: {err}"))
}

/// Convert stored conversation items to input items for the prompt (`PH-6a`).
///
/// A message item keeps its `role` and `content`, with each content part
/// normalized to `{type, text}`. That drops a stored `id`, `status`, or
/// `annotations` field before the strict `InputItem` parsing, and it skips an
/// item that carries an unsupported part. A `function_call` or
/// `function_call_output` item keeps its tool fields.
pub fn stored_items_as_input(items: &[serde_json::Value]) -> Vec<serde_json::Value> {
    items.iter().filter_map(stored_item_as_input).collect()
}

fn stored_item_as_input(item: &serde_json::Value) -> Option<serde_json::Value> {
    if let Some(role) = item.get("role").and_then(serde_json::Value::as_str) {
        let content = match item.get("content")? {
            serde_json::Value::String(text) => serde_json::Value::String(text.clone()),
            serde_json::Value::Array(parts) => {
                let mut normalized = Vec::with_capacity(parts.len());
                for part in parts {
                    normalized.push(normalized_part(part)?);
                }
                serde_json::Value::Array(normalized)
            }
            _ => return None,
        };
        return Some(serde_json::json!({"role": role, "content": content}));
    }
    match item.get("type").and_then(serde_json::Value::as_str) {
        Some("function_call") => Some(serde_json::json!({
            "type": "function_call",
            "call_id": item.get("call_id")?,
            "name": item.get("name")?,
            "arguments": item
                .get("arguments")
                .cloned()
                .unwrap_or(serde_json::Value::String("{}".to_string())),
        })),
        Some("function_call_output") => Some(serde_json::json!({
            "type": "function_call_output",
            "call_id": item.get("call_id")?,
            "output": item.get("output").cloned().unwrap_or(serde_json::Value::Null),
        })),
        // A stored reasoning item keeps its shape, so `previous_response_id`
        // and a conversation round-trip the trace (`RD-39`).
        Some("reasoning") => Some(item.clone()),
        _ => None,
    }
}

fn normalized_part(part: &serde_json::Value) -> Option<serde_json::Value> {
    let kind = part.get("type").and_then(serde_json::Value::as_str)?;
    if kind != "input_text" && kind != "output_text" {
        return None;
    }
    Some(serde_json::json!({
        "type": kind,
        "text": part.get("text").and_then(serde_json::Value::as_str).unwrap_or_default(),
    }))
}

/// Map `instructions` and `input` onto the `run_prompt(system, user)` pair.
///
/// `ponytail:` message roles collapse into two text blocks, so a multi-turn
/// conversation is flattened. Fine for a single text generation; revisit if a
/// client needs the turn structure preserved.
pub fn map_input(
    instructions: Option<&str>,
    input: &ResponseInput,
) -> Result<(String, String), String> {
    let mut system: Vec<String> = Vec::new();
    if let Some(instructions) = instructions {
        system.push(instructions.to_string());
    }

    let mut user: Vec<String> = Vec::new();
    match input {
        ResponseInput::Text(text) => user.push(text.clone()),
        ResponseInput::Messages(items) => {
            for item in items {
                match item {
                    InputItem::Message(message) => {
                        let text = message_text(message)?;
                        if SYSTEM_ROLES.contains(&message.role.as_str()) {
                            system.push(text);
                        } else {
                            user.push(text);
                        }
                    }
                    InputItem::FunctionCall(call) => {
                        user.push(format!(
                            "Tool call `{}` ({}) with arguments {}",
                            call.name,
                            call.call_id,
                            call.arguments.as_deref().unwrap_or("{}")
                        ));
                    }
                    InputItem::FunctionCallOutput(output) => {
                        let text = match &output.output {
                            Some(serde_json::Value::String(text)) => text.clone(),
                            Some(value) => value.to_string(),
                            None => String::new(),
                        };
                        user.push(format!("Tool result for `{}`: {}", output.call_id, text));
                    }
                    InputItem::Reasoning(reasoning) => {
                        // `RD-39`: the plain-text content merges into the
                        // prompt at this position. `summary` does not.
                        let text = reasoning_text(reasoning)?;
                        if !text.is_empty() {
                            user.push(text);
                        }
                    }
                }
            }
        }
    }

    Ok((system.join("\n"), user.join("\n")))
}

/// The plain-text `content` of an input reasoning item (`RD-39`). A part type
/// other than `reasoning_text` or `output_text` is a request error that names
/// the type.
fn reasoning_text(item: &ReasoningItem) -> Result<String, String> {
    let Some(parts) = item.content.as_ref() else {
        return Ok(String::new());
    };
    let mut texts = Vec::new();
    for part in parts {
        if part.kind != "reasoning_text" && part.kind != "output_text" {
            return Err(format!("unsupported reasoning content type: {}", part.kind));
        }
        if let Some(field) = part.extra.keys().next() {
            return Err(format!(
                "unsupported reasoning content field `{field}` on content type {}",
                part.kind
            ));
        }
        if let Some(text) = part.text.as_ref() {
            texts.push(text.clone());
        }
    }
    Ok(texts.join("\n"))
}

fn message_text(message: &Message) -> Result<String, String> {
    if !TEXT_ROLES.contains(&message.role.as_str()) {
        return Err(format!("unsupported message role: {}", message.role));
    }

    match &message.content {
        MessageContent::Text(text) => Ok(text.clone()),
        MessageContent::Parts(parts) => {
            let mut texts = Vec::new();
            for part in parts {
                if part.kind != "input_text" && part.kind != "output_text" {
                    return Err(format!("unsupported modality: {}", part.kind));
                }
                if let Some(field) = part.extra.keys().next() {
                    return Err(format!(
                        "unsupported modality field `{field}` on content type {}",
                        part.kind
                    ));
                }
                let text = part
                    .text
                    .clone()
                    .ok_or_else(|| "content part is missing text".to_string())?;
                texts.push(text);
            }
            Ok(texts.join("\n"))
        }
    }
}
