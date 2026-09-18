//! `input` request handling for `POST /v1/responses` (PH-1, PH-4b).
//!
//! An `input` array holds messages and, since `PH-4b`, the tool items
//! `function_call` and `function_call_output`. A local text model has no native
//! tool template on this route, so the mapping folds both tool items into the
//! prompt text.

use serde::Deserialize;

/// `input` is a plain string or a message array.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ResponseInput {
    Text(String),
    Messages(Vec<InputItem>),
}

/// One `input` array entry. The OpenAI Responses API allows a message and the
/// tool items (`function_call` and `function_call_output`) in the same array
/// (PH-4b). The variant order selects the first shape that matches.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum InputItem {
    Message(Message),
    FunctionCall(FunctionCallItem),
    FunctionCallOutput(FunctionCallOutputItem),
}

/// An assistant tool call sent back as input. The route folds it into the
/// prompt text, because the local decode path is text-only. `id` and `status`
/// exist so an OpenAI SDK can echo the item back unchanged.
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionCallItem {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    pub call_id: String,
    pub name: String,
    #[serde(default)]
    pub arguments: Option<String>,
}

/// A tool result sent back as input (PH-4b). `id` and `status` exist so an
/// OpenAI SDK can echo the item back unchanged.
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionCallOutputItem {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    pub call_id: String,
    #[serde(default)]
    pub output: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Message {
    pub role: String,
    pub content: MessageContent,
}

#[derive(Debug, Deserialize)]
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
#[derive(Debug, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

const SYSTEM_ROLES: [&str; 2] = ["system", "developer"];
const TEXT_ROLES: [&str; 4] = ["system", "developer", "user", "assistant"];

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
                }
            }
        }
    }

    Ok((system.join("\n"), user.join("\n")))
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
