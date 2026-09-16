//! `POST /v1/responses` (roadmap PH-1 and PH-3).
//!
//! The route is a text adapter over `LLM::run_prompt_usage`. It is not an
//! OpenAI proxy: no upstream call, no second decode path. Only this route uses
//! the OpenAI-shaped nested error body; the existing LibreTranslate routes keep
//! their flat `{"error": "<string>"}` body.
//!
//! `stream: true` answers with `text/event-stream` (PH-3, RD-17). The body is one
//! generation attempt and carries the exact token usage of RD-18.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use actix_web::{HttpRequest, HttpResponse, http::StatusCode, http::header, post, web};
use serde::Deserialize;

use crate::Args;
use crate::llm;

/// Request body of `POST /v1/responses`.
///
/// `deny_unknown_fields` rejects `tools` and every other unsupported field with
/// a 400 instead of silently ignoring it. `metadata` is the one carried field:
/// its OpenAI limits are validated, and an accepted value is echoed.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CreateRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub instructions: Option<String>,
    pub input: ResponseInput,
    #[serde(default)]
    pub stream: Option<bool>,
    /// The OpenAI request metadata. Omitted and `null` are accepted. Otherwise
    /// it is an object of string entries inside the limits of
    /// `METADATA_MAX_ENTRIES`, `METADATA_MAX_KEY_CHARS`, and
    /// `METADATA_MAX_VALUE_CHARS`. An accepted value is echoed in the
    /// successful response (`null` when absent or `null`).
    #[serde(default, deserialize_with = "deserialize_metadata")]
    pub metadata: Option<serde_json::Value>,
    /// The LibreTranslate-style body credential. This route does not accept it
    /// (RD-4); `parse_and_authorize` rejects it with a clear 400.
    #[serde(default)]
    pub api_key: Option<serde_json::Value>,
}

/// OpenAI `metadata` limits: at most 16 entries, key at most 64 characters,
/// value at most 512 characters. Every bound is inclusive.
const METADATA_MAX_ENTRIES: usize = 16;
const METADATA_MAX_KEY_CHARS: usize = 64;
const METADATA_MAX_VALUE_CHARS: usize = 512;

/// Reject an invalid `metadata` at parse time, so the OpenAI-shaped 400 body
/// carries a message that names `metadata` (SP-MUST-011).
fn deserialize_metadata<'de, D>(deserializer: D) -> Result<Option<serde_json::Value>, D::Error>
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

/// `input` is a plain string or a message array.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ResponseInput {
    Text(String),
    Messages(Vec<Message>),
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
        ResponseInput::Messages(messages) => {
            for message in messages {
                let text = message_text(message)?;
                if SYSTEM_ROLES.contains(&message.role.as_str()) {
                    system.push(text);
                } else {
                    user.push(text);
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

/// Canonical loaded-model identifier shared by request validation and the
/// success response echo (SP-MUST-009, RD-5). `--model-file` overrides
/// `--model`, so the selected local GGUF string is the identity in that case.
fn loaded_model_identifier(args: &Args) -> String {
    if args.model_file.is_empty() {
        args.model.clone()
    } else {
        args.model_file.clone()
    }
}

/// Validate every supported-statefulness and model choice before generation.
///
/// `stream` is not validated here: `stream: true` selects the SSE response of
/// `build_stream_body` (RD-17). Every rejection below happens before the first
/// event, so it uses the normal OpenAI-shaped HTTP error (RD-13).
fn validate(
    request: &CreateRequest,
    loaded_model: &str,
) -> Result<(String, String), (u16, String)> {
    if let Some(model) = &request.model {
        if model != loaded_model {
            return Err((
                400,
                format!(
                    "model `{}` is not loaded; loaded model is `{}`",
                    model, loaded_model
                ),
            ));
        }
    }
    map_input(request.instructions.as_deref(), &request.input).map_err(|err| (400, err))
}

/// Parse the body before the credential check. This order is RD-4: a request
/// that carries the body field `api_key` is a 400 even when `Authorization` is
/// absent or wrong, never a 401.
fn parse_and_authorize(
    body: &[u8],
    api_key: &str,
    authorization: Option<&str>,
) -> Result<CreateRequest, (u16, String)> {
    let request: CreateRequest = serde_json::from_slice(body)
        .map_err(|err| (400, format!("invalid request body: {}", err)))?;
    if request.api_key.is_some() {
        return Err((
            400,
            "the body field `api_key` is not supported on this route; send `Authorization: Bearer <key>`"
                .to_string(),
        ));
    }
    check_auth(api_key, authorization).map_err(|message| (401, message))?;
    Ok(request)
}

/// The new route accepts `Authorization: Bearer` only (RD-4).
fn check_auth(api_key: &str, authorization: Option<&str>) -> Result<(), String> {
    if api_key.is_empty() {
        return Ok(());
    }
    match authorization {
        Some(value) if value == format!("Bearer {api_key}") => Ok(()),
        Some(_) => Err("incorrect API key provided".to_string()),
        None => Err("missing bearer token".to_string()),
    }
}

/// Convert a payload-extractor failure (for example the default oversized-body
/// `413`) into the same OpenAI-shaped nested error body as every other route
/// error. Without this the extractor rejection bypasses the handler and Actix
/// answers with `text/plain`.
fn extractor_error(err: actix_web::Error) -> HttpResponse {
    let status = err.as_response_error().status_code();
    error_json(status.as_u16(), format!("invalid request body: {}", err))
}

fn error_json(status: u16, message: String) -> HttpResponse {
    let error_type = if status >= 500 {
        "server_error"
    } else {
        "invalid_request_error"
    };
    HttpResponse::build(StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR))
        .json(serde_json::json!({
            "error": {
                "message": message,
                "type": error_type,
                "param": serde_json::Value::Null,
                "code": serde_json::Value::Null,
            }
        }))
}

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

fn new_id(prefix: &str) -> String {
    let count = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{prefix}{nanos:x}{count:x}")
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// One `output_text` content part.
fn output_text_part(text: &str) -> serde_json::Value {
    serde_json::json!({"type": "output_text", "text": text, "annotations": []})
}

/// One assistant `message` output item. `text` is `None` while the item is in
/// progress, because its content part arrives in a later event.
fn message_item(item_id: &str, status: &str, text: Option<&str>) -> serde_json::Value {
    let content = match text {
        Some(text) => serde_json::json!([output_text_part(text)]),
        None => serde_json::json!([]),
    };
    serde_json::json!({
        "id": item_id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": content,
    })
}

/// The exact `usage` object of RD-18. `total_tokens` is the exact sum of the two
/// exact counts. No value is an estimate (`SP-NEVER-010`).
fn usage_json(usage: &llm::TokenUsage) -> serde_json::Value {
    serde_json::json!({
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens(),
    })
}

/// The OpenAI-shaped response object, shared by the non-streaming body and the
/// `response.completed` and `response.in_progress` events.
fn response_object(
    id: &str,
    created_at: u64,
    status: &str,
    model: &str,
    metadata: Option<&serde_json::Value>,
    output: serde_json::Value,
    usage: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "model": model,
        "metadata": metadata,
        "parallel_tool_calls": false,
        "tool_choice": "none",
        "tools": [],
        "output": output,
        "usage": usage,
    })
}

/// OpenAI-shaped non-streaming response. `usage` carries the exact counts of
/// the one generation (PH-3, RD-18). `parallel_tool_calls`, `tool_choice`, and
/// `tools` are the truthful values of the text-only route: it offers no tool and
/// calls no tool. `metadata` echoes the accepted request value, or `null` when
/// the request carried none.
fn build_response(
    model: &str,
    text: &str,
    metadata: Option<&serde_json::Value>,
    usage: &llm::TokenUsage,
) -> serde_json::Value {
    response_object(
        &new_id("resp_"),
        now_secs(),
        "completed",
        model,
        metadata,
        serde_json::json!([message_item(&new_id("msg_"), "completed", Some(text))]),
        usage_json(usage),
    )
}

/// Append one SSE frame. The frame carries the OpenAI event name line and the
/// JSON payload line. The payload carries its event `type` and the next
/// `sequence_number` (RD-17).
fn push_event(
    body: &mut String,
    sequence: &mut u32,
    event_type: &str,
    mut payload: serde_json::Value,
) {
    let map = payload
        .as_object_mut()
        .expect("an event payload is a JSON object");
    map.insert("type".to_string(), serde_json::Value::from(event_type));
    map.insert(
        "sequence_number".to_string(),
        serde_json::Value::from(*sequence),
    );
    *sequence += 1;
    body.push_str("event: ");
    body.push_str(event_type);
    body.push_str("\ndata: ");
    body.push_str(&payload.to_string());
    body.push_str("\n\n");
}

/// Build the SSE body of a successful `stream: true` request (RD-17).
///
/// The event order is exactly `response.created`, `response.in_progress`,
/// `response.output_item.added`, `response.content_part.added`, zero or more
/// `response.output_text.delta`, `response.output_text.done`,
/// `response.content_part.done`, `response.output_item.done`, and
/// `response.completed`. `PH-3` makes one generation attempt and emits no
/// `response.failed`, because every failure happens before the first event
/// (RD-13).
///
/// `ponytail:` the body is built after the single generation attempt, so it
/// carries one delta with the whole text. RD-17 permits zero or more delta
/// events, and `clean_output` can retract text at the end, so the whole text is
/// the only safe delta. Emit finer deltas when the decode path can hand out text
/// that the cleanup will not retract.
fn build_stream_body(
    model: &str,
    text: &str,
    metadata: Option<&serde_json::Value>,
    usage: &llm::TokenUsage,
) -> String {
    let response_id = new_id("resp_");
    let item_id = new_id("msg_");
    let created_at = now_secs();
    let output_index = 0;
    let content_index = 0;
    let in_progress = response_object(
        &response_id,
        created_at,
        "in_progress",
        model,
        metadata,
        serde_json::json!([]),
        serde_json::Value::Null,
    );
    let completed = response_object(
        &response_id,
        created_at,
        "completed",
        model,
        metadata,
        serde_json::json!([message_item(&item_id, "completed", Some(text))]),
        usage_json(usage),
    );

    let mut sequence = 0_u32;
    let mut body = String::new();
    push_event(
        &mut body,
        &mut sequence,
        "response.created",
        serde_json::json!({"response": in_progress}),
    );
    push_event(
        &mut body,
        &mut sequence,
        "response.in_progress",
        serde_json::json!({"response": in_progress}),
    );
    push_event(
        &mut body,
        &mut sequence,
        "response.output_item.added",
        serde_json::json!({
            "output_index": output_index,
            "item": message_item(&item_id, "in_progress", None),
        }),
    );
    push_event(
        &mut body,
        &mut sequence,
        "response.content_part.added",
        serde_json::json!({
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "part": output_text_part(""),
        }),
    );
    push_event(
        &mut body,
        &mut sequence,
        "response.output_text.delta",
        serde_json::json!({
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "delta": text,
            "logprobs": [],
        }),
    );
    push_event(
        &mut body,
        &mut sequence,
        "response.output_text.done",
        serde_json::json!({
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "text": text,
        }),
    );
    push_event(
        &mut body,
        &mut sequence,
        "response.content_part.done",
        serde_json::json!({
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "part": output_text_part(text),
        }),
    );
    push_event(
        &mut body,
        &mut sequence,
        "response.output_item.done",
        serde_json::json!({
            "output_index": output_index,
            "item": message_item(&item_id, "completed", Some(text)),
        }),
    );
    push_event(
        &mut body,
        &mut sequence,
        "response.completed",
        serde_json::json!({"response": completed}),
    );
    body
}

#[post("/v1/responses")]
pub async fn create_response(
    req: HttpRequest,
    body: Result<web::Bytes, actix_web::Error>,
    args: web::Data<Arc<Args>>,
    llm: web::Data<Arc<llm::LLM>>,
) -> HttpResponse {
    let authorization = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok());

    let body = match body {
        Ok(body) => body,
        Err(err) => return extractor_error(err),
    };

    let request = match parse_and_authorize(&body, &args.api_key, authorization) {
        Ok(request) => request,
        Err((status, message)) => return error_json(status, message),
    };

    let loaded_model = loaded_model_identifier(&args);
    let (system, user) = match validate(&request, &loaded_model) {
        Ok(prompt) => prompt,
        Err((status, message)) => return error_json(status, message),
    };

    // `RD-25` reasoning support is added with the request field; this commit
    // only threads the parameter through the generation path.
    match llm.run_prompt_usage(system, user, &llm::Reasoning::default()) {
        Ok((text, usage)) if request.stream == Some(true) => HttpResponse::Ok()
            .content_type("text/event-stream")
            .body(build_stream_body(
                &loaded_model,
                &text,
                request.metadata.as_ref(),
                &usage,
            )),
        Ok((text, usage)) => HttpResponse::Ok().json(build_response(
            &loaded_model,
            &text,
            request.metadata.as_ref(),
            &usage,
        )),
        Err(err) => {
            let status = match err.downcast_ref::<llm::LLMError>() {
                Some(llm::LLMError::Busy) => 503,
                _ => 500,
            };
            eprintln!("responses error: {:#}", err);
            error_json(status, format!("{:#}", err))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::TokenUsage;
    use clap::Parser;

    fn parse(json: &str) -> CreateRequest {
        serde_json::from_str(json).expect("request should parse")
    }

    #[test]
    fn model_identity_uses_named_builtin_model() {
        let args = Args::parse_from(["ltengine", "--model", "gemma3-1b"]);
        assert_eq!(loaded_model_identifier(&args), "gemma3-1b");
    }

    #[test]
    fn model_identity_uses_model_file_override() {
        let args = Args::parse_from(["ltengine", "--model-file", "/models/custom.gguf"]);
        let loaded = loaded_model_identifier(&args);
        assert_eq!(loaded, "/models/custom.gguf");
        // The unrelated `--model` default must be rejected, the override accepted.
        let default_request = parse(r#"{"model":"gemma3-4b","input":"hi"}"#);
        assert!(validate(&default_request, &loaded).is_err());
        let override_request = parse(r#"{"model":"/models/custom.gguf","input":"hi"}"#);
        assert!(validate(&override_request, &loaded).is_ok());
    }

    #[test]
    fn rejects_unknown_field_on_message_and_content_part() {
        // Nested shapes must reject unknown fields, not silently ignore them.
        assert!(
            serde_json::from_str::<CreateRequest>(
                r#"{"input":[{"role":"user","content":"hi","extra":1}]}"#
            )
            .is_err()
        );
        // A content part carries its undeclared fields, and the mapping check
        // rejects them, so the request still fails instead of ignoring them.
        let part_request = parse(
            r#"{"input":[{"role":"user","content":[{"type":"input_text","text":"hi","extra":1}]}]}"#,
        );
        assert!(validate(&part_request, "gemma3-4b").is_err());
        // Accepted text and message inputs still parse.
        assert!(serde_json::from_str::<CreateRequest>(r#"{"input":"hi"}"#).is_ok());
        assert!(
            serde_json::from_str::<CreateRequest>(r#"{"input":[{"role":"user","content":"hi"}]}"#)
                .is_ok()
        );
        assert!(
            serde_json::from_str::<CreateRequest>(
                r#"{"input":[{"role":"user","content":[{"type":"input_text","text":"hi"}]}]}"#
            )
            .is_ok()
        );
    }

    async fn probe_body(body: Result<web::Bytes, actix_web::Error>) -> HttpResponse {
        match body {
            Ok(_) => HttpResponse::Ok().finish(),
            Err(err) => extractor_error(err),
        }
    }

    #[actix_web::test]
    async fn oversized_body_returns_nested_error() {
        let app = actix_web::test::init_service(
            actix_web::App::new()
                .app_data(web::PayloadConfig::new(16))
                .route("/probe", web::post().to(probe_body)),
        )
        .await;

        let req = actix_web::test::TestRequest::post()
            .uri("/probe")
            .set_payload("x".repeat(64))
            .to_request();
        let resp = actix_web::test::call_service(&app, req).await;

        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let body: serde_json::Value = actix_web::test::read_body_json(resp).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], serde_json::Value::Null);
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("invalid request body")
        );
    }

    #[test]
    fn maps_text_input() {
        let (system, user) =
            map_input(Some("be terse"), &ResponseInput::Text("hi".into())).unwrap();
        assert_eq!(system, "be terse");
        assert_eq!(user, "hi");
    }

    #[test]
    fn maps_message_array_with_standard_text_roles() {
        let input: ResponseInput = serde_json::from_str(
            r#"[{"role":"system","content":"be nice"},
                {"role":"user","content":"hi"},
                {"role":"assistant","content":[{"type":"output_text","text":"hello"}]},
                {"role":"user","content":"bye"}]"#,
        )
        .unwrap();
        let (system, user) = map_input(Some("extra"), &input).unwrap();
        assert_eq!(system, "extra\nbe nice");
        assert_eq!(user, "hi\nhello\nbye");
    }

    #[test]
    fn rejects_unsupported_role_and_modality() {
        let bad_role: ResponseInput =
            serde_json::from_str(r#"[{"role":"tool","content":"x"}]"#).unwrap();
        assert!(map_input(None, &bad_role).is_err());

        // A native modality payload parses and then fails with a clear
        // unsupported-modality error, not a generic unknown-field error.
        for json in [
            r#"[{"role":"user","content":[{"type":"input_image","image_url":"x"}]}]"#,
            r#"[{"role":"user","content":[{"type":"input_image","image_url":"x","detail":"high"}]}]"#,
            r#"[{"role":"user","content":[{"type":"input_file","file_id":"file-1","filename":"a.txt"}]}]"#,
            r#"[{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"x","format":"wav"}}]}]"#,
        ] {
            let input: ResponseInput = serde_json::from_str(json).unwrap();
            let err = map_input(None, &input).unwrap_err();
            assert!(err.contains("unsupported modality"), "{json} -> {err}");
        }

        // An extra field is rejected, never silently ignored. That holds for a
        // native field on a text part and for a field of no known part type.
        for json in [
            r#"[{"role":"user","content":[{"type":"input_text","text":"hi","image_url":"x"}]}]"#,
            r#"[{"role":"user","content":[{"type":"input_text","text":"hi","bogus":1}]}]"#,
        ] {
            let input: ResponseInput = serde_json::from_str(json).unwrap();
            let err = map_input(None, &input).unwrap_err();
            assert!(
                err.contains("unsupported modality field"),
                "{json} -> {err}"
            );
        }
    }

    #[test]
    fn body_api_key_is_400_before_the_auth_check() {
        // RD-4: the body field wins over a missing or a wrong Authorization.
        for authorization in [None, Some("Bearer wrong"), Some("Bearer secret")] {
            let (status, message) = parse_and_authorize(
                br#"{"input":"hi","api_key":"secret"}"#,
                "secret",
                authorization,
            )
            .unwrap_err();
            assert_eq!(status, 400);
            assert!(message.contains("api_key"), "{message}");
        }

        // Without the body field the credential check still runs.
        let (status, message) =
            parse_and_authorize(br#"{"input":"hi"}"#, "secret", None).unwrap_err();
        assert_eq!(status, 401);
        assert!(message.contains("bearer"), "{message}");
        assert!(parse_and_authorize(br#"{"input":"hi"}"#, "secret", Some("Bearer secret")).is_ok());
    }

    #[test]
    fn accepts_stream_true_and_selects_the_sse_body() {
        // PH-3 replaced the PH-1 rejection of `stream: true` with SSE output.
        let request = parse(r#"{"model":"gemma3-4b","input":"hi","stream":true}"#);
        assert!(validate(&request, "gemma3-4b").is_ok());
        assert!(!build_stream_body("gemma3-4b", "hi", None, &TokenUsage::default()).is_empty());
    }

    #[test]
    fn rejects_model_mismatch() {
        let request = parse(r#"{"model":"gemma3-1b","input":"hi"}"#);
        let (status, message) = validate(&request, "gemma3-4b").unwrap_err();
        assert_eq!(status, 400);
        assert!(message.contains("model"));

        let ok = parse(r#"{"model":"gemma3-4b","input":"hi","stream":false}"#);
        assert!(validate(&ok, "gemma3-4b").is_ok());
    }

    #[test]
    fn accepts_absent_null_and_object_metadata() {
        assert_eq!(parse(r#"{"input":"hi"}"#).metadata, None);
        assert_eq!(parse(r#"{"input":"hi","metadata":null}"#).metadata, None);
        let request = parse(r#"{"input":"hi","metadata":{"a":"b"}}"#);
        assert_eq!(request.metadata.unwrap()["a"], "b");
    }

    #[test]
    fn metadata_boundaries_are_inclusive() {
        // 64 key characters, 512 value characters, and 16 entries are accepted.
        let key = "k".repeat(METADATA_MAX_KEY_CHARS);
        let value = "v".repeat(METADATA_MAX_VALUE_CHARS);
        let json = format!(r#"{{"input":"hi","metadata":{{"{key}":"{value}"}}}}"#);
        assert!(serde_json::from_str::<CreateRequest>(&json).is_ok());

        assert!(
            serde_json::from_str::<CreateRequest>(&metadata_json(METADATA_MAX_ENTRIES)).is_ok()
        );
    }

    /// One `metadata` object with `count` entries, each inside the limits.
    fn metadata_json(count: usize) -> String {
        let entries: Vec<String> = (0..count).map(|i| format!(r#""k{i}":"v""#)).collect();
        format!(r#"{{"input":"hi","metadata":{{{}}}}}"#, entries.join(","))
    }

    #[test]
    fn rejects_invalid_metadata_naming_metadata() {
        let long_key = "k".repeat(METADATA_MAX_KEY_CHARS + 1);
        let long_value = "v".repeat(METADATA_MAX_VALUE_CHARS + 1);
        let cases = [
            r#"{"input":"hi","metadata":"x"}"#.to_string(),
            r#"{"input":"hi","metadata":5}"#.to_string(),
            r#"{"input":"hi","metadata":[]}"#.to_string(),
            r#"{"input":"hi","metadata":{"a":1}}"#.to_string(),
            r#"{"input":"hi","metadata":{"a":{"b":"c"}}}"#.to_string(),
            format!(r#"{{"input":"hi","metadata":{{"{long_key}":"v"}}}}"#),
            format!(r#"{{"input":"hi","metadata":{{"a":"{long_value}"}}}}"#),
            metadata_json(METADATA_MAX_ENTRIES + 1),
        ];
        for json in cases {
            let err = serde_json::from_str::<CreateRequest>(&json)
                .expect_err("invalid metadata must fail");
            assert!(err.to_string().contains("metadata"), "{json} -> {err}");
        }

        // The handler path turns the parse failure into the OpenAI-shaped 400.
        let (status, message) =
            parse_and_authorize(br#"{"input":"hi","metadata":{"a":1}}"#, "", None).unwrap_err();
        assert_eq!(status, 400);
        assert!(message.contains("metadata"), "{message}");
    }

    #[test]
    fn rejects_tools_and_other_unsupported_fields() {
        for json in [
            r#"{"input":"hi","tools":[]}"#,
            r#"{"input":"hi","tool_choice":"none"}"#,
            r#"{"input":"hi","parallel_tool_calls":false}"#,
        ] {
            assert!(
                serde_json::from_str::<CreateRequest>(json).is_err(),
                "{json}"
            );
        }
    }

    #[test]
    fn requires_bearer_only_when_key_configured() {
        assert!(check_auth("", None).is_ok());
        assert!(check_auth("secret", Some("Bearer secret")).is_ok());
        assert!(check_auth("secret", Some("Bearer wrong")).is_err());
        assert!(check_auth("secret", None).is_err());
        // The body field `api_key` is not a credential source on this route.
        assert!(parse_and_authorize(br#"{"input":"hi","api_key":"secret"}"#, "", None).is_err());
    }

    #[test]
    fn response_shape_has_message_output_and_usage() {
        let usage = TokenUsage {
            input_tokens: 7,
            output_tokens: 3,
        };
        let body = build_response("gemma3-4b", "hello", None, &usage);
        assert_eq!(body["object"], "response");
        assert_eq!(body["status"], "completed");
        assert_eq!(body["model"], "gemma3-4b");
        assert_eq!(body["output"][0]["type"], "message");
        assert_eq!(body["output"][0]["content"][0]["type"], "output_text");
        assert_eq!(body["output"][0]["content"][0]["text"], "hello");
        assert!(body["id"].as_str().unwrap().starts_with("resp_"));
        // PH-3: the exact RD-18 counts, never an estimate.
        assert_eq!(body["usage"]["input_tokens"], 7);
        assert_eq!(body["usage"]["output_tokens"], 3);
        assert_eq!(body["usage"]["total_tokens"], 10);
        // The required fields of the text-only route are always present.
        assert_eq!(body["parallel_tool_calls"], false);
        assert_eq!(body["tool_choice"], "none");
        assert_eq!(body["tools"].as_array().map(Vec::len), Some(0));
    }

    /// Parse an SSE body into its JSON payloads. Each frame must carry a `data:`
    /// line whose payload `type` equals the `event:` line (RD-17).
    fn parse_sse(body: &str) -> Vec<serde_json::Value> {
        body.split("\n\n")
            .filter(|frame| !frame.trim().is_empty())
            .map(|frame| {
                let name = frame
                    .lines()
                    .find_map(|line| line.strip_prefix("event: "))
                    .expect("frame has an event line");
                let data = frame
                    .lines()
                    .find_map(|line| line.strip_prefix("data: "))
                    .expect("frame has a data line");
                let payload: serde_json::Value =
                    serde_json::from_str(data).expect("payload is JSON");
                assert_eq!(payload["type"], name, "payload type matches the event name");
                payload
            })
            .collect()
    }

    #[test]
    fn stream_body_follows_the_rd17_order_with_increasing_sequence_numbers() {
        let usage = TokenUsage {
            input_tokens: 4,
            output_tokens: 2,
        };
        let body = build_stream_body("gemma3-4b", "hello", None, &usage);
        let events = parse_sse(&body);
        let types: Vec<&str> = events
            .iter()
            .map(|event| event["type"].as_str().unwrap())
            .collect();
        assert_eq!(
            types,
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        for (index, event) in events.iter().enumerate() {
            assert_eq!(event["sequence_number"], index);
        }

        // The delta, the part, the item, and the completed response agree.
        let item_id = events[4]["item_id"].as_str().unwrap();
        assert_eq!(events[4]["delta"], "hello");
        assert_eq!(events[5]["text"], "hello");
        assert_eq!(events[6]["part"]["text"], "hello");
        assert_eq!(events[7]["item"]["id"], item_id);
        assert_eq!(events[3]["part"]["type"], "output_text");
        assert_eq!(events[2]["item"]["status"], "in_progress");
        assert_eq!(events[8]["response"]["status"], "completed");
        assert_eq!(events[8]["response"]["model"], "gemma3-4b");
        assert_eq!(events[8]["response"]["output"][0]["id"], item_id);
        assert_eq!(events[8]["response"]["usage"]["input_tokens"], 4);
        assert_eq!(events[8]["response"]["usage"]["output_tokens"], 2);
        assert_eq!(events[8]["response"]["usage"]["total_tokens"], 6);
        // The stream carries no failure event: PH-3 fails before the first
        // event instead (RD-13).
        assert!(!types.contains(&"response.failed"));
        assert!(!types.contains(&"response.incomplete"));
    }

    #[test]
    fn stream_body_sets_the_content_type_and_echoes_metadata() {
        let metadata = serde_json::json!({"a": "b"});
        let usage = TokenUsage {
            input_tokens: 1,
            output_tokens: 1,
        };
        let body = build_stream_body("gemma3-4b", "x", Some(&metadata), &usage);
        let events = parse_sse(&body);
        assert_eq!(
            events[0]["response"]["metadata"],
            serde_json::json!({"a": "b"})
        );
        assert_eq!(
            events[8]["response"]["metadata"],
            serde_json::json!({"a": "b"})
        );
        assert_eq!(events[0]["response"]["status"], "in_progress");
        assert!(events[0]["response"]["usage"].is_null());
        // The two SSE frames share one response identifier.
        assert_eq!(events[0]["response"]["id"], events[8]["response"]["id"]);
    }

    #[test]
    fn response_echoes_metadata_or_null() {
        let usage = TokenUsage::default();
        let metadata = serde_json::json!({"a": "b"});
        let echoed = build_response("gemma3-4b", "x", Some(&metadata), &usage);
        assert_eq!(echoed["metadata"], serde_json::json!({"a": "b"}));
        let absent = build_response("gemma3-4b", "x", None, &usage);
        assert!(absent["metadata"].is_null());
    }

    #[test]
    fn response_ids_are_unique_and_prefixed() {
        let usage = TokenUsage::default();
        let first = build_response("gemma3-4b", "a", None, &usage);
        let second = build_response("gemma3-4b", "b", None, &usage);
        let first_id = first["id"].as_str().unwrap();
        let second_id = second["id"].as_str().unwrap();
        assert!(first_id.starts_with("resp_"));
        assert!(second_id.starts_with("resp_"));
        assert_ne!(first_id, second_id);
    }
}
