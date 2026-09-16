//! Non-streaming `POST /v1/responses` (roadmap PH-1).
//!
//! The route is a non-streaming text adapter over `LLM::run_prompt`. It is not
//! an OpenAI proxy: no upstream call, no second decode path. Only this route
//! uses the OpenAI-shaped nested error body; the existing LibreTranslate routes
//! keep their flat `{"error": "<string>"}` body.

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
fn validate(
    request: &CreateRequest,
    loaded_model: &str,
) -> Result<(String, String), (u16, String)> {
    if request.stream == Some(true) {
        return Err((
            400,
            "streaming is not supported: use stream false".to_string(),
        ));
    }
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

/// OpenAI-shaped non-streaming response. `usage` is omitted on purpose: real
/// token usage is PH-3 work and fake usage is prohibited (SP-NEVER-010).
/// `parallel_tool_calls`, `tool_choice`, and `tools` are the truthful values of
/// the text-only route: it offers no tool and calls no tool. `metadata` echoes
/// the accepted request value, or `null` when the request carried none.
fn build_response(
    model: &str,
    text: &str,
    metadata: Option<&serde_json::Value>,
) -> serde_json::Value {
    serde_json::json!({
        "id": new_id("resp_"),
        "object": "response",
        "created_at": now_secs(),
        "status": "completed",
        "model": model,
        "metadata": metadata,
        "parallel_tool_calls": false,
        "tool_choice": "none",
        "tools": [],
        "output": [{
            "id": new_id("msg_"),
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": text,
                "annotations": [],
            }],
        }],
    })
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

    match llm.run_prompt(system, user) {
        Ok(text) => HttpResponse::Ok().json(build_response(
            &loaded_model,
            &text,
            request.metadata.as_ref(),
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
    fn rejects_stream_true() {
        let request = parse(r#"{"model":"gemma3-4b","input":"hi","stream":true}"#);
        let (status, message) = validate(&request, "gemma3-4b").unwrap_err();
        assert_eq!(status, 400);
        assert!(message.contains("stream"));
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
    fn response_shape_has_message_output_and_no_usage() {
        let body = build_response("gemma3-4b", "hello", None);
        assert_eq!(body["object"], "response");
        assert_eq!(body["status"], "completed");
        assert_eq!(body["model"], "gemma3-4b");
        assert_eq!(body["output"][0]["type"], "message");
        assert_eq!(body["output"][0]["content"][0]["type"], "output_text");
        assert_eq!(body["output"][0]["content"][0]["text"], "hello");
        assert!(body["id"].as_str().unwrap().starts_with("resp_"));
        assert!(body.get("usage").is_none());
        // The required fields of the text-only route are always present.
        assert_eq!(body["parallel_tool_calls"], false);
        assert_eq!(body["tool_choice"], "none");
        assert_eq!(body["tools"].as_array().map(Vec::len), Some(0));
    }

    #[test]
    fn response_echoes_metadata_or_null() {
        let metadata = serde_json::json!({"a": "b"});
        let echoed = build_response("gemma3-4b", "x", Some(&metadata));
        assert_eq!(echoed["metadata"], serde_json::json!({"a": "b"}));
        let absent = build_response("gemma3-4b", "x", None);
        assert!(absent["metadata"].is_null());
    }

    #[test]
    fn response_ids_are_unique_and_prefixed() {
        let first = build_response("gemma3-4b", "a", None);
        let second = build_response("gemma3-4b", "b", None);
        let first_id = first["id"].as_str().unwrap();
        let second_id = second["id"].as_str().unwrap();
        assert!(first_id.starts_with("resp_"));
        assert!(second_id.starts_with("resp_"));
        assert_ne!(first_id, second_id);
    }
}
