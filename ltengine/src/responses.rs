//! `POST /v1/responses` (roadmap PH-1, PH-3, PH-4a).
//!
//! The route is a text adapter over `LLM::run_prompt_usage_grammar`. It is not an
//! OpenAI proxy: no upstream call, no second decode path. Only this route uses
//! the OpenAI-shaped nested error body; the existing LibreTranslate routes keep
//! their flat `{"error": "<string>"}` body.
//!
//! `stream: true` answers with `text/event-stream` (PH-3, RD-17). The body is one
//! generation attempt and carries the exact token usage of RD-18.
//!
//! `text.format` follows the OpenAI contract (PH-4a, RD-19). The request types and
//! the schema-to-grammar derivation live in `responses_schema.rs`; the response
//! object and the SSE events live in `responses_shape.rs`.

use std::sync::Arc;

use actix_web::{HttpRequest, HttpResponse, http::StatusCode, http::header, post, web};
use serde::Deserialize;

use crate::Args;
use crate::llm;
use crate::responses_schema::{derive_format, ResponseTextConfig, StructuredFormat};
use crate::responses_shape::{
    StreamOutput, build_response, build_stream_body, calls_output, message_output,
};
use crate::responses_tools::{ModelTurn, ToolEcho, ToolRequest, parse_tools, parse_turn};
use crate::responses_input::{ResponseInput, map_input};

/// Request body of `POST /v1/responses`.
///
/// `deny_unknown_fields` rejects an unsupported field with a 400 instead of
/// silently ignoring it. The OpenAI tool fields `tools`, `tool_choice`, and
/// `parallel_tool_calls` are validated by `parse_tools`; a tool request that
/// the route cannot honor returns a clear error. `metadata` is the one carried
/// field: its OpenAI limits are validated, and an accepted value is echoed.
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
    #[serde(default, deserialize_with = "crate::responses_metadata::deserialize_metadata")]
    pub metadata: Option<serde_json::Value>,
    /// OpenAI structured-output configuration (PH-4a). `text.format` selects the
    /// response format; `derive_format` validates it and derives the grammar.
    #[serde(default)]
    pub text: Option<ResponseTextConfig>,
    /// The LibreTranslate-style body credential. This route does not accept it
    /// (RD-4); `parse_and_authorize` rejects it with a clear 400.
    #[serde(default)]
    pub api_key: Option<serde_json::Value>,
    /// OpenAI tool definitions (PH-4b). A `function` tool is accepted and
    /// offered to the model through the transcription envelope. A tool of any
    /// other type returns a clear 400 that names `tools`. An empty array is the
    /// OpenAI default and calls no tool. `parse_tools` validates the entries.
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    /// OpenAI tool choice. `none`, `auto`, `required`, and
    /// `{"type":"function","name":"<declared>"}` are accepted (PH-4b). Any
    /// other value returns a clear 400 that names `tool_choice`.
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    /// OpenAI parallel-call switch. `true` permits more than one
    /// `function_call` item; `false` permits at most one (PH-4b,
    /// `SP-PLANNED-005`).
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
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

/// A validated request: the prompt pair, the decoded `text.format` effect, and
/// the accepted tool request.
#[derive(Debug)]
struct PreparedPrompt {
    system: String,
    user: String,
    format: StructuredFormat,
    tools: ToolRequest,
}

/// Validate every supported-statefulness and model choice before generation.
///
/// `stream` is not validated here: `stream: true` selects the SSE response of
/// `build_stream_body` (RD-17). Every rejection below happens before the first
/// event, so it uses the normal OpenAI-shaped HTTP error (RD-13).
fn validate(request: &CreateRequest, loaded_model: &str) -> Result<PreparedPrompt, (u16, String)> {
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
    let tools = parse_tools(
        request.tools.as_ref(),
        request.tool_choice.as_ref(),
        request.parallel_tool_calls,
    )
    .map_err(|message| (400, message))?;
    let format = derive_format(request.text.as_ref()).map_err(|message| (400, message))?;
    if tools.offers_tools() && format.json_required {
        // Combining the tool envelope with a structured output format needs two
        // grammars. The route does not implement the combination, so it
        // returns a clear error instead of mishandling it (`SP-NEVER-010`).
        return Err((
            400,
            "`tools` and a structured `text.format` cannot be combined on this route".to_string(),
        ));
    }
    let (mut system, user) =
        map_input(request.instructions.as_deref(), &request.input).map_err(|err| (400, err))?;
    if let Some(note) = tools.system_note() {
        if !system.is_empty() {
            system.push('\n');
        }
        system.push_str(&note);
    }
    Ok(PreparedPrompt {
        system,
        user,
        format,
        tools,
    })
}

/// `json_object` and `json_schema` require the generated text to parse as JSON
/// (RD-19). A `strict: true` schema is already guaranteed by the grammar; this
/// check covers `json_object` and non-strict `json_schema`. A non-conforming
/// output is never returned as a valid body (`SP-NEVER-010`).
fn require_json(text: &str) -> Result<(), String> {
    serde_json::from_str::<serde_json::Value>(text.trim())
        .map(|_| ())
        .map_err(|err| format!("the model did not produce valid JSON for `text.format`: {err}"))
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
    let prepared = match validate(&request, &loaded_model) {
        Ok(prepared) => prepared,
        Err((status, message)) => return error_json(status, message),
    };
    let PreparedPrompt {
        system,
        user,
        format,
        tools,
    } = prepared;

    let streaming = request.stream == Some(true);
    let grammar = if tools.offers_tools() {
        tools.grammar.as_deref()
    } else {
        format.grammar.as_deref()
    };

    match llm.run_prompt_usage_grammar(system, user, grammar, &crate::llm::Reasoning::default()) {
        Ok((text, usage)) => {
            let echo = ToolEcho::from_request(&tools);
            if tools.offers_tools() {
                // The model answers with the transcription envelope (RD-20).
                return match parse_turn(&text, &tools) {
                    Ok(ModelTurn::Message(answer)) => {
                        if streaming {
                            sse_body(&loaded_model, StreamOutput::Text(&answer), &echo, request.metadata.as_ref(), &usage)
                        } else {
                            HttpResponse::Ok().json(build_response(
                                &loaded_model,
                                message_output(&answer),
                                &echo,
                                request.metadata.as_ref(),
                                &usage,
                            ))
                        }
                    }
                    Ok(ModelTurn::Calls(calls)) => {
                        if streaming {
                            sse_body(&loaded_model, StreamOutput::Calls(&calls), &echo, request.metadata.as_ref(), &usage)
                        } else {
                            HttpResponse::Ok().json(build_response(
                                &loaded_model,
                                calls_output(&calls),
                                &echo,
                                request.metadata.as_ref(),
                                &usage,
                            ))
                        }
                    }
                    Err(message) => error_json(500, message),
                };
            }
            if format.json_required {
                if let Err(message) = require_json(&text) {
                    return error_json(500, message);
                }
            }
            if streaming {
                sse_body(&loaded_model, StreamOutput::Text(&text), &echo, request.metadata.as_ref(), &usage)
            } else {
                HttpResponse::Ok().json(build_response(
                    &loaded_model,
                    message_output(&text),
                    &echo,
                    request.metadata.as_ref(),
                    &usage,
                ))
            }
        }
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

/// The SSE success response for one generated turn (RD-17, RD-20).
fn sse_body(
    model: &str,
    output: StreamOutput<'_>,
    echo: &ToolEcho,
    metadata: Option<&serde_json::Value>,
    usage: &llm::TokenUsage,
) -> HttpResponse {
    HttpResponse::Ok()
        .content_type("text/event-stream")
        .body(build_stream_body(model, output, echo, metadata, usage))
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::TokenUsage;
    use crate::responses_shape::{
        StreamOutput, build_response, build_stream_body, calls_output, message_output,
    };
    use crate::responses_tools::{ModelCall, ModelTurn, ToolEcho, parse_turn};
    use crate::responses_metadata::{METADATA_MAX_ENTRIES, METADATA_MAX_KEY_CHARS, METADATA_MAX_VALUE_CHARS};
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
        assert!(!build_stream_body("gemma3-4b", StreamOutput::Text("hi"), &ToolEcho::text_only(), None, &TokenUsage::default()).is_empty());
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
    fn accepts_openai_tool_fields_without_a_tool_call() {
        // The response body reports `tools: []`, `tool_choice: "none"`, and
        // `parallel_tool_calls: false`, so the request must accept the same
        // no-tool forms instead of rejecting the fields with a 400.
        for json in [
            r#"{"input":"hi","tools":[]}"#,
            r#"{"input":"hi","tool_choice":"none"}"#,
            r#"{"input":"hi","tool_choice":"auto"}"#,
            r#"{"input":"hi","parallel_tool_calls":false}"#,
            r#"{"input":"hi","parallel_tool_calls":true}"#,
            r#"{"input":"hi","tools":[],"tool_choice":"auto","parallel_tool_calls":true}"#,
        ] {
            let request = parse(json);
            assert!(validate(&request, "gemma3-4b").is_ok(), "{json}");
        }
    }

    #[test]
    fn rejects_a_tool_request_naming_the_field() {
        // A tool request that the route cannot honor fails with a clear 400
        // that names the field, never a silent ignore (SP-MUST-011, PH4B-04).
        for (json, field) in [
            (r#"{"input":"hi","tools":[{"type":"web_search"}]}"#, "tools"),
            (r#"{"input":"hi","tools":[{"type":"function"}]}"#, "tools"),
            (
                r#"{"input":"hi","tools":[{"type":"function","name":"f"}]}"#,
                "tools",
            ),
            (r#"{"input":"hi","tool_choice":"required"}"#, "tool_choice"),
            (r#"{"input":"hi","tool_choice":"bogus"}"#, "tool_choice"),
            (
                r#"{"input":"hi","tool_choice":{"type":"function","name":"f"}}"#,
                "tool_choice",
            ),
        ] {
            let request = parse(json);
            let (status, message) = validate(&request, "gemma3-4b").unwrap_err();
            assert_eq!(status, 400, "{json}");
            assert!(message.contains(field), "{json} -> {message}");
        }
    }

    #[test]
    fn accepts_a_function_tool_and_echoes_the_request() {
        // PH4B-07 (unit part): the body echoes the accepted tool request.
        let request = parse(
            r#"{"input":"hi","tools":[{"type":"function","name":"get_weather","description":"w","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"],"additionalProperties":false},"strict":true}],"tool_choice":"auto","parallel_tool_calls":true}"#,
        );
        let prepared = validate(&request, "gemma3-4b").expect("valid");
        assert!(prepared.tools.offers_tools());
        assert!(prepared.system.contains("get_weather"));
        let echo = ToolEcho::from_request(&prepared.tools);
        assert!(echo.parallel_tool_calls);
        assert_eq!(echo.tool_choice, serde_json::json!("auto"));
        assert_eq!(echo.tools[0]["name"], "get_weather");
        assert_eq!(echo.tools[0]["strict"], true);
    }

    #[test]
    fn named_tool_choice_requires_a_declared_tool() {
        // PH4B-09 (unit part).
        let request = parse(
            r#"{"input":"hi","tools":[{"type":"function","name":"a","parameters":{"type":"object"}}],"tool_choice":{"type":"function","name":"b"}}"#,
        );
        let (status, message) = validate(&request, "gemma3-4b").unwrap_err();
        assert_eq!(status, 400);
        assert!(message.contains("tool_choice"), "{message}");
    }

    #[test]
    fn tools_and_structured_format_cannot_be_combined() {
        let request = parse(
            r#"{"input":"hi","tools":[{"type":"function","name":"a","parameters":{"type":"object"}}],"text":{"format":{"type":"json_object"}}}"#,
        );
        let (status, message) = validate(&request, "gemma3-4b").unwrap_err();
        assert_eq!(status, 400);
        assert!(message.contains("text.format"), "{message}");
    }

    #[test]
    fn parse_turn_decodes_calls_and_messages() {
        // PH4B-01 and PH4B-05 (unit part).
        let request = parse(
            r#"{"input":"hi","tools":[{"type":"function","name":"get_weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"],"additionalProperties":false}}],"tool_choice":"auto"}"#,
        );
        let tools = validate(&request, "gemma3-4b").expect("valid").tools;
        let turn = parse_turn(
            r#"{"calls":[{"name":"get_weather","arguments":{"location":"Paris"}}]}"#,
            &tools,
        )
        .expect("turn");
        assert_eq!(
            turn,
            ModelTurn::Calls(vec![ModelCall {
                name: "get_weather".to_string(),
                arguments: r#"{"location":"Paris"}"#.to_string(),
            }])
        );
        assert_eq!(
            parse_turn(r#"{"message":"hi"}"#, &tools).expect("turn"),
            ModelTurn::Message("hi".to_string())
        );
        // An undeclared name is rejected.
        assert!(parse_turn(r#"{"calls":[{"name":"nope","arguments":{}}]}"#, &tools).is_err());
    }

    #[test]
    fn parse_turn_enforces_the_parallel_tool_call_limit() {
        // PH4B-05 (unit part): the route caps the call count when
        // `parallel_tool_calls` is false and permits more than one when it is
        // true. The envelope grammar enforces the same cap.
        let two = r#"[{"type":"function","name":"a","parameters":{"type":"object"}},{"type":"function","name":"b","parameters":{"type":"object"}}]"#;
        let parallel = validate(
            &parse(&format!(
                r#"{{"input":"hi","tools":{two},"tool_choice":"required","parallel_tool_calls":true}}"#
            )),
            "gemma3-4b",
        )
        .expect("valid")
        .tools;
        let calls = r#"{"calls":[{"name":"a","arguments":{}},{"name":"b","arguments":{}}]}"#;
        match parse_turn(calls, &parallel).expect("turn") {
            ModelTurn::Calls(calls) => assert_eq!(calls.len(), 2),
            other => panic!("expected calls, got {other:?}"),
        }

        let serial = validate(
            &parse(&format!(
                r#"{{"input":"hi","tools":{two},"tool_choice":"required","parallel_tool_calls":false}}"#
            )),
            "gemma3-4b",
        )
        .expect("valid")
        .tools;
        assert!(parse_turn(calls, &serial).is_err());
    }

    #[test]
    fn falls_back_to_non_strict_when_the_schema_cannot_form_a_grammar() {
        // PH4B-06 (unit part): an omitted `strict` that cannot convert falls
        // back to best effort, and the body echoes `strict: false`. An explicit
        // `strict: true` instead returns a 400.
        let lenient = parse(
            r#"{"input":"hi","tools":[{"type":"function","name":"a","parameters":{"type":"bogus-type"}}]}"#,
        );
        let prepared = validate(&lenient, "gemma3-4b").expect("valid");
        assert!(prepared.tools.offers_tools());
        assert!(prepared.tools.grammar.is_none());
        assert_eq!(ToolEcho::from_request(&prepared.tools).tools[0]["strict"], false);

        let strict = parse(
            r#"{"input":"hi","tools":[{"type":"function","name":"a","parameters":{"type":"bogus-type"},"strict":true}]}"#,
        );
        let (status, message) = validate(&strict, "gemma3-4b").unwrap_err();
        assert_eq!(status, 400);
        assert!(message.contains("tools"), "{message}");
    }

    #[test]
    fn calls_output_carries_the_function_call_fields() {
        // PH4B-01 (unit part).
        let calls = vec![ModelCall {
            name: "f".to_string(),
            arguments: "{}".to_string(),
        }];
        let output = calls_output(&calls);
        assert_eq!(output[0]["type"], "function_call");
        assert!(output[0]["id"].as_str().unwrap().starts_with("fc_"));
        assert!(output[0]["call_id"].as_str().unwrap().starts_with("call_"));
        assert_eq!(output[0]["name"], "f");
        assert_eq!(output[0]["arguments"], "{}");
        assert_eq!(output[0]["status"], "completed");
    }

    #[test]
    fn maps_function_call_and_output_input_items() {
        // PH-4b: a client can close a tool loop by sending the call and its
        // output back in one request. Both fold into the prompt text.
        let input: ResponseInput = serde_json::from_str(
            r#"[{"role":"user","content":"weather?"},
                {"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":"{\"location\":\"Paris\"}","status":"completed"},
                {"type":"function_call_output","call_id":"call_1","output":"sunny"}]"#,
        )
        .unwrap();
        let (system, user) = map_input(None, &input).unwrap();
        assert_eq!(system, "");
        assert!(user.contains("weather?"), "{user}");
        assert!(user.contains("get_weather"), "{user}");
        assert!(user.contains("sunny"), "{user}");
    }

    #[test]
    fn rejects_other_unknown_fields() {
        // A field the route does not implement still fails with a 400. That
        // includes the OpenAI generation parameters, which stay rejected until
        // a phase implements them and their behavior is defined.
        for json in [
            r#"{"input":"hi","bogus":1}"#,
            r#"{"input":"hi","temperature":0.5}"#,
        ] {
            assert!(
                serde_json::from_str::<CreateRequest>(json).is_err(),
                "{json}"
            );
        }
    }

    #[test]
    fn derives_structured_format_for_json_object() {
        // PH4A-01 (unit part): json_object requires JSON and constrains nothing.
        let request = parse(r#"{"input":"hi","text":{"format":{"type":"json_object"}}}"#);
        let prepared = validate(&request, "gemma3-4b").expect("valid");
        assert!(prepared.format.json_required);
        assert!(prepared.format.grammar.is_some());
    }

    #[test]
    fn derives_a_grammar_for_strict_json_schema() {
        // PH4A-02 (unit part): strict json_schema yields a grammar.
        let request = parse(
            r#"{"input":"hi","text":{"format":{"type":"json_schema","name":"answer","strict":true,"schema":{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}}}}"#,
        );
        let prepared = validate(&request, "gemma3-4b").expect("valid");
        assert!(prepared.format.json_required);
        assert!(prepared.format.grammar.is_some());
    }

    #[test]
    fn rejects_unknown_text_format_type_naming_the_field() {
        // PH4A-03.
        let request = parse(r#"{"input":"hi","text":{"format":{"type":"bogus"}}}"#);
        let (status, message) = validate(&request, "gemma3-4b").unwrap_err();
        assert_eq!(status, 400);
        assert!(message.contains("text.format.type"), "{message}");
    }

    #[test]
    fn rejects_json_schema_without_schema_naming_the_field() {
        // PH4A-04.
        let request = parse(
            r#"{"input":"hi","text":{"format":{"type":"json_schema","name":"answer"}}}"#,
        );
        let (status, message) = validate(&request, "gemma3-4b").unwrap_err();
        assert_eq!(status, 400);
        assert!(message.contains("text.format"), "{message}");
    }

    #[test]
    fn rejects_an_unsupported_schema_naming_the_field() {
        // PH4A-05. The pinned converter rejects an unknown `type` value.
        let request = parse(
            r#"{"input":"hi","text":{"format":{"type":"json_schema","name":"answer","strict":true,"schema":{"type":"bogus-type"}}}}"#,
        );
        let (status, message) = validate(&request, "gemma3-4b").unwrap_err();
        assert_eq!(status, 400);
        assert!(message.contains("text.format.schema"), "{message}");
    }

    #[test]
    fn rejects_text_verbosity_naming_the_field() {
        // PH4A-07.
        let request = parse(r#"{"input":"hi","text":{"verbosity":"low"}}"#);
        let (status, message) = validate(&request, "gemma3-4b").unwrap_err();
        assert_eq!(status, 400);
        assert!(message.contains("text.verbosity"), "{message}");
    }

    #[test]
    fn rejects_json_text_that_does_not_parse() {
        // PH4A-06 (unit part): the JSON gate rejects non-JSON text.
        assert!(require_json("not json").is_err());
        assert!(require_json(r#"{"a":1}"#).is_ok());
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
        let body = build_response("gemma3-4b", message_output("hello"), &ToolEcho::text_only(), None, &usage);
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
        let body = build_stream_body("gemma3-4b", StreamOutput::Text("hello"), &ToolEcho::text_only(), None, &usage);
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
        let body = build_stream_body("gemma3-4b", StreamOutput::Text("x"), &ToolEcho::text_only(), Some(&metadata), &usage);
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
        let echoed = build_response("gemma3-4b", message_output("x"), &ToolEcho::text_only(), Some(&metadata), &usage);
        assert_eq!(echoed["metadata"], serde_json::json!({"a": "b"}));
        let absent = build_response("gemma3-4b", message_output("x"), &ToolEcho::text_only(), None, &usage);
        assert!(absent["metadata"].is_null());
    }

    #[test]
    fn response_ids_are_unique_and_prefixed() {
        let usage = TokenUsage::default();
        let first = build_response("gemma3-4b", message_output("a"), &ToolEcho::text_only(), None, &usage);
        let second = build_response("gemma3-4b", message_output("b"), &ToolEcho::text_only(), None, &usage);
        let first_id = first["id"].as_str().unwrap();
        let second_id = second["id"].as_str().unwrap();
        assert!(first_id.starts_with("resp_"));
        assert!(second_id.starts_with("resp_"));
        assert_ne!(first_id, second_id);
    }
}
