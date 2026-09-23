//! Tests for the `POST /v1/responses` route (PH-1 through PH-5).


use super::*;
use crate::llm::TokenUsage;
use crate::responses_shape::{
    ResponseControls, StreamOutput, build_response_with_conversation,
    build_stream_body_with_conversation, calls_output, message_output, replay_stream_body,
};

/// The pre-`PH-6a` body shape: no conversation echo.
fn build_response(
    model: &str,
    output: serde_json::Value,
    echo: &ToolEcho,
    metadata: Option<&serde_json::Value>,
    usage: &TokenUsage,
) -> serde_json::Value {
    build_response_with_conversation(model, output, echo, metadata, None, &ResponseControls::default(), usage)
}

/// The pre-`PH-6a` stream shape: no conversation echo.
fn build_stream_body(
    model: &str,
    output: StreamOutput<'_>,
    echo: &ToolEcho,
    metadata: Option<&serde_json::Value>,
    usage: &TokenUsage,
) -> (String, serde_json::Value) {
    build_stream_body_with_conversation(model, output, echo, metadata, None, &ResponseControls::default(), usage)
}
use crate::responses_store::{
    ConversationRecord, FailingStore, FileStore, ResponseStore, StoredResponse, tests::TempStoreDir,
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
    assert!(!build_stream_body("gemma3-4b", StreamOutput::Text("hi"), &ToolEcho::text_only(), None, &TokenUsage::default()).0.is_empty());
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
    // A field the route does not implement still fails with a 400. `PH-8a`
    // removed the generation parameters from this set.
    assert!(serde_json::from_str::<CreateRequest>(r#"{"input":"hi","bogus":1}"#).is_err());
}

/// `PH8-01`, changed behavior: `temperature` is accepted in 0.0–2.0 and
/// rejected outside it, naming the field.
#[test]
fn ph8_01_temperature_range() {
    for json in [
        r#"{"input":"hi","temperature":0.0}"#,
        r#"{"input":"hi","temperature":2.0}"#,
    ] {
        let request = parse(json);
        assert!(validate(&request, "gemma3-4b").is_ok(), "{json}");
    }
    for json in [
        r#"{"input":"hi","temperature":-0.1}"#,
        r#"{"input":"hi","temperature":2.1}"#,
        r#"{"input":"hi","temperature":"warm"}"#,
    ] {
        let request = parse(json);
        let (status, message) = validate(&request, "gemma3-4b").unwrap_err();
        assert_eq!(status, 400, "{json}");
        assert!(message.contains("temperature"), "{json}: {message}");
    }
}

/// `PH8-02`, changed behavior: `top_p` is accepted in 0.0–1.0 and rejected
/// outside it, naming the field.
#[test]
fn ph8_02_top_p_range() {
    for json in [
        r#"{"input":"hi","top_p":0.0}"#,
        r#"{"input":"hi","top_p":1.0}"#,
    ] {
        let request = parse(json);
        assert!(validate(&request, "gemma3-4b").is_ok(), "{json}");
    }
    for json in [
        r#"{"input":"hi","top_p":-0.1}"#,
        r#"{"input":"hi","top_p":1.1}"#,
        r#"{"input":"hi","top_p":"wide"}"#,
    ] {
        let request = parse(json);
        let (status, message) = validate(&request, "gemma3-4b").unwrap_err();
        assert_eq!(status, 400, "{json}");
        assert!(message.contains("top_p"), "{json}: {message}");
    }
}

/// `PH8-03`, changed behavior: `max_output_tokens` accepts `1`; `0`, a
/// negative value, a float, and a non-number are rejected, naming the field.
#[test]
fn ph8_03_max_output_tokens_range() {
    let request = parse(r#"{"input":"hi","max_output_tokens":1}"#);
    assert!(validate(&request, "gemma3-4b").is_ok());
    for json in [
        r#"{"input":"hi","max_output_tokens":0}"#,
        r#"{"input":"hi","max_output_tokens":-1}"#,
        r#"{"input":"hi","max_output_tokens":1.5}"#,
        r#"{"input":"hi","max_output_tokens":"ten"}"#,
    ] {
        let request = parse(json);
        let (status, message) = validate(&request, "gemma3-4b").unwrap_err();
        assert_eq!(status, 400, "{json}");
        assert!(message.contains("max_output_tokens"), "{json}: {message}");
    }
}

/// `PH8-06`, preserved behavior: a request without the new fields is accepted,
/// and a `null` control is the same as an absent one.
#[test]
fn ph8_06_absent_controls_keep_defaults() {
    for json in [
        r#"{"input":"hi"}"#,
        r#"{"input":"hi","temperature":null,"top_p":null,"max_output_tokens":null}"#,
    ] {
        let request = parse(json);
        let prepared = validate(&request, "gemma3-4b").expect("absent controls are valid");
        assert_eq!(prepared.generation.temperature, 0.0, "{json}");
        assert_eq!(prepared.generation.top_p, 0.95, "{json}");
        assert_eq!(prepared.generation.max_output_tokens, None, "{json}");
    }
}

/// `PH8-07`, changed behavior: an accepted `temperature` and `top_p` reach the
/// generation value that `create_sampler` receives.
#[test]
fn ph8_07_generation_controls_reach_the_sampler() {
    let request = parse(r#"{"input":"hi","temperature":0.7,"top_p":0.3}"#);
    let prepared = validate(&request, "gemma3-4b").expect("valid controls");
    assert_eq!(prepared.generation.temperature, 0.7);
    assert_eq!(prepared.generation.top_p, 0.3);
}

/// `PH8-08`, changed behavior: the response echoes each generation control only
/// when the request carried it.
#[test]
fn ph8_08_generation_echo_follows_the_request() {
    let usage = TokenUsage::default();
    let carried = ResponseControls {
        temperature: Some(1.5),
        top_p: Some(0.25),
        max_output_tokens: Some(7),
        verbosity: None,
    };
    let body = build_response_with_conversation(
        "gemma3-4b",
        message_output("x"),
        &ToolEcho::text_only(),
        None,
        None,
        &carried,
        &usage,
    );
    assert_eq!(body["temperature"], 1.5);
    assert_eq!(body["top_p"], 0.25);
    assert_eq!(body["max_output_tokens"], 7);

    let absent = build_response_with_conversation(
        "gemma3-4b",
        message_output("x"),
        &ToolEcho::text_only(),
        None,
        None,
        &ResponseControls::default(),
        &usage,
    );
    assert!(absent.get("temperature").is_none());
    assert!(absent.get("top_p").is_none());
    assert!(absent.get("max_output_tokens").is_none());
}

/// `PH8-09`, changed behavior: `low`, `medium`, and `high` are accepted; any
/// other value, and a non-string, are rejected naming `text.verbosity`.
#[test]
fn ph8_09_verbosity_values() {
    for verbosity in ["low", "medium", "high"] {
        let json = format!(r#"{{"input":"hi","text":{{"verbosity":"{verbosity}"}}}}"#);
        let request = parse(&json);
        assert!(validate(&request, "gemma3-4b").is_ok(), "{verbosity}");
    }
    for json in [
        r#"{"input":"hi","text":{"verbosity":"loud"}}"#,
        r#"{"input":"hi","text":{"verbosity":3}}"#,
    ] {
        let request = parse(json);
        let (status, message) = validate(&request, "gemma3-4b").unwrap_err();
        assert_eq!(status, 400, "{json}");
        assert!(message.contains("text.verbosity"), "{json}: {message}");
    }
}

/// `PH8-10`, changed behavior: `low` and `high` append their directive to the
/// system text; `medium`, absent, and `null` append nothing.
#[test]
fn ph8_10_verbosity_directive() {
    let low = parse(r#"{"input":"hi","text":{"verbosity":"low"}}"#);
    let system = validate(&low, "gemma3-4b").expect("low accepted").system;
    assert!(system.contains("Be concise."), "{system}");

    let high = parse(r#"{"input":"hi","text":{"verbosity":"high"}}"#);
    let system = validate(&high, "gemma3-4b").expect("high accepted").system;
    assert!(system.contains("Be thorough."), "{system}");

    for json in [
        r#"{"input":"hi"}"#,
        r#"{"input":"hi","text":{"verbosity":"medium"}}"#,
        r#"{"input":"hi","text":{"verbosity":null}}"#,
    ] {
        let request = parse(json);
        let system = validate(&request, "gemma3-4b").expect("accepted").system;
        assert!(
            !system.contains("Be concise.") && !system.contains("Be thorough."),
            "{json}: {system}"
        );
    }
}

/// `PH8-11`, preserved behavior: an absent `text.verbosity` adds no prompt text.
#[test]
fn ph8_11_absent_verbosity_keeps_the_prompt() {
    let request = parse(r#"{"input":"hi"}"#);
    let (expected, _) = map_input(None, &request.input).expect("input maps");
    let system = validate(&request, "gemma3-4b").expect("accepted").system;
    assert_eq!(system, expected);
}

/// `PH8-12`, changed behavior: the response echoes the effective verbosity only
/// when the request carried it.
#[test]
fn ph8_12_verbosity_echo_follows_the_request() {
    let usage = TokenUsage::default();
    let carried = ResponseControls {
        verbosity: Some("high"),
        ..ResponseControls::default()
    };
    let body = build_response_with_conversation(
        "gemma3-4b",
        message_output("x"),
        &ToolEcho::text_only(),
        None,
        None,
        &carried,
        &usage,
    );
    assert_eq!(body["text"]["verbosity"], "high");

    let absent = build_response_with_conversation(
        "gemma3-4b",
        message_output("x"),
        &ToolEcho::text_only(),
        None,
        None,
        &ResponseControls::default(),
        &usage,
    );
    assert!(absent.get("text").is_none());
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
fn accepts_text_verbosity_naming_the_field() {
    // PH8-09 changed PH4A-07: `low` is accepted, and an unknown value names
    // `text.verbosity`.
    let accepted = parse(r#"{"input":"hi","text":{"verbosity":"low"}}"#);
    assert!(validate(&accepted, "gemma3-4b").is_ok());
    let rejected = parse(r#"{"input":"hi","text":{"verbosity":"loud"}}"#);
    let (status, message) = validate(&rejected, "gemma3-4b").unwrap_err();
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
    let (body, _) = build_stream_body("gemma3-4b", StreamOutput::Text("hello"), &ToolEcho::text_only(), None, &usage);
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
    let (body, _) = build_stream_body("gemma3-4b", StreamOutput::Text("x"), &ToolEcho::text_only(), Some(&metadata), &usage);
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

#[test]
fn previous_response_id_prepends_context() {
    // PH5-07.
    let previous = StoredResponse {
        response: serde_json::json!({
            "id": "resp_prev",
            "output": [
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hello"}]},
            ],
        }),
        input_items: vec![serde_json::json!({"role": "user", "content": "first"})],
    };
    let request = parse(
        r#"{"input":"second","instructions":"new system","previous_response_id":"resp_prev"}"#,
    );
    let prepared = validate(&request, "gemma3-4b").expect("valid");
    let prompt = effective_prompt(&request, &prepared, Some(&previous), None).expect("prompt");
    // The referenced `instructions` are not carried; the request's are.
    assert_eq!(prompt.system, "new system");
    assert_eq!(prompt.user, "first\nhello\nsecond");
    assert_eq!(prompt.input_items.len(), 3);
    assert_eq!(prompt.input_items[0]["content"], "first");
    assert_eq!(prompt.input_items[1]["content"][0]["text"], "hello");
    assert_eq!(prompt.input_items[2]["content"][0]["text"], "second");
}

#[test]
fn previous_response_id_is_transitive() {
    // PH5-07: the stored effective input already carries earlier turns.
    let previous = StoredResponse {
        response: serde_json::json!({"id": "resp_2", "output": []}),
        input_items: vec![
            serde_json::json!({"role": "user", "content": "one"}),
            serde_json::json!({"role": "assistant", "content": "two"}),
        ],
    };
    let request = parse(r#"{"input":"three","previous_response_id":"resp_2"}"#);
    let prepared = validate(&request, "gemma3-4b").expect("valid");
    let prompt = effective_prompt(&request, &prepared, Some(&previous), None).expect("prompt");
    assert_eq!(prompt.user, "one\ntwo\nthree");
}

#[test]
fn previous_response_id_absent_reuses_the_plain_prompt() {
    // PH5-13: the no-previous path preserves the PH-1 through PH-4b behavior.
    let request = parse(r#"{"input":"hi","instructions":"be terse"}"#);
    let prepared = validate(&request, "gemma3-4b").expect("valid");
    let prompt = effective_prompt(&request, &prepared, None, None).expect("prompt");
    assert_eq!(prompt.system, prepared.system);
    assert_eq!(prompt.user, prepared.user);
    assert_eq!(prompt.input_items.len(), 1);
}

#[test]
fn unknown_previous_response_id_is_not_found() {
    // PH5-08.
    let temp = TempStoreDir::new();
    let store = FileStore::new(temp.path.clone()).expect("store dir");
    let response = resolve_previous(&store, Some("resp_missing"), 0).expect_err("must 404");
    assert_eq!(response.status(), actix_web::http::StatusCode::NOT_FOUND);

    let record = StoredResponse {
        response: serde_json::json!({"id": "resp_1"}),
        input_items: Vec::new(),
    };
    store.put("resp_1", &record).expect("put");
    assert_eq!(
        resolve_previous(&store, Some("resp_1"), 0).expect("resolved"),
        Some(record)
    );
    assert_eq!(resolve_previous(&store, None, 0).expect("none"), None);
}

#[test]
fn failed_store_write_is_a_500() {
    // PH5-09 (unit part): the write is fail-closed.
    let record = StoredResponse {
        response: serde_json::json!({"id": "resp_1"}),
        input_items: Vec::new(),
    };
    let (status, message) = persist(&FailingStore, "resp_1", &record).expect_err("must fail");
    assert_eq!(status, 500);
    assert!(message.contains("store"), "{message}");
}

#[test]
fn store_flag_controls_persistence() {
    // PH5-01 and PH5-02 (store part).
    let temp = TempStoreDir::new();
    let store = FileStore::new(temp.path.clone()).expect("store dir");
    maybe_store(&store, true, &serde_json::json!({"id": "resp_aaaa"}), &[])
        .expect("store");
    assert!(store.get("resp_aaaa").expect("get").is_some());

    maybe_store(&store, false, &serde_json::json!({"id": "resp_bbbb"}), &[])
        .expect("skip");
    assert!(store.get("resp_bbbb").expect("get").is_none());
}

#[test]
fn replay_matches_the_create_stream() {
    // PH5A-01: replay regenerates the same event list, for a message and for a
    // function call.
    let usage = TokenUsage {
        input_tokens: 4,
        output_tokens: 2,
    };
    let (body, completed) = build_stream_body(
        "gemma3-4b",
        StreamOutput::Text("hello"),
        &ToolEcho::text_only(),
        None,
        &usage,
    );
    let replay = replay_stream_body(&completed, None).expect("replay");
    assert_eq!(parse_sse(&replay), parse_sse(&body));

    let calls = vec![ModelCall {
        name: "f".to_string(),
        arguments: "{}".to_string(),
    }];
    let (call_body, call_completed) = build_stream_body(
        "gemma3-4b",
        StreamOutput::Calls(&calls),
        &ToolEcho::text_only(),
        None,
        &usage,
    );
    let replay = replay_stream_body(&call_completed, None).expect("replay");
    assert_eq!(parse_sse(&replay), parse_sse(&call_body));
}

#[test]
fn replay_starting_after_filters() {
    // PH5A-02 (unit part): the original sequence numbers are kept.
    let usage = TokenUsage::default();
    let (_, completed) = build_stream_body(
        "gemma3-4b",
        StreamOutput::Text("hello"),
        &ToolEcho::text_only(),
        None,
        &usage,
    );
    let full = parse_sse(&replay_stream_body(&completed, None).expect("replay"));
    let tail = parse_sse(&replay_stream_body(&completed, Some(4)).expect("replay"));
    assert_eq!(tail, full[5..]);
    assert_eq!(tail[0]["sequence_number"], 5);
}

#[test]
fn replay_of_a_non_stream_response() {
    // PH5A-01: a response created without `stream` replays too.
    let usage = TokenUsage::default();
    let completed = build_response(
        "gemma3-4b",
        message_output("hello"),
        &ToolEcho::text_only(),
        None,
        &usage,
    );
    let events = parse_sse(&replay_stream_body(&completed, None).expect("replay"));
    assert_eq!(events.len(), 9);
    assert_eq!(events[8]["response"], completed);
}

#[test]
fn replay_is_deterministic() {
    // PH5A-08: replay is a pure function of the stored record.
    let usage = TokenUsage::default();
    let (_, completed) = build_stream_body(
        "gemma3-4b",
        StreamOutput::Text("hello"),
        &ToolEcho::text_only(),
        None,
        &usage,
    );
    let before = completed.clone();
    let first = replay_stream_body(&completed, None).expect("replay");
    let second = replay_stream_body(&completed, None).expect("replay");
    assert_eq!(first, second);
    assert_eq!(completed, before);
}

#[test]
fn conversation_prepends_and_appends() {
    // PH6A-09 (prompt part).
    let conversation = ConversationRecord {
        id: "conv_1".to_string(),
        created_at: 1,
        metadata: serde_json::Value::Null,
        items: vec![
            serde_json::json!({"id": "msg_1", "role": "user", "content": "first"}),
            serde_json::json!({
                "id": "msg_2",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello", "annotations": []}],
            }),
        ],
    };
    let request = parse(r#"{"input":"second","instructions":"sys","conversation":"conv_1"}"#);
    let prepared = validate(&request, "gemma3-4b").expect("valid");
    let prompt = effective_prompt(&request, &prepared, None, Some(&conversation)).expect("prompt");
    assert_eq!(prompt.system, "sys");
    assert_eq!(prompt.user, "first\nhello\nsecond");
    // Only the request's own items are stored on the response record.
    assert_eq!(prompt.input_items.len(), 1);
    assert_eq!(prompt.input_items[0]["content"][0]["text"], "second");
}

#[test]
fn conversation_with_previous_response_id_is_a_400() {
    // PH6A-10.
    let request =
        parse(r#"{"input":"hi","conversation":"conv_1","previous_response_id":"resp_1"}"#);
    let (status, message) = validate(&request, "gemma3-4b").unwrap_err();
    assert_eq!(status, 400);
    assert!(message.contains("conversation"), "{message}");
}

#[test]
fn conversation_limit_is_checked_before_generation() {
    // PH6A-11: the bound uses the declared tool count.
    let tools = validate(
        &parse(
            r#"{"input":"hi","tools":[{"type":"function","name":"a","parameters":{"type":"object"}},{"type":"function","name":"b","parameters":{"type":"object"}}],"tool_choice":"auto","parallel_tool_calls":true}"#,
        ),
        "gemma3-4b",
    )
    .expect("valid")
    .tools;
    assert_eq!(max_output_items(&tools), 2);
    let text_only = validate(&parse(r#"{"input":"hi"}"#), "gemma3-4b")
        .expect("valid")
        .tools;
    assert_eq!(max_output_items(&text_only), 1);
}

#[test]
fn background_requires_store() {
    // PH6B-08.
    let request = parse(r#"{"input":"hi","background":true,"store":false}"#);
    let (status, message) = validate(&request, "gemma3-4b").unwrap_err();
    assert_eq!(status, 400);
    assert!(message.contains("background"), "{message}");
}

#[test]
fn background_rejects_stream() {
    // PH6B-09.
    let request = parse(r#"{"input":"hi","background":true,"stream":true}"#);
    let (status, message) = validate(&request, "gemma3-4b").unwrap_err();
    assert_eq!(status, 400);
    assert!(message.contains("background"), "{message}");
}
