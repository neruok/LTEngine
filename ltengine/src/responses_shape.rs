//! OpenAI-shaped response object and SSE event construction for
//! `POST /v1/responses` (PH-1, PH-3, PH-4b, RD-17, RD-20).
//!
//! This module owns the response shape: the non-streaming body and the SSE
//! event order for a text message and for function calls. The handler in
//! `responses.rs` owns request parsing and one generation.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

use crate::llm;
use crate::responses_tools::{ModelCall, ToolEcho};

/// Process-wide counter that makes `resp_*`, `msg_*`, `fc_*`, and `call_*`
/// identifiers unique within one run (SP-MUST-010).
static NEXT_ID: AtomicU64 = AtomicU64::new(0);

pub(crate) fn new_id(prefix: &str) -> String {
    let count = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{prefix}{nanos:x}{count:x}")
}

pub(crate) fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// One `output_text` content part.
fn output_text_part(text: &str) -> Value {
    json!({"type": "output_text", "text": text, "annotations": []})
}

/// One assistant `message` output item. `text` is `None` while the item is in
/// progress, because its content part arrives in a later event.
fn message_item(item_id: &str, status: &str, text: Option<&str>) -> Value {
    let content = match text {
        Some(text) => json!([output_text_part(text)]),
        None => json!([]),
    };
    json!({
        "id": item_id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": content,
    })
}

/// One `function_call` output item (RD-20).
fn function_call_item(
    item_id: &str,
    call_id: &str,
    name: &str,
    arguments: &str,
    status: &str,
) -> Value {
    json!({
        "id": item_id,
        "type": "function_call",
        "status": status,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    })
}

/// The `output` array of a text message.
pub(crate) fn message_output(text: &str) -> Value {
    json!([message_item(&new_id("msg_"), "completed", Some(text))])
}

/// The `output` array of one or more function calls.
pub(crate) fn calls_output(calls: &[ModelCall]) -> Value {
    Value::Array(
        calls
            .iter()
            .map(|call| {
                function_call_item(
                    &new_id("fc_"),
                    &new_id("call_"),
                    &call.name,
                    &call.arguments,
                    "completed",
                )
            })
            .collect(),
    )
}

/// The exact `usage` object of RD-18. `total_tokens` is the exact sum of the two
/// exact counts. No value is an estimate (`SP-NEVER-010`).
fn usage_json(usage: &llm::TokenUsage) -> Value {
    json!({
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
    metadata: Option<&Value>,
    echo: &ToolEcho,
    output: Value,
    usage: Value,
) -> Value {
    json!({
        "id": id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "model": model,
        "metadata": metadata,
        "parallel_tool_calls": echo.parallel_tool_calls,
        "tool_choice": echo.tool_choice,
        "tools": echo.tools,
        "output": output,
        "usage": usage,
    })
}

/// OpenAI-shaped non-streaming response. `output` is the item array that the
/// handler built. `usage` carries the exact counts of the one generation
/// (PH-3, RD-18). `echo` carries the truthful tool values of the request
/// (`PH-4b`, RD-20). `metadata` echoes the accepted request value, or `null`
/// when the request carried none.
pub(crate) fn build_response(
    model: &str,
    output: Value,
    echo: &ToolEcho,
    metadata: Option<&Value>,
    usage: &llm::TokenUsage,
) -> Value {
    response_object(
        &new_id("resp_"),
        now_secs(),
        "completed",
        model,
        metadata,
        echo,
        output,
        usage_json(usage),
    )
}

/// The generated turn as the streaming builder needs it.
pub(crate) enum StreamOutput<'a> {
    Text(&'a str),
    Calls(&'a [ModelCall]),
}

/// Append one SSE frame. The frame carries the OpenAI event name line and the
/// JSON payload line. The payload carries its event `type` and the next
/// `sequence_number` (RD-17).
fn push_event(body: &mut String, sequence: &mut u32, event_type: &str, mut payload: Value) {
    let map = payload
        .as_object_mut()
        .expect("an event payload is a JSON object");
    map.insert("type".to_string(), Value::from(event_type));
    map.insert("sequence_number".to_string(), Value::from(*sequence));
    *sequence += 1;
    body.push_str("event: ");
    body.push_str(event_type);
    body.push_str("\ndata: ");
    body.push_str(&payload.to_string());
    body.push_str("\n\n");
}

/// Build the SSE body of a successful `stream: true` request (RD-17, RD-20).
///
/// A text message keeps the `RD-17` order: `response.created`,
/// `response.in_progress`, `response.output_item.added`,
/// `response.content_part.added`, zero or more `response.output_text.delta`,
/// `response.output_text.done`, `response.content_part.done`,
/// `response.output_item.done`, and `response.completed`.
///
/// A function call uses the OpenAI tool order: `response.output_item.added`,
/// zero or more `response.function_call_arguments.delta`,
/// `response.function_call_arguments.done`, and `response.output_item.done`.
/// Multiple calls repeat that group with an increasing `output_index`.
///
/// `PH-3` and `PH-4b` make one generation attempt and emit no `response.failed`,
/// because every failure happens before the first event (RD-13).
///
/// `ponytail:` the body is built after the single generation attempt, so a text
/// message carries one delta with the whole text. RD-17 permits zero or more
/// delta events, and `clean_output` can retract text at the end, so the whole
/// text is the only safe delta.
pub(crate) fn build_stream_body(
    model: &str,
    output: StreamOutput<'_>,
    echo: &ToolEcho,
    metadata: Option<&Value>,
    usage: &llm::TokenUsage,
) -> String {
    let response_id = new_id("resp_");
    let created_at = now_secs();
    // Build the final items once so the per-item events and the completed body
    // carry the same identifiers.
    let items: Vec<Value> = match output {
        StreamOutput::Text(text) => vec![message_item(&new_id("msg_"), "completed", Some(text))],
        StreamOutput::Calls(calls) => calls
            .iter()
            .map(|call| {
                function_call_item(
                    &new_id("fc_"),
                    &new_id("call_"),
                    &call.name,
                    &call.arguments,
                    "completed",
                )
            })
            .collect(),
    };
    let in_progress = response_object(
        &response_id,
        created_at,
        "in_progress",
        model,
        metadata,
        echo,
        json!([]),
        Value::Null,
    );
    let completed = response_object(
        &response_id,
        created_at,
        "completed",
        model,
        metadata,
        echo,
        Value::Array(items.clone()),
        usage_json(usage),
    );

    let mut sequence = 0_u32;
    let mut body = String::new();
    push_event(
        &mut body,
        &mut sequence,
        "response.created",
        json!({"response": in_progress}),
    );
    push_event(
        &mut body,
        &mut sequence,
        "response.in_progress",
        json!({"response": in_progress}),
    );

    for (output_index, item) in items.iter().enumerate() {
        match item["type"].as_str() {
            Some("message") => {
                let item_id = item["id"].as_str().unwrap_or_default().to_string();
                let text = item["content"][0]["text"].as_str().unwrap_or_default();
                push_event(
                    &mut body,
                    &mut sequence,
                    "response.output_item.added",
                    json!({
                        "output_index": output_index,
                        "item": message_item(&item_id, "in_progress", None),
                    }),
                );
                push_event(
                    &mut body,
                    &mut sequence,
                    "response.content_part.added",
                    json!({
                        "item_id": item_id,
                        "output_index": output_index,
                        "content_index": 0,
                        "part": output_text_part(""),
                    }),
                );
                push_event(
                    &mut body,
                    &mut sequence,
                    "response.output_text.delta",
                    json!({
                        "item_id": item_id,
                        "output_index": output_index,
                        "content_index": 0,
                        "delta": text,
                        "logprobs": [],
                    }),
                );
                push_event(
                    &mut body,
                    &mut sequence,
                    "response.output_text.done",
                    json!({
                        "item_id": item_id,
                        "output_index": output_index,
                        "content_index": 0,
                        "text": text,
                    }),
                );
                push_event(
                    &mut body,
                    &mut sequence,
                    "response.content_part.done",
                    json!({
                        "item_id": item_id,
                        "output_index": output_index,
                        "content_index": 0,
                        "part": output_text_part(text),
                    }),
                );
                push_event(
                    &mut body,
                    &mut sequence,
                    "response.output_item.done",
                    json!({"output_index": output_index, "item": item}),
                );
            }
            Some("function_call") => {
                let item_id = item["id"].as_str().unwrap_or_default().to_string();
                let call_id = item["call_id"].as_str().unwrap_or_default();
                let name = item["name"].as_str().unwrap_or_default();
                let arguments = item["arguments"].as_str().unwrap_or_default();
                push_event(
                    &mut body,
                    &mut sequence,
                    "response.output_item.added",
                    json!({
                        "output_index": output_index,
                        "item": function_call_item(&item_id, call_id, name, "", "in_progress"),
                    }),
                );
                push_event(
                    &mut body,
                    &mut sequence,
                    "response.function_call_arguments.delta",
                    json!({
                        "item_id": item_id,
                        "output_index": output_index,
                        "delta": arguments,
                    }),
                );
                push_event(
                    &mut body,
                    &mut sequence,
                    "response.function_call_arguments.done",
                    json!({
                        "item_id": item_id,
                        "output_index": output_index,
                        "arguments": arguments,
                    }),
                );
                push_event(
                    &mut body,
                    &mut sequence,
                    "response.output_item.done",
                    json!({"output_index": output_index, "item": item}),
                );
            }
            _ => {}
        }
    }

    push_event(
        &mut body,
        &mut sequence,
        "response.completed",
        json!({"response": completed}),
    );
    body
}
