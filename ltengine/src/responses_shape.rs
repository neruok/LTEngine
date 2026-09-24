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

/// One `reasoning` output item (`RD-33`, `RD-40` row 9). Under the `verbatim`
/// disposition, `content` carries the trace as one part of `part_type` and
/// `summary` is `[]`. `encrypted_content` is `null` in every profile.
pub(crate) fn reasoning_item(trace: &str, part_type: &str) -> Value {
    json!({
        "id": new_id("rs_"),
        "type": "reasoning",
        "status": "completed",
        "summary": [],
        "content": [{"type": part_type, "text": trace}],
        "encrypted_content": null,
    })
}

/// The `in_progress` form of a reasoning item, for the `added` event.
fn reasoning_item_in_progress(item_id: &str) -> Value {
    json!({
        "id": item_id,
        "type": "reasoning",
        "status": "in_progress",
        "summary": [],
        "content": [],
        "encrypted_content": null,
    })
}

/// Prepend a reasoning item to an output array, so it appears before the
/// message item (`RD-33`).
pub(crate) fn prepend_reasoning(output: Value, reasoning: Option<&Value>) -> Value {
    let Some(item) = reasoning else {
        return output;
    };
    let mut items = vec![item.clone()];
    if let Some(rest) = output.as_array() {
        items.extend(rest.iter().cloned());
    }
    Value::Array(items)
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
/// exact counts. `output_tokens_details.reasoning_tokens` is the `RD-38` subset
/// of `output_tokens`. No value is an estimate (`SP-NEVER-010`).
pub(crate) fn usage_json(usage: &llm::TokenUsage) -> Value {
    json!({
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens(),
        "output_tokens_details": {
            "reasoning_tokens": usage.reasoning_tokens,
        },
    })
}

/// The request-derived values that the response echoes beyond the tool echo
/// (`RD-29` 3.4, `RD-30` 4.4).
///
/// A field is present only when the request carried it, so a request without
/// the new fields keeps the pre-`PH-8a` body (`INVARIANT PH8-1`).
#[derive(Clone, Debug, Default)]
pub struct ResponseControls {
    /// `temperature` echo. `None` omits the key.
    pub temperature: Option<f64>,
    /// `top_p` echo. `None` omits the key.
    pub top_p: Option<f64>,
    /// `max_output_tokens` echo. `None` omits the key.
    pub max_output_tokens: Option<u32>,
    /// The effective `text.verbosity`. `None` omits the `text` key.
    pub verbosity: Option<&'static str>,
}

/// The OpenAI-shaped response object, shared by the non-streaming body and the
/// `response.completed` and `response.in_progress` events. The `conversation`
/// key is added only when the request named a conversation, so a request that
/// carries none keeps the pre-`PH-6a` shape (`PH6A-14`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_response_object(
    id: &str,
    created_at: u64,
    status: &str,
    model: &str,
    metadata: Option<&Value>,
    conversation: Option<&str>,
    echo: &ToolEcho,
    controls: &ResponseControls,
    output: Value,
    usage: Value,
) -> Value {
    let mut object = json!({
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
    });
    if let Some(id) = conversation {
        object["conversation"] = json!({"id": id});
    }
    if let Some(temperature) = controls.temperature {
        object["temperature"] = json!(temperature);
    }
    if let Some(top_p) = controls.top_p {
        object["top_p"] = json!(top_p);
    }
    if let Some(max_output_tokens) = controls.max_output_tokens {
        object["max_output_tokens"] = json!(max_output_tokens);
    }
    if let Some(verbosity) = controls.verbosity {
        object["text"] = json!({"verbosity": verbosity});
    }
    object
}

/// OpenAI-shaped non-streaming response (`PH-1`, `PH-3`, `PH-4b`).
///
/// `output` is the item array that the handler built. `usage` carries the exact
/// counts of the one generation (`RD-18`). `echo` carries the truthful tool
/// values of the request (`RD-20`). `metadata` echoes the accepted request
/// value, or `null` when the request carried none. `conversation` adds the
/// `PH-6a` echo only when the request named a conversation (`PH6A-14`).
pub(crate) fn build_response_with_conversation(
    model: &str,
    output: Value,
    echo: &ToolEcho,
    metadata: Option<&Value>,
    conversation: Option<&str>,
    controls: &ResponseControls,
    usage: &llm::TokenUsage,
) -> Value {
    build_response_object(
        &new_id("resp_"),
        now_secs(),
        "completed",
        model,
        metadata,
        conversation,
        echo,
        controls,
        output,
        usage_json(usage),
    )
}

/// The generated turn as the streaming builder needs it.
pub(crate) enum StreamOutput<'a> {
    Text(&'a str),
    Calls(&'a [ModelCall]),
}

/// Append one SSE event to the ordered event list. The payload carries its
/// event `type` and the next `sequence_number` (RD-17).
fn push_event(
    events: &mut Vec<(String, Value)>,
    sequence: &mut u32,
    event_type: &str,
    mut payload: Value,
) {
    let map = payload
        .as_object_mut()
        .expect("an event payload is a JSON object");
    map.insert("type".to_string(), Value::from(event_type));
    map.insert("sequence_number".to_string(), Value::from(*sequence));
    *sequence += 1;
    events.push((event_type.to_string(), payload));
}

/// The final output items of a streamed turn. The identifiers are created once,
/// so the per-item events and the completed body carry the same ones.
fn stream_items(output: StreamOutput<'_>) -> Vec<Value> {
    match output {
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
    }
}

/// The ordered `RD-17` event list of one response.
///
/// `in_progress` and `completed` are the two response objects that the stream
/// carries. `items` are the final output items. The per-item events derive from
/// `items`, so the list is the same whether the response was streamed at create
/// time or replayed later (`PH-5a`).
fn build_events(in_progress: Value, completed: Value, items: &[Value]) -> Vec<(String, Value)> {
    let mut sequence = 0_u32;
    let mut events: Vec<(String, Value)> = Vec::new();
    push_event(
        &mut events,
        &mut sequence,
        "response.created",
        json!({"response": in_progress.clone()}),
    );
    push_event(
        &mut events,
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
                    &mut events,
                    &mut sequence,
                    "response.output_item.added",
                    json!({
                        "output_index": output_index,
                        "item": message_item(&item_id, "in_progress", None),
                    }),
                );
                push_event(
                    &mut events,
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
                    &mut events,
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
                    &mut events,
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
                    &mut events,
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
                    &mut events,
                    &mut sequence,
                    "response.output_item.done",
                    json!({"output_index": output_index, "item": item}),
                );
            }
            Some("reasoning") => {
                let item_id = item["id"].as_str().unwrap_or_default().to_string();
                let text = item["content"][0]["text"].as_str().unwrap_or_default();
                push_event(
                    &mut events,
                    &mut sequence,
                    "response.output_item.added",
                    json!({
                        "output_index": output_index,
                        "item": reasoning_item_in_progress(&item_id),
                    }),
                );
                push_event(
                    &mut events,
                    &mut sequence,
                    "response.reasoning_text.delta",
                    json!({
                        "item_id": item_id,
                        "output_index": output_index,
                        "content_index": 0,
                        "delta": text,
                    }),
                );
                push_event(
                    &mut events,
                    &mut sequence,
                    "response.reasoning_text.done",
                    json!({
                        "item_id": item_id,
                        "output_index": output_index,
                        "content_index": 0,
                        "text": text,
                    }),
                );
                push_event(
                    &mut events,
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
                    &mut events,
                    &mut sequence,
                    "response.output_item.added",
                    json!({
                        "output_index": output_index,
                        "item": function_call_item(&item_id, call_id, name, "", "in_progress"),
                    }),
                );
                push_event(
                    &mut events,
                    &mut sequence,
                    "response.function_call_arguments.delta",
                    json!({
                        "item_id": item_id,
                        "output_index": output_index,
                        "delta": arguments,
                    }),
                );
                push_event(
                    &mut events,
                    &mut sequence,
                    "response.function_call_arguments.done",
                    json!({
                        "item_id": item_id,
                        "output_index": output_index,
                        "arguments": arguments,
                    }),
                );
                push_event(
                    &mut events,
                    &mut sequence,
                    "response.output_item.done",
                    json!({"output_index": output_index, "item": item}),
                );
            }
            _ => {}
        }
    }

    push_event(
        &mut events,
        &mut sequence,
        "response.completed",
        json!({"response": completed}),
    );
    events
}

/// Render an event list as the SSE body. `starting_after` keeps the original
/// `sequence_number` values and drops only the events up to and including it
/// (`RD-27`).
fn render_events(events: &[(String, Value)], starting_after: Option<u32>) -> String {
    let mut body = String::new();
    for (event_type, payload) in events {
        if let Some(start) = starting_after {
            let sequence = payload["sequence_number"].as_u64().unwrap_or(u64::MAX);
            if sequence <= u64::from(start) {
                continue;
            }
        }
        body.push_str("event: ");
        body.push_str(event_type);
        body.push_str("\ndata: ");
        body.push_str(&payload.to_string());
        body.push_str("\n\n");
    }
    body
}

/// Build the SSE body of a successful `stream: true` request (RD-17, RD-20).
///
/// Returns the body text and the completed response object, so the caller can
/// store the response (`PH-5`, RD-22) with the same identifier that the stream
/// carries.
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
pub(crate) fn build_stream_body_with_conversation(
    model: &str,
    output: StreamOutput<'_>,
    reasoning: Option<&Value>,
    echo: &ToolEcho,
    metadata: Option<&Value>,
    conversation: Option<&str>,
    controls: &ResponseControls,
    usage: &llm::TokenUsage,
) -> (String, Value) {
    let response_id = new_id("resp_");
    let created_at = now_secs();
    // The reasoning item, when present, leads the output array (`RD-33`).
    let items = prepend_reasoning(Value::Array(stream_items(output)), reasoning)
        .as_array()
        .cloned()
        .unwrap_or_default();
    let in_progress = build_response_object(
        &response_id,
        created_at,
        "in_progress",
        model,
        metadata,
        conversation,
        echo,
        controls,
        json!([]),
        Value::Null,
    );
    let completed = build_response_object(
        &response_id,
        created_at,
        "completed",
        model,
        metadata,
        conversation,
        echo,
        controls,
        Value::Array(items.clone()),
        usage_json(usage),
    );
    let events = build_events(in_progress, completed.clone(), &items);
    (render_events(&events, None), completed)
}

/// Replay the SSE sequence of a stored response (`PH-5a`, `RD-27`).
///
/// Regenerates the event list from the stored response object. It calls no
/// model and reruns no generation. `starting_after` filters by
/// `sequence_number`. Returns `None` when the stored object cannot be replayed,
/// because its `id`, `created_at`, or `output` is missing.
pub(crate) fn replay_stream_body(response: &Value, starting_after: Option<u32>) -> Option<String> {
    let items = response["output"].as_array()?.clone();
    response["id"].as_str()?;
    response["created_at"].as_u64()?;
    let mut in_progress = response.clone();
    if let Some(map) = in_progress.as_object_mut() {
        map.insert("status".to_string(), Value::from("in_progress"));
        map.insert("output".to_string(), json!([]));
        map.insert("usage".to_string(), Value::Null);
    }
    let events = build_events(in_progress, response.clone(), &items);
    Some(render_events(&events, starting_after))
}
