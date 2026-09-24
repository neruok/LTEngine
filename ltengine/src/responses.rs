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
//!
//! `PH-5` stores the completed response (`store`, default `true`), extends the
//! input from `previous_response_id`, and serves retrieval, deletion, and
//! input-item listing from `responses_retrieve.rs`. The storage boundary is
//! `responses_store.rs` and the shared HTTP helpers are `responses_http.rs`.

use std::sync::Arc;

use actix_web::{HttpRequest, HttpResponse, post, web};
use serde::Deserialize;

use crate::Args;
use crate::llm;
use crate::responses_http::{bearer, check_auth, error_json, extractor_error, not_found};
use crate::responses_generation::{GenerationEcho, derive_generation};
use crate::responses_input::{
    ResponseInput, input_items, map_input, output_items_as_input, parse_input_items,
    stored_items_as_input,
};
use crate::responses_profile::{ReasoningTrace, ResponsesApi};
use crate::responses_reasoning::{ReasoningEffect, ResponseReasoning, derive_reasoning};
use crate::responses_schema::{ResponseTextConfig, StructuredFormat, Verbosity, derive_text};
use crate::responses_shape::{
    ResponseControls, StreamOutput, build_response_with_conversation,
    build_stream_body_with_conversation, calls_output, message_output, prepend_reasoning,
    reasoning_item,
};
use crate::responses_store::{AppStore, ConversationRecord, ResponseStore, StoredResponse};
use crate::responses_tools::{ModelTurn, ToolEcho, ToolRequest, parse_tools, parse_turn};

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
    /// OpenAI `store` (PH-5, RD-22). `true`, the default, persists the completed
    /// response. `false` does not persist it, so it is not retrievable.
    #[serde(default)]
    pub store: Option<bool>,
    /// OpenAI `previous_response_id` (PH-5, RD-22). The referenced record's
    /// input items and output items are prepended to this request's input.
    #[serde(default)]
    pub previous_response_id: Option<String>,
    /// OpenAI `conversation` (PH-6a, RD-23). A `conv_*` identifier. The
    /// conversation's items are prepended to this request's input, and the
    /// request's input items and the response's output items are appended to
    /// the conversation. It cannot be combined with `previous_response_id`.
    #[serde(default)]
    pub conversation: Option<String>,
    /// OpenAI reasoning configuration (`PH-7`, `RD-28`). `effort` is
    /// implemented and reaches the model's chat template. The other contract
    /// members are rejected by name (`SP-NEVER-010`).
    #[serde(default)]
    pub reasoning: Option<ResponseReasoning>,
    /// OpenAI `background` (PH-6b, RD-24). `true` stores the response with
    /// status `queued`, returns it, and runs one generation in the background.
    /// It requires `store` to be `true` and it cannot be combined with
    /// `stream: true`.
    #[serde(default)]
    pub background: Option<bool>,
    /// OpenAI `temperature` (PH-8a, `RD-29`). A number in 0.0–2.0. Absent or
    /// `null` keeps the greedy sampler.
    #[serde(default)]
    pub temperature: Option<serde_json::Value>,
    /// OpenAI `top_p` (PH-8a, `RD-29`). A number in 0.0–1.0. Absent or `null`
    /// keeps the current `0.95`.
    #[serde(default)]
    pub top_p: Option<serde_json::Value>,
    /// OpenAI `max_output_tokens` (PH-8a, `RD-29`). An integer of at least 1.
    /// Absent or `null` keeps the `3 × prompt` budget.
    #[serde(default)]
    pub max_output_tokens: Option<serde_json::Value>,
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
    /// The decoded `reasoning` request (`RD-28`).
    reasoning: ReasoningEffect,
    /// The decoded generation controls (`RD-29`).
    generation: llm::Generation,
    /// The generation values that the response echoes (`RD-29` 3.4).
    generation_echo: GenerationEcho,
    /// The effective `text.verbosity` (`RD-30`).
    verbosity: Verbosity,
    /// The directive to append, or `None` when the profile or the value adds
    /// none (`RD-30`, `RD-40` row 4).
    verbosity_directive: Option<&'static str>,
}

/// Validate every supported-statefulness and model choice before generation.
///
/// `stream` is not validated here: `stream: true` selects the SSE response of
/// `build_stream_body` (RD-17). Every rejection below happens before the first
/// event, so it uses the normal OpenAI-shaped HTTP error (RD-13).
fn validate_request(
    request: &CreateRequest,
    loaded_model: &str,
    profile: ResponsesApi,
) -> Result<PreparedPrompt, (u16, String)> {
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
    if request.conversation.is_some() && request.previous_response_id.is_some() {
        return Err((
            400,
            "`conversation` cannot be combined with `previous_response_id`".to_string(),
        ));
    }
    if request.background == Some(true) {
        if request.store == Some(false) {
            return Err((
                400,
                "`background` requires `store` to be `true`".to_string(),
            ));
        }
        if request.stream == Some(true) {
            return Err((
                400,
                "`background` cannot be combined with `stream: true`".to_string(),
            ));
        }
    }
    let tools = parse_tools(
        request.tools.as_ref(),
        request.tool_choice.as_ref(),
        request.parallel_tool_calls,
    )
    .map_err(|message| (400, message))?;
    let text = derive_text(request.text.as_ref()).map_err(|message| (400, message))?;
    let requested = derive_generation(request).map_err(|message| (400, message))?;
    let reasoning = derive_reasoning(request.reasoning.as_ref(), profile)
        .map_err(|message| (400, message))?;
    // The profile may adjust the sampler values (`RD-40` row 5). The echo stays
    // the requested numbers (`RD-29` 3.4).
    let generation = profile.sampling(reasoning.thinking, requested.generation);
    let verbosity_directive = if profile.applies_verbosity_directive() {
        text.verbosity.directive()
    } else {
        None
    };
    if tools.offers_tools() && text.format.json_required {
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
    let prepared = PreparedPrompt {
        system: String::new(),
        user,
        format: text.format,
        tools,
        reasoning,
        generation,
        generation_echo: requested.echo,
        verbosity: text.verbosity,
        verbosity_directive,
    };
    append_system_notes(&mut system, &prepared);
    Ok(PreparedPrompt { system, ..prepared })
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

/// The prompt and the effective input items of one request (PH-5, RD-22).
struct EffectivePrompt {
    system: String,
    user: String,
    input_items: Vec<serde_json::Value>,
}

/// Build the prompt and the effective input items.
///
/// Without `conversation` or `previous_response_id` the prepared prompt is
/// reused unchanged, so the behavior of `PH-1` through `PH-4b` is preserved
/// (`PH5-13`, `PH6A-14`).
///
/// With `conversation`, the conversation items are prepended to this request's
/// input items (`RD-23`). With `previous_response_id`, the referenced record's
/// input items and output items are prepended (`RD-22`). Neither carries the
/// referenced `instructions`, because those are not an item.
fn effective_prompt(
    request: &CreateRequest,
    prepared: &PreparedPrompt,
    previous: Option<&StoredResponse>,
    conversation: Option<&ConversationRecord>,
) -> Result<EffectivePrompt, String> {
    let current = input_items(&request.input);
    if let Some(conversation) = conversation {
        let mut items = stored_items_as_input(&conversation.items);
        items.extend(current.clone());
        let input = parse_input_items(items)?;
        let (mut system, user) = map_input(request.instructions.as_deref(), &input)?;
        append_system_notes(&mut system, prepared);
        // Only the request's own items are stored on the response record.
        return Ok(EffectivePrompt {
            system,
            user,
            input_items: current,
        });
    }
    let Some(previous) = previous else {
        return Ok(EffectivePrompt {
            system: prepared.system.clone(),
            user: prepared.user.clone(),
            input_items: current,
        });
    };
    let mut items = previous.input_items.clone();
    items.extend(output_items_as_input(&previous.response["output"]));
    items.extend(current);
    let input = parse_input_items(items.clone())?;
    let (mut system, user) = map_input(request.instructions.as_deref(), &input)?;
    append_system_notes(&mut system, prepared);
    Ok(EffectivePrompt {
        system,
        user,
        input_items: items,
    })
}

/// Append the tool transcription note and then the `text.verbosity` directive
/// to the system text, in that order (`RD-20`, `RD-30` 4.3). Both use the same
/// `\n` separator rule.
fn append_system_notes(system: &mut String, prepared: &PreparedPrompt) {
    if let Some(note) = prepared.tools.system_note() {
        push_system_note(system, &note);
    }
    if let Some(directive) = prepared.verbosity_directive {
        push_system_note(system, directive);
    }
}

/// Append one note with a `\n` separator when the system text is not empty.
fn push_system_note(system: &mut String, note: &str) {
    if !system.is_empty() {
        system.push('\n');
    }
    system.push_str(note);
}

/// The largest number of output items one generation can produce (`RD-23`).
fn max_output_items(tools: &ToolRequest) -> usize {
    if tools.offers_tools() && tools.parallel_tool_calls {
        tools.tools.len().max(1)
    } else {
        1
    }
}

/// Resolve `previous_response_id` before generation (PH-5, RD-22). An unknown
/// identifier is the shared 404 body, never a generation.
fn resolve_previous(
    store: &dyn ResponseStore,
    previous_response_id: Option<&str>,
    retention_secs: u64,
) -> Result<Option<StoredResponse>, HttpResponse> {
    let Some(id) = previous_response_id else {
        return Ok(None);
    };
    match crate::responses_limits::load_response(store, id, retention_secs) {
        Ok(Some(record)) => Ok(Some(record)),
        Ok(None) => Err(not_found(id)),
        Err(err) => Err(error_json(
            500,
            format!("failed to read the response store: {err}"),
        )),
    }
}

/// One fail-closed store write (`RD-15`). A failed write is an OpenAI-shaped
/// HTTP 500 `server_error`, never a success body without storage.
fn persist(
    store: &dyn ResponseStore,
    id: &str,
    record: &StoredResponse,
) -> Result<(), (u16, String)> {
    store
        .put(id, record)
        .map_err(|err| (500, format!("failed to store the response: {err}")))
}

/// Store the completed response when `store` is enabled (PH5-01, PH5-02).
fn maybe_store(
    store: &dyn ResponseStore,
    enabled: bool,
    completed: &serde_json::Value,
    input_items: &[serde_json::Value],
) -> Result<(), (u16, String)> {
    if !enabled {
        return Ok(());
    }
    let id = completed["id"].as_str().unwrap_or_default().to_string();
    let record = StoredResponse {
        response: completed.clone(),
        input_items: input_items.to_vec(),
    };
    persist(store, &id, &record)
}

/// Parse the body before the credential check. This order is RD-4: a request
/// that carries the body field `api_key` is a 400 even when `Authorization` is
/// absent or wrong, never a 401.
/// The top-level request fields the route implements. An unsupported field is
/// a request error, except under the `deepseek` profile (`RD-41`).
const KNOWN_FIELDS: [&str; 18] = [
    "model",
    "instructions",
    "input",
    "stream",
    "metadata",
    "text",
    "api_key",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "store",
    "previous_response_id",
    "conversation",
    "reasoning",
    "background",
    "temperature",
    "top_p",
    "max_output_tokens",
];

/// Decode the body. An unsupported top-level field is an HTTP 400 that names it
/// (`RD-40` row 8), except under the `deepseek` profile, which drops it and
/// logs its name (`RD-41`).
fn parse_body(body: &[u8], profile: ResponsesApi) -> Result<CreateRequest, (u16, String)> {
    let value: serde_json::Value = serde_json::from_slice(body)
        .map_err(|err| (400, format!("invalid request body: {}", err)))?;
    let Some(object) = value.as_object() else {
        return Err((
            400,
            "invalid request body: expected a JSON object".to_string(),
        ));
    };
    let unknown: Vec<&String> = object
        .keys()
        .filter(|key| !KNOWN_FIELDS.contains(&key.as_str()))
        .collect();
    if unknown.is_empty() {
        return serde_json::from_value(value)
            .map_err(|err| (400, format!("invalid request body: {}", err)));
    }
    if !profile.ignores_unsupported_fields() {
        let field = unknown[0];
        return Err((400, format!("unknown field `{field}`")));
    }
    for field in &unknown {
        eprintln!(
            "ltengine: `deepseek` profile ignored the unsupported request field `{field}`"
        );
    }
    let mut cleaned = object.clone();
    for field in &unknown {
        cleaned.remove(*field);
    }
    serde_json::from_value(serde_json::Value::Object(cleaned))
        .map_err(|err| (400, format!("invalid request body: {}", err)))
}

fn parse_and_authorize(
    body: &[u8],
    api_key: &str,
    authorization: Option<&str>,
    profile: ResponsesApi,
) -> Result<CreateRequest, (u16, String)> {
    let request = parse_body(body, profile)?;
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

#[post("/v1/responses")]
pub async fn create_response(
    req: HttpRequest,
    body: Result<web::Bytes, actix_web::Error>,
    args: web::Data<Arc<Args>>,
    llm: web::Data<Arc<llm::LLM>>,
    store: web::Data<AppStore>,
    registry: web::Data<Arc<crate::responses_background::CancelRegistry>>,
) -> HttpResponse {
    let authorization = bearer(&req);

    let body = match body {
        Ok(body) => body,
        Err(err) => return extractor_error(err),
    };

    let request = match parse_and_authorize(
        &body,
        &args.api_key,
        authorization,
        args.responses_api,
    ) {
        Ok(request) => request,
        Err((status, message)) => return error_json(status, message),
    };

    let loaded_model = loaded_model_identifier(&args);
    let prepared = match validate_request(&request, &loaded_model, args.responses_api) {
        Ok(prepared) => prepared,
        Err((status, message)) => return error_json(status, message),
    };

    // Resolve `previous_response_id` or `conversation` before generation, so an
    // unknown identifier is a 404 that never reaches the model (RD-22, RD-23).
    let previous = match resolve_previous(
        store.get_ref().as_ref(),
        request.previous_response_id.as_deref(),
        args.retention_secs,
    ) {
        Ok(previous) => previous,
        Err(response) => return response,
    };
    let conversation_id = request.conversation.clone();
    let conversation = match conversation_id.as_deref() {
        None => None,
        Some(id) => match crate::responses_conversations::load_conversation(
            store.get_ref().as_ref(),
            id,
            args.retention_secs,
        ) {
            Ok(Some(record)) => Some(record),
            Ok(None) => return crate::responses_conversations::conversation_not_found(id),
            Err(err) => {
                return error_json(500, format!("failed to read the conversation store: {err}"));
            }
        },
    };
    let EffectivePrompt {
        system,
        user,
        input_items,
    } = match effective_prompt(&request, &prepared, previous.as_ref(), conversation.as_ref()) {
        Ok(prompt) => prompt,
        Err(message) => return error_json(400, message),
    };
    let format = &prepared.format;
    let tools = &prepared.tools;
    // Each echo key is present only when the request carried its field
    // (`RD-29` 3.4, `RD-30` 4.4, `INVARIANT PH8-1`).
    let controls = ResponseControls {
        temperature: prepared.generation_echo.temperature,
        top_p: prepared.generation_echo.top_p,
        max_output_tokens: prepared.generation_echo.max_output_tokens,
        verbosity: request
            .text
            .as_ref()
            .and_then(|text| text.verbosity.as_ref())
            .map(|_| prepared.verbosity.as_str()),
    };
    // The operator option overrides the profile default (`RD-40` row 1).
    let trace_mode = args
        .reasoning_trace
        .unwrap_or(args.responses_api.reasoning_trace());
    let reasoning_part_type = args.responses_api.reasoning_content_type();

    // The conversation item limit is checked before generation, so the append
    // can never overflow it (RD-23).
    if let Some(record) = &conversation {
        let limit = args.max_conversation_items;
        let needed = record.items.len() + input_items.len() + max_output_items(tools);
        if limit != 0 && needed > limit {
            return error_json(
                400,
                format!("the conversation item limit of {limit} would be exceeded"),
            );
        }
    }

    let streaming = request.stream == Some(true);
    let grammar = if tools.offers_tools() {
        tools.grammar.as_deref()
    } else {
        format.grammar.as_deref()
    };
    let store_enabled = request.store.unwrap_or(true);

    // A background request stores a `queued` response, returns it, and runs one
    // generation in a blocking task (`PH-6b`, `RD-24`).
    if request.background == Some(true) {
        let concrete: Arc<llm::LLM> = Arc::clone(llm.get_ref());
        let generator: Arc<dyn crate::responses_background::Generate> = concrete;
        return crate::responses_background::start(crate::responses_background::BackgroundRequest {
            store: Arc::clone(store.get_ref()),
            generator,
            registry: Arc::clone(registry.get_ref()),
            model: loaded_model,
            metadata: request.metadata.clone(),
            conversation: conversation_id,
            echo: ToolEcho::from_request(tools),
            tools: tools.clone(),
            grammar: grammar.map(str::to_string),
            json_required: format.json_required,
            reasoning: prepared.reasoning.clone(),
            generation: prepared.generation,
            controls: controls.clone(),
            reasoning_trace: trace_mode,
            reasoning_part_type,
            system,
            user,
            input_items,
            conversation_record: conversation,
            retention_secs: args.retention_secs,
            max_stored_responses: args.max_stored_responses,
        });
    }

    // Remove the expired records and enforce the storage limit before the one
    // generation attempt (`PH-6c`, `RD-25`).
    if store_enabled {
        match crate::responses_limits::prepare_response_write(
            store.get_ref().as_ref(),
            args.retention_secs,
            args.max_stored_responses,
        ) {
            Ok(true) => {}
            Ok(false) => {
                return error_json(
                    507,
                    crate::responses_limits::storage_full_message(args.max_stored_responses),
                );
            }
            Err(err) => {
                return error_json(500, format!("failed to prepare the response store: {err}"));
            }
        }
    }

    match llm.run_prompt_usage_grammar(
        system,
        user,
        grammar,
        &prepared.reasoning.as_reasoning(),
        &prepared.generation,
    ) {
        Ok((text, trace, usage)) => {
            let echo = ToolEcho::from_request(tools);
            let conversation_ref = conversation_id.as_deref();
            // The reasoning item appears only under `verbatim` and only when a
            // trace was generated (`RD-33`, `RD-34`, `SP-NEVER-003`).
            let reasoning = if trace_mode == ReasoningTrace::Verbatim && !trace.is_empty() {
                Some(reasoning_item(&trace, reasoning_part_type))
            } else {
                None
            };
            let reasoning_ref = reasoning.as_ref();
            let (response, completed) = if tools.offers_tools() {
                // The model answers with the transcription envelope (RD-20).
                match parse_turn(&text, tools) {
                    Ok(ModelTurn::Message(answer)) => build_text_like(
                        &loaded_model,
                        &answer,
                        reasoning_ref,
                        &echo,
                        &request,
                        conversation_ref,
                        &controls,
                        &usage,
                        streaming,
                    ),
                    Ok(ModelTurn::Calls(calls)) => {
                        if streaming {
                            let (body, completed) = build_stream_body_with_conversation(
                                &loaded_model,
                                StreamOutput::Calls(&calls),
                                reasoning_ref,
                                &echo,
                                request.metadata.as_ref(),
                                conversation_ref,
                                &controls,
                                &usage,
                            );
                            (sse_response(body), completed)
                        } else {
                            let body = build_response_with_conversation(
                                &loaded_model,
                                prepend_reasoning(calls_output(&calls), reasoning_ref),
                                &echo,
                                request.metadata.as_ref(),
                                conversation_ref,
                                &controls,
                                &usage,
                            );
                            (HttpResponse::Ok().json(body.clone()), body)
                        }
                    }
                    Err(message) => return error_json(500, message),
                }
            } else {
                if format.json_required {
                    if let Err(message) = require_json(&text) {
                        return error_json(500, message);
                    }
                }
                build_text_like(
                    &loaded_model,
                    &text,
                    reasoning_ref,
                    &echo,
                    &request,
                    conversation_ref,
                    &controls,
                    &usage,
                    streaming,
                )
            };
            // One fail-closed attempt to store the completed response (RD-15).
            if let Err((status, message)) = maybe_store(
                store.get_ref().as_ref(),
                store_enabled,
                &completed,
                &input_items,
            ) {
                return error_json(status, message);
            }
            // Append the request input items and the output items to the
            // conversation (RD-23).
            if let (Some(id), Some(mut record)) = (conversation_id.as_deref(), conversation) {
                record.items.extend(input_items.iter().cloned());
                if let Some(output) = completed["output"].as_array() {
                    record.items.extend(output.iter().cloned());
                }
                if let Err(err) = store.get_ref().put_conversation(id, &record) {
                    return error_json(500, format!("failed to store the conversation: {err}"));
                }
            }
            response
        }
        Err(err) => {
            // The `max_output_tokens` context rejection is a request error, so
            // it is HTTP 400 with its own message (`RD-29` 3.3, `PH8-04`).
            let (status, message) = match err.downcast_ref::<llm::LLMError>() {
                Some(llm::LLMError::Busy) => (503, format!("{:#}", err)),
                Some(llm::LLMError::OutputTokensExceedContext(message)) => {
                    (400, message.clone())
                }
                _ => (500, format!("{:#}", err)),
            };
            eprintln!("responses error: {:#}", err);
            error_json(status, message)
        }
    }
}

/// Build the text response, streaming or not.
#[allow(clippy::too_many_arguments)]
fn build_text_like(
    model: &str,
    text: &str,
    reasoning: Option<&serde_json::Value>,
    echo: &ToolEcho,
    request: &CreateRequest,
    conversation: Option<&str>,
    controls: &ResponseControls,
    usage: &llm::TokenUsage,
    streaming: bool,
) -> (HttpResponse, serde_json::Value) {
    if streaming {
        let (body, completed) = build_stream_body_with_conversation(
            model,
            StreamOutput::Text(text),
            reasoning,
            echo,
            request.metadata.as_ref(),
            conversation,
            controls,
            usage,
        );
        (sse_response(body), completed)
    } else {
        let body = build_response_with_conversation(
            model,
            prepend_reasoning(message_output(text), reasoning),
            echo,
            request.metadata.as_ref(),
            conversation,
            controls,
            usage,
        );
        (HttpResponse::Ok().json(body.clone()), body)
    }
}

/// The SSE success response for one generated turn (RD-17, RD-20).
fn sse_response(body: String) -> HttpResponse {
    HttpResponse::Ok()
        .content_type("text/event-stream")
        .body(body)
}

#[cfg(test)]
#[path = "responses_tests.rs"]
mod tests;
