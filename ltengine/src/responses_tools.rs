//! Function-calling request handling for `POST /v1/responses` (PH-4b, RD-20).
//!
//! The route matches the OpenAI Responses API tool contract: a `function` tool
//! is `{type, name, parameters, description?, strict?}`, `tool_choice` is
//! `none`, `auto`, `required`, or `{type:"function", name}`, and
//! `parallel_tool_calls` is a boolean. The route executes no tool and calls no
//! upstream service (`SP-NEVER-009`).
//!
//! # Transcription protocol
//!
//! A local model has no native tool template on this route, so the route asks
//! for one JSON envelope and constrains the decode with a GBNF grammar when the
//! schemas allow it:
//!
//! - A direct answer: `{"message": "<text>"}`.
//! - One or more calls: `{"calls": [{"name": "<tool>", "arguments": {...}}]}`.
//!
//! `tool_choice` `none` offers no tool and keeps the plain-text path. `auto`
//! allows either shape. `required` and `{type:"function", name}` force the
//! `calls` shape.

use serde_json::{Value, json};

/// One accepted `function` tool after validation.
#[derive(Debug, Clone)]
pub struct FunctionTool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Value,
    /// The effective strict mode echoed in the response. It is true when the
    /// caller asked for strict, or when the caller omitted `strict` and the
    /// schema converts to a grammar (the OpenAI "attempt strict" default).
    pub strict: bool,
}

/// The accepted `tool_choice` value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    None,
    Auto,
    Required,
    Function(String),
}

/// The validated tool request.
#[derive(Debug, Clone)]
pub struct ToolRequest {
    pub tools: Vec<FunctionTool>,
    pub choice: ToolChoice,
    pub parallel_tool_calls: bool,
    /// The envelope grammar. `None` when no tool is offered or when the schemas
    /// cannot form a grammar and the caller did not require strict mode.
    pub grammar: Option<String>,
}

/// One tool call produced by the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelCall {
    pub name: String,
    /// The arguments as a JSON string, the OpenAI `function_call` shape.
    pub arguments: String,
}

/// The decoded model turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelTurn {
    Message(String),
    Calls(Vec<ModelCall>),
}

/// The values the response body echoes for the accepted tool request.
#[derive(Debug, Clone)]
pub struct ToolEcho {
    pub parallel_tool_calls: bool,
    pub tool_choice: Value,
    pub tools: Value,
}

impl ToolEcho {
    /// The echo of a request that offers no tool.
    pub fn text_only() -> Self {
        Self {
            parallel_tool_calls: false,
            tool_choice: json!("none"),
            tools: json!([]),
        }
    }

    pub fn from_request(request: &ToolRequest) -> Self {
        if !request.offers_tools() {
            // The route offers no tool, so the truthful echo is the text-only
            // default, regardless of the accepted `tools` and `tool_choice`
            // forms (`PH-4b` keeps the pre-tool body for those requests).
            return Self::text_only();
        }
        Self {
            parallel_tool_calls: request.parallel_tool_calls,
            tool_choice: request.echo_tool_choice(),
            tools: request.echo_tools(),
        }
    }
}

impl ToolRequest {
    /// True when the route offers at least one tool to the model.
    pub fn offers_tools(&self) -> bool {
        !self.tools.is_empty() && self.choice != ToolChoice::None
    }

    /// The tool definitions echoed in the response body.
    pub fn echo_tools(&self) -> Value {
        Value::Array(
            self.tools
                .iter()
                .map(|tool| {
                    json!({
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                        "strict": tool.strict,
                    })
                })
                .collect(),
        )
    }

    /// The resolved `tool_choice` echoed in the response body.
    pub fn echo_tool_choice(&self) -> Value {
        match &self.choice {
            ToolChoice::None => json!("none"),
            ToolChoice::Auto => json!("auto"),
            ToolChoice::Required => json!("required"),
            ToolChoice::Function(name) => json!({"type": "function", "name": name}),
        }
    }

    /// The system note that describes the envelope and the declared tools.
    pub fn system_note(&self) -> Option<String> {
        if !self.offers_tools() {
            return None;
        }
        let mut note = String::from(
            "You can call the tools below. Reply with one JSON object and nothing else.\n\
             To answer the user directly, reply {\"message\": \"<answer>\"}.\n\
             To call a tool, reply {\"calls\": [{\"name\": \"<tool>\", \"arguments\": {...}}]}.\n",
        );
        match &self.choice {
            ToolChoice::Required => note.push_str("You must call at least one tool.\n"),
            ToolChoice::Function(name) => {
                note.push_str(&format!("You must call the function `{name}`.\n"));
            }
            _ => {}
        }
        note.push_str("Tools:\n");
        for tool in &self.tools {
            note.push_str(&format!(
                "- {}: {} arguments {}\n",
                tool.name,
                tool.description.as_deref().unwrap_or("no description"),
                tool.parameters
            ));
        }
        Some(note)
    }
}

/// Validation wording for one `tools` entry. Every message names `tools`, so
/// the handler returns a 400 that names the rejected field (`SP-MUST-011`).
fn tools_error(message: &str) -> String {
    format!("`tools` {message}")
}

/// Parse and validate `tools`, `tool_choice`, and `parallel_tool_calls`.
pub fn parse_tools(
    tools: Option<&Vec<Value>>,
    tool_choice: Option<&Value>,
    parallel_tool_calls: Option<bool>,
) -> Result<ToolRequest, String> {
    let mut pending = Vec::new();
    if let Some(list) = tools {
        for tool in list {
            pending.push(parse_tool(tool)?);
        }
    }
    let choice = parse_tool_choice(tool_choice, &pending)?;
    let parallel_tool_calls = parallel_tool_calls.unwrap_or(true);

    let mut request = ToolRequest {
        tools: Vec::new(),
        choice,
        parallel_tool_calls,
        grammar: None,
    };
    if pending.is_empty() || request.choice == ToolChoice::None {
        // No tool is offered, so the effective strict mode is false and the
        // body echoes the accepted declarations with that truthful value.
        request.tools = pending
            .iter()
            .map(|tool| FunctionTool {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool.parameters.clone(),
                strict: false,
            })
            .collect();
        return Ok(request);
    }

    // Build the envelope grammar. A strict-requested tool whose schema cannot
    // convert is a 400; an omitted strict mode falls back to best effort.
    let grammar = match envelope_schema(&pending, request.parallel_tool_calls, &request.choice)
        .ok_or_else(|| "`tools` cannot form a grammar".to_string())
        .and_then(|schema| {
            llama_cpp_common::json_schema_to_grammar(&schema.to_string())
                .map_err(|err| format!("`tools` schema is not a supported JSON schema: {err}"))
        }) {
        Ok(grammar) => Some(grammar),
        Err(message) => {
            if pending
                .iter()
                .any(|tool| tool.strict_requested == Some(true))
            {
                return Err(message);
            }
            None
        }
    };
    // The tools are offered in both outcomes. The effective strict mode is true
    // only when the envelope grammar was built and the caller did not opt out.
    let built = grammar.is_some();
    request.tools = pending
        .iter()
        .map(|tool| FunctionTool {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: tool.parameters.clone(),
            strict: built && tool.strict_requested != Some(false),
        })
        .collect();
    request.grammar = grammar;
    Ok(request)
}

/// One `tools` entry before the effective strict mode is known.
#[derive(Debug, Clone)]
struct PendingTool {
    name: String,
    description: Option<String>,
    parameters: Value,
    strict_requested: Option<bool>,
}

fn parse_tool(value: &Value) -> Result<PendingTool, String> {
    let object = value
        .as_object()
        .ok_or_else(|| tools_error("entry must be an object"))?;
    for key in object.keys() {
        if !matches!(
            key.as_str(),
            "type" | "name" | "description" | "parameters" | "strict"
        ) {
            return Err(tools_error(&format!("entry has an unsupported field `{key}`")));
        }
    }
    let kind = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| tools_error("entry requires a string `type`"))?;
    if kind != "function" {
        return Err(tools_error(&format!(
            "entry type `{kind}` is not supported; only `function` tools are implemented"
        )));
    }
    let name = object
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| tools_error("function entry requires a non-empty string `name`"))?
        .to_string();
    let parameters = object
        .get("parameters")
        .filter(|value| value.is_object())
        .ok_or_else(|| tools_error(&format!("function `{name}` requires an object `parameters`")))?;
    let description = match object.get("description") {
        None | Some(Value::Null) => None,
        Some(Value::String(text)) => Some(text.clone()),
        Some(_) => {
            return Err(tools_error(&format!(
                "function `{name}` `description` must be a string"
            )));
        }
    };
    let strict_requested = match object.get("strict") {
        None | Some(Value::Null) => None,
        Some(Value::Bool(value)) => Some(*value),
        Some(_) => {
            return Err(tools_error(&format!(
                "function `{name}` `strict` must be a boolean"
            )));
        }
    };
    Ok(PendingTool {
        name,
        description,
        parameters: parameters.clone(),
        strict_requested,
    })
}

fn parse_tool_choice(value: Option<&Value>, tools: &[PendingTool]) -> Result<ToolChoice, String> {
    let Some(value) = value else {
        return Ok(ToolChoice::Auto);
    };
    match value {
        Value::Null => Ok(ToolChoice::Auto),
        Value::String(choice) => match choice.as_str() {
            "none" => Ok(ToolChoice::None),
            "auto" => Ok(ToolChoice::Auto),
            "required" => {
                if tools.is_empty() {
                    Err("`tool_choice` `required` needs at least one `tools` entry".to_string())
                } else {
                    Ok(ToolChoice::Required)
                }
            }
            other => Err(format!(
                "`tool_choice` `{other}` is not supported; use `none`, `auto`, `required`, or a named function"
            )),
        },
        Value::Object(object) => {
            let kind = object.get("type").and_then(Value::as_str);
            let name = object.get("name").and_then(Value::as_str);
            match (kind, name) {
                (Some("function"), Some(name)) => {
                    if tools.iter().any(|tool| tool.name == name) {
                        Ok(ToolChoice::Function(name.to_string()))
                    } else {
                        Err(format!("`tool_choice` names `{name}`, which is not a declared tool"))
                    }
                }
                _ => Err(
                    "`tool_choice` object must be `{\"type\":\"function\",\"name\":\"<declared>\"}`"
                        .to_string(),
                ),
            }
        }
        _ => Err("`tool_choice` must be `none`, `auto`, `required`, or a named function".to_string()),
    }
}

/// The JSON schema of the transcription envelope.
fn envelope_schema(tools: &[PendingTool], parallel: bool, choice: &ToolChoice) -> Option<Value> {
    // The message shape is available only when the model may answer directly.
    let allow_message = matches!(choice, ToolChoice::Auto);
    let allowed: Vec<&PendingTool> = match choice {
        ToolChoice::Function(name) => tools.iter().filter(|tool| &tool.name == name).collect(),
        _ => tools.iter().collect(),
    };
    if allowed.is_empty() {
        return None;
    }
    let call_items: Vec<Value> = allowed
        .iter()
        .map(|tool| {
            json!({
                "type": "object",
                "properties": {
                    "name": {"enum": [tool.name]},
                    "arguments": tool.parameters,
                },
                "required": ["name", "arguments"],
                "additionalProperties": false,
            })
        })
        .collect();
    let call_schema = if call_items.len() == 1 {
        call_items[0].clone()
    } else {
        json!({"anyOf": call_items})
    };
    let mut calls_array = json!({"type": "array", "minItems": 1, "items": call_schema});
    if !parallel {
        calls_array["maxItems"] = json!(1);
    }
    let calls_shape = json!({
        "type": "object",
        "properties": {"calls": calls_array},
        "required": ["calls"],
        "additionalProperties": false,
    });
    if allow_message {
        let message_shape = json!({
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
            "additionalProperties": false,
        });
        Some(json!({"anyOf": [message_shape, calls_shape]}))
    } else {
        Some(calls_shape)
    }
}

/// Decode the model output into a turn. `request` supplies the declared tool
/// names and the parallel-call limit.
pub fn parse_turn(text: &str, request: &ToolRequest) -> Result<ModelTurn, String> {
    let value: Value = serde_json::from_str(text.trim())
        .map_err(|err| format!("the model did not produce the tool envelope: {err}"))?;
    if let Some(calls) = value.get("calls") {
        let calls = calls
            .as_array()
            .ok_or_else(|| "the tool envelope `calls` must be an array".to_string())?;
        if calls.is_empty() {
            return Err("the tool envelope `calls` must not be empty".to_string());
        }
        if !request.parallel_tool_calls && calls.len() > 1 {
            return Err(format!(
                "the model returned {} calls but `parallel_tool_calls` is false",
                calls.len()
            ));
        }
        let mut decoded = Vec::with_capacity(calls.len());
        for call in calls {
            let name = call
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| "a tool call requires a string `name`".to_string())?;
            if !request.tools.iter().any(|tool| tool.name == name) {
                return Err(format!("the model named `{name}`, which is not a declared tool"));
            }
            if let ToolChoice::Function(required) = &request.choice {
                if required != name {
                    return Err(format!(
                        "`tool_choice` requires `{required}` but the model named `{name}`"
                    ));
                }
            }
            let arguments = call
                .get("arguments")
                .filter(|value| value.is_object())
                .ok_or_else(|| format!("tool call `{name}` requires an object `arguments`"))?;
            decoded.push(ModelCall {
                name: name.to_string(),
                arguments: arguments.to_string(),
            });
        }
        return Ok(ModelTurn::Calls(decoded));
    }
    if let Some(message) = value.get("message").and_then(Value::as_str) {
        if request.choice != ToolChoice::Auto {
            return Err("`tool_choice` requires a tool call, but the model answered with text".to_string());
        }
        return Ok(ModelTurn::Message(message.to_string()));
    }
    Err("the model output is not the expected tool envelope".to_string())
}
