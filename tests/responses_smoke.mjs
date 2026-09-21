// OpenAI JavaScript SDK smoke test for the local `POST /v1/responses` route.
//
// Roadmap PH-2 smoke test. Target SDK: `openai` 7.15.0 on Node >= 22 (RD-8).
// Command: `node tests/responses_smoke.mjs`.
//
// The client is synchronous: every request is a plain awaited non-streaming
// call. The script calls the local LTEngine server only; it never calls the
// OpenAI service (RD-1). It adds no dependency: it uses the `openai` package
// that the test environment provides.
//
// Environment:
// - `LTENGINE_BASE_URL`: base URL of the local server. Default
//   `http://127.0.0.1:5050/v1`. The host must be a local host.
// - `LTENGINE_API_KEY`: value of the server `--api-key` option. Empty when the
//   server runs without a key.
// - `LTENGINE_MODEL`: identifier of the loaded model. The request `model` field
//   and the response `model` field must equal this value (SP-MUST-009, RD-5).
// - `LTENGINE_SDK_VERSION_ANY`: value `1` bypasses the pinned-SDK-version check.
//
// Both requests also check the required response fields of the text-only route
// (`parallel_tool_calls`, `tool_choice`, `tools`) and the `metadata` echo.
// Request 1 sends `metadata`; request 2 omits it and the response must carry
// `null`.
//
// The script also covers the planned items of `PH-4` through the SDK:
// `text.format` `json_object` and strict `json_schema` (`SP-PLANNED-003`), a
// forced function call (`SP-PLANNED-004`), and a named `tool_choice` with
// `parallel_tool_calls: true` (`SP-PLANNED-005`). Those requests run against
// the loaded local model, so the generated text is not part of the contract.
// The checks assert the response shape, the identifier prefixes, and JSON
// validity, and they never assert a particular generated value.
//
// The script also covers the stored and long-running planned items through the
// SDK: a stored response, `previous_response_id`, retrieval, deletion, and the
// input-item list (`SP-PLANNED-006` through `SP-PLANNED-009`); conversations
// and conversation items (`SP-PLANNED-010`); and a background response with
// polling and cancellation (`SP-PLANNED-011`). Those checks create server-side
// records and delete the records that they create.
//
// The script also covers the streamed and counted planned items through the
// SDK: the `RD-17` event order over `client.responses.stream(...)`
// (`SP-PLANNED-001`) and the `usage` counts of a non-streaming response
// (`SP-PLANNED-002`).
//
// Exit code 0 means every check passed.

import OpenAI from "openai";
import { VERSION as SDK_VERSION } from "openai/version";

const PINNED_SDK = "7.15.0";
const PINNED_NODE_MAJOR = 22;
const DEFAULT_BASE_URL = "http://127.0.0.1:5050/v1";
const LOCAL_HOSTS = new Set(["127.0.0.1", "localhost", "::1"]);

const baseUrl = process.env.LTENGINE_BASE_URL ?? DEFAULT_BASE_URL;
const apiKey = process.env.LTENGINE_API_KEY ?? "";
const model = (process.env.LTENGINE_MODEL ?? "").trim();
const smokeMetadata = { suite: "responses-smoke" };

const plannedTool = {
  type: "function",
  name: "get_weather",
  description: "Get the current weather for a city.",
  parameters: {
    type: "object",
    properties: { location: { type: "string" } },
    required: ["location"],
    additionalProperties: false,
  },
};

const capitalSchema = {
  type: "object",
  properties: { answer: { type: "string" } },
  required: ["answer"],
  additionalProperties: false,
};

const terminalStatuses = new Set(["completed", "failed", "cancelled"]);
const declaredStatuses = new Set(["queued", "in_progress", ...terminalStatuses]);
const backgroundPrompt = "Count from 1 to 300, one number per line.";
const expectedStreamTypes = [
  "response.created",
  "response.in_progress",
  "response.output_item.added",
  "response.content_part.added",
  "response.output_text.done",
  "response.content_part.done",
  "response.output_item.done",
  "response.completed",
];

const failures = [];

function report(label, status, detail = "") {
  console.log(`${status} ${label}${detail ? ` - ${detail}` : ""}`);
  if (status === "FAIL") failures.push(label);
}

function configError(message) {
  console.log(`FAIL config: ${message}`);
  return 1;
}

function buildClient(key) {
  return new OpenAI({ baseURL: baseUrl, apiKey: key });
}

// Check the OpenAI-shaped success contract of PH-1 (SP-MUST-006, SP-MUST-009,
// SP-MUST-010).
function verify(label, response, metadata = null) {
  report(
    `${label}: object is \`response\``,
    response.object === "response" ? "PASS" : "FAIL",
    `object=${JSON.stringify(response.object)}`,
  );

  report(
    `${label}: id starts with \`resp_\``,
    String(response.id ?? "").startsWith("resp_") ? "PASS" : "FAIL",
    `id=${JSON.stringify(response.id)}`,
  );

  report(
    `${label}: model is the loaded model`,
    response.model === model ? "PASS" : "FAIL",
    `model=${JSON.stringify(response.model)} expected=${JSON.stringify(model)}`,
  );

  const text = response.output_text ?? "";
  report(
    `${label}: output_text is present`,
    text.trim() ? "PASS" : "FAIL",
    `output_text=${JSON.stringify(text)}`,
  );

  const echoed = response.metadata ?? null;
  report(
    `${label}: metadata is echoed`,
    JSON.stringify(echoed) === JSON.stringify(metadata) ? "PASS" : "FAIL",
    `metadata=${JSON.stringify(echoed)} expected=${JSON.stringify(metadata)}`,
  );

  report(
    `${label}: parallel_tool_calls is false`,
    response.parallel_tool_calls === false ? "PASS" : "FAIL",
    `parallel_tool_calls=${JSON.stringify(response.parallel_tool_calls)}`,
  );

  report(
    `${label}: tool_choice is \`none\``,
    response.tool_choice === "none" ? "PASS" : "FAIL",
    `tool_choice=${JSON.stringify(response.tool_choice)}`,
  );

  report(
    `${label}: tools is empty`,
    Array.isArray(response.tools) && response.tools.length === 0 ? "PASS" : "FAIL",
    `tools=${JSON.stringify(response.tools)}`,
  );

  verifyUsage(label, response);
}

// Check SP-PLANNED-002: `usage` carries real counts and the exact sum (RD-18).
function verifyUsage(label, response) {
  const usage = response.usage ?? null;
  if (usage === null) {
    report(`${label}: usage is present`, "FAIL", "usage is null");
    return;
  }
  const { input_tokens: inputTokens, output_tokens: outputTokens, total_tokens: total } = usage;
  const positive =
    Number.isInteger(inputTokens) &&
    Number.isInteger(outputTokens) &&
    Number.isInteger(total) &&
    inputTokens > 0 &&
    outputTokens > 0 &&
    total > 0;
  report(
    `${label}: usage carries positive integer counts`,
    positive ? "PASS" : "FAIL",
    `usage=${inputTokens}/${outputTokens}/${total}`,
  );
  report(
    `${label}: total_tokens is the exact sum`,
    total === inputTokens + outputTokens ? "PASS" : "FAIL",
    `total=${total} sum=${inputTokens}+${outputTokens}`,
  );
}

// Check SP-PLANNED-001: the SDK stream carries the RD-17 event order.
async function verifyStream(client) {
  let events;
  let final;
  try {
    const stream = client.responses.stream({ model, input: "Say OK" });
    events = [];
    for await (const event of stream) events.push([event.type, event.sequence_number]);
    final = await stream.finalResponse();
  } catch (error) {
    report(
      "streaming: the SDK stream completes",
      "FAIL",
      `${error?.constructor?.name}: ${error?.message}`,
    );
    return;
  }

  const collected = events.map(([type]) => type);
  report("streaming: the SDK stream completes", "PASS", `events=${events.length}`);
  const nonDelta = collected.filter((type) => type !== "response.output_text.delta");
  report(
    "streaming: the RD-17 event order",
    JSON.stringify(nonDelta) === JSON.stringify(expectedStreamTypes) ? "PASS" : "FAIL",
    `types=${JSON.stringify(collected)}`,
  );
  const numbers = events.map(([, number]) => number);
  const increasing = numbers.every((number, index) => index === 0 || number > numbers[index - 1]);
  report(
    "streaming: sequence_number increases",
    increasing ? "PASS" : "FAIL",
    `sequence_numbers=${JSON.stringify(numbers)}`,
  );
  const text = final?.output_text ?? "";
  report(
    "streaming: the final response carries output_text",
    final?.object === "response" && text.trim() ? "PASS" : "FAIL",
    `object=${JSON.stringify(final?.object)}`,
  );
}

// Run one request. Report a FAIL and return `null` when the SDK raises.
async function request(label, call) {
  try {
    return await call();
  } catch (error) {
    report(label, "FAIL", `${error?.constructor?.name}: ${error?.message}`);
    return null;
  }
}

function functionCalls(response) {
  return (response?.output ?? []).filter((item) => item.type === "function_call");
}

// Check SP-PLANNED-003: `output_text` parses as JSON, and conforms to the
// schema.
function verifyJsonText(label, response, schemaKey = null) {
  if (!response) return;
  const text = response.output_text ?? "";
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch (error) {
    report(
      `${label}: output_text parses as JSON`,
      "FAIL",
      `output_text=${JSON.stringify(text)} error=${error.message}`,
    );
    return;
  }
  report(`${label}: output_text parses as JSON`, "PASS", `value=${JSON.stringify(parsed)}`);
  if (schemaKey === null) return;
  const conforms =
    typeof parsed === "object" && parsed !== null && typeof parsed[schemaKey] === "string";
  report(
    `${label}: output conforms to the schema`,
    conforms ? "PASS" : "FAIL",
    `value=${JSON.stringify(parsed)}`,
  );
}

// Check SP-PLANNED-004 and SP-PLANNED-005: the `function_call` item fields.
function verifyFunctionCall(label, response, expectedName) {
  if (!response) return;
  const calls = functionCalls(response);
  report(
    `${label}: at least one function_call item`,
    calls.length > 0 ? "PASS" : "FAIL",
    `count=${calls.length}`,
  );
  calls.forEach((item, index) => {
    const callLabel = `${label}: call ${index}`;
    report(
      `${callLabel} id starts with \`fc_\``,
      String(item.id ?? "").startsWith("fc_") ? "PASS" : "FAIL",
      `id=${JSON.stringify(item.id)}`,
    );
    report(
      `${callLabel} call_id starts with \`call_\``,
      String(item.call_id ?? "").startsWith("call_") ? "PASS" : "FAIL",
      `call_id=${JSON.stringify(item.call_id)}`,
    );
    report(
      `${callLabel} name is ${JSON.stringify(expectedName)}`,
      item.name === expectedName ? "PASS" : "FAIL",
      `name=${JSON.stringify(item.name)}`,
    );
    let parsed = true;
    try {
      JSON.parse(item.arguments);
    } catch {
      parsed = false;
    }
    report(
      `${callLabel} arguments parse as JSON`,
      parsed ? "PASS" : "FAIL",
      `arguments=${JSON.stringify(item.arguments)}`,
    );
  });
}

// Check that a request is rejected with the OpenAI-shaped HTTP 404 (RD-22).
async function expectNotFound(label, call) {
  try {
    await call();
  } catch (error) {
    report(
      label,
      error?.status === 404 ? "PASS" : "FAIL",
      `${error?.constructor?.name}: ${error?.message}`,
    );
    return;
  }
  report(label, "FAIL", "the request succeeded");
}

// SP-PLANNED-013: the `reasoning` request field. The loaded model's template
// decides what the effort value changes, so the check asserts the request
// contract only.
async function verifyReasoning(client) {
  const accepted = await request("reasoning effort accepted", () =>
    client.responses.create({ model, input: "Say OK.", reasoning: { effort: "low" } }),
  );
  if (accepted) {
    report(
      "reasoning effort: the response carries a resp_ id",
      String(accepted.id ?? "").startsWith("resp_") ? "PASS" : "FAIL",
      `id=${JSON.stringify(accepted.id)}`,
    );
  }

  try {
    await client.responses.create({
      model,
      input: "Say OK.",
      reasoning: { effort: "enormous" },
    });
  } catch (error) {
    const status = error?.status ?? error?.statusCode;
    report(
      "reasoning effort: an unsupported value is a 400",
      status === 400 ? "PASS" : "FAIL",
      `status=${JSON.stringify(status)}`,
    );
    report(
      "reasoning effort: the rejection names reasoning.effort",
      String(error?.message ?? "").includes("reasoning.effort") ? "PASS" : "FAIL",
      String(error?.message ?? "").split("\n")[0].slice(0, 100),
    );
    return;
  }
  report("reasoning effort: an unsupported value is a 400", "FAIL", "the request succeeded");
}

// Check SP-PLANNED-006 through SP-PLANNED-009.
async function verifyStoredFlow(client) {
  const stored = await request("stored response", () =>
    client.responses.create({ model, input: "Say OK", store: true }),
  );
  if (!stored) return;
  report(
    "stored response: id starts with `resp_`",
    String(stored.id ?? "").startsWith("resp_") ? "PASS" : "FAIL",
    `id=${JSON.stringify(stored.id)}`,
  );

  const retrieved = await request("stored response: retrieve", () =>
    client.responses.retrieve(stored.id),
  );
  if (retrieved) {
    report(
      "stored response: retrieve returns the same id",
      retrieved.id === stored.id ? "PASS" : "FAIL",
      `id=${JSON.stringify(retrieved.id)}`,
    );
  }

  const items = await request("stored response: input items", () =>
    client.responses.inputItems.list(stored.id),
  );
  if (items) {
    const data = items.data ?? [];
    const listed = data.some((item) => item.role === "user");
    report(
      "stored response: the input item is listed",
      listed ? "PASS" : "FAIL",
      `count=${data.length}`,
    );
  }

  const chained = await request("previous_response_id", () =>
    client.responses.create({
      model,
      input: "Say OK again",
      previous_response_id: stored.id,
      store: true,
    }),
  );
  if (chained) {
    report(
      "previous_response_id: a new `resp_` id",
      String(chained.id ?? "").startsWith("resp_") && chained.id !== stored.id ? "PASS" : "FAIL",
      `id=${JSON.stringify(chained.id)}`,
    );
  }

  await request("stored response: delete", () => client.responses.delete(stored.id));
  await expectNotFound("deleted response: retrieve is not found", () =>
    client.responses.retrieve(stored.id),
  );
  if (chained) {
    await request("previous_response_id: delete the chained response", () =>
      client.responses.delete(chained.id),
    );
  }
}

// Check SP-PLANNED-010.
async function verifyConversationFlow(client) {
  const conversation = await request("conversation: create", () => client.conversations.create());
  if (!conversation) return;
  report(
    "conversation: id starts with `conv_`",
    String(conversation.id ?? "").startsWith("conv_") ? "PASS" : "FAIL",
    `id=${JSON.stringify(conversation.id)}`,
  );

  await request("conversation: add an item", () =>
    client.conversations.items.create(conversation.id, {
      items: [
        {
          type: "message",
          role: "user",
          content: [{ type: "input_text", text: "Say hi" }],
        },
      ],
    }),
  );

  const items = await request("conversation: list items", () =>
    client.conversations.items.list(conversation.id),
  );
  if (items) {
    const data = items.data ?? [];
    const listed = data.some((item) => item.role === "user");
    report("conversation: the added item is listed", listed ? "PASS" : "FAIL", `count=${data.length}`);
  }

  const response = await request("conversation: response with the field", () =>
    client.responses.create({ model, input: "Say OK", conversation: conversation.id }),
  );
  if (response) {
    report(
      "conversation: the response carries a `resp_` id",
      String(response.id ?? "").startsWith("resp_") ? "PASS" : "FAIL",
      `id=${JSON.stringify(response.id)}`,
    );
  }

  const deleted = await request("conversation: delete", () =>
    client.conversations.delete(conversation.id),
  );
  if (deleted) {
    report(
      "conversation: delete reports `conversation.deleted`",
      deleted.object === "conversation.deleted" ? "PASS" : "FAIL",
      `object=${JSON.stringify(deleted.object)}`,
    );
  }
}

// Check SP-PLANNED-011.
async function verifyBackgroundFlow(client) {
  const job = await request("background: create", () =>
    client.responses.create({ model, input: "Say OK", background: true, store: true }),
  );
  if (!job) return;
  report(
    "background: the created response is `queued`",
    job.status === "queued" ? "PASS" : "FAIL",
    `status=${JSON.stringify(job.status)}`,
  );

  const seen = [];
  let final = job;
  for (let attempt = 0; attempt < 120; attempt += 1) {
    final = await request("background: poll", () => client.responses.retrieve(job.id));
    if (!final) break;
    seen.push(final.status);
    if (terminalStatuses.has(final.status)) break;
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  report(
    "background: the job reaches a terminal status",
    final !== null && final !== undefined && terminalStatuses.has(final.status) ? "PASS" : "FAIL",
    `status=${JSON.stringify(final?.status)}`,
  );
  report(
    "background: every observed status is declared",
    seen.every((status) => declaredStatuses.has(status)) ? "PASS" : "FAIL",
    `observed=${JSON.stringify(seen)}`,
  );

  const target = await request("background cancel: create", () =>
    client.responses.create({ model, input: backgroundPrompt, background: true, store: true }),
  );
  if (target) {
    const cancelled = await request("background cancel: cancel", () =>
      client.responses.cancel(target.id),
    );
    if (cancelled) {
      report(
        "background cancel: the response id is unchanged",
        cancelled.id === target.id ? "PASS" : "FAIL",
        `id=${JSON.stringify(cancelled.id)}`,
      );
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
    const after = await request("background cancel: status", () =>
      client.responses.retrieve(target.id),
    );
    if (after) {
      report(
        "background cancel: the status is `cancelled`",
        after.status === "cancelled" ? "PASS" : "FAIL",
        `status=${JSON.stringify(after.status)}`,
      );
    }
    const again = await request("background cancel: second cancel", () =>
      client.responses.cancel(target.id),
    );
    if (again) {
      report(
        "background cancel: the second cancel is idempotent",
        again.id === target.id ? "PASS" : "FAIL",
        `id=${JSON.stringify(again.id)}`,
      );
    }
    await request("background cancel: delete the cancelled job", () =>
      client.responses.delete(target.id),
    );
  }

  await request("background: delete the completed job", () => client.responses.delete(job.id));
}

async function main() {
  const nodeMajor = Number(process.version.replace(/^v/, "").split(".")[0]);
  if (nodeMajor < PINNED_NODE_MAJOR) {
    return configError(
      `Node ${process.version} is below the tested target (Node >= ${PINNED_NODE_MAJOR})`,
    );
  }

  let host;
  try {
    host = new URL(baseUrl).hostname;
  } catch {
    return configError(`base URL ${baseUrl} is not a valid URL`);
  }
  if (!LOCAL_HOSTS.has(host)) {
    return configError(
      `base URL host ${JSON.stringify(host)} is not a local host; this test never calls the OpenAI service`,
    );
  }
  if (!model) {
    return configError("LTENGINE_MODEL is empty; set it to the loaded model identifier");
  }

  const sdkVersion = SDK_VERSION;
  if (sdkVersion !== PINNED_SDK && process.env.LTENGINE_SDK_VERSION_ANY !== "1") {
    return configError(
      `openai ${sdkVersion} does not match the pinned target ${PINNED_SDK}; ` +
        "set LTENGINE_SDK_VERSION_ANY=1 to test a different version",
    );
  }

  console.log(`INFO sdk: openai ${sdkVersion} on Node ${process.version}`);
  console.log(`INFO target: ${baseUrl} with model ${JSON.stringify(model)}`);

  const client = buildClient(apiKey || "not-configured");

  if (apiKey) {
    try {
      await buildClient("wrong-key").responses.create({ model, input: "Say OK" });
      report("bearer auth rejects a wrong key", "FAIL", "the request succeeded");
    } catch (error) {
      report(
        "bearer auth rejects a wrong key",
        error?.status === 401 ? "PASS" : "FAIL",
        `${error?.constructor?.name}: ${error?.message}`,
      );
    }
  } else {
    report("bearer auth rejects a wrong key", "SKIP", "LTENGINE_API_KEY is empty");
  }

  verify(
    "text input",
    await client.responses.create({
      model,
      instructions: "Answer in one short word.",
      input: "Say OK",
      metadata: smokeMetadata,
    }),
    smokeMetadata,
  );

  verify(
    "message-array input",
    await client.responses.create({
      model,
      input: [
        { role: "system", content: "Answer in one short word." },
        { role: "user", content: [{ type: "input_text", text: "Say hi" }] },
      ],
    }),
  );

  // SP-PLANNED-003: structured output.
  verifyJsonText(
    "structured output json_object",
    await request("structured output json_object", () =>
      client.responses.create({
        model,
        instructions: "Answer with JSON only.",
        input: "Return a JSON object with one key, answer, whose value is the capital of France.",
        text: { format: { type: "json_object" } },
      }),
    ),
  );

  verifyJsonText(
    "structured output json_schema",
    await request("structured output json_schema", () =>
      client.responses.create({
        model,
        instructions: "Answer with JSON only.",
        input: "Return the capital of France in the answer field.",
        text: {
          format: { type: "json_schema", name: "capital", strict: true, schema: capitalSchema },
        },
      }),
    ),
    "answer",
  );

  // SP-PLANNED-004: a forced custom function call.
  verifyFunctionCall(
    "function calling required",
    await request("function calling required", () =>
      client.responses.create({
        model,
        input: "What is the weather in Paris?",
        tools: [plannedTool],
        tool_choice: "required",
      }),
    ),
    "get_weather",
  );

  // SP-PLANNED-005: a named `tool_choice` and the `parallel_tool_calls` echo.
  const named = await request("named tool_choice", () =>
    client.responses.create({
      model,
      input: "What is the weather in Paris?",
      tools: [plannedTool],
      tool_choice: { type: "function", name: "get_weather" },
      parallel_tool_calls: true,
    }),
  );
  verifyFunctionCall("named tool_choice", named, "get_weather");
  if (named) {
    report(
      "named tool_choice: parallel_tool_calls is echoed",
      named.parallel_tool_calls === true ? "PASS" : "FAIL",
      `parallel_tool_calls=${JSON.stringify(named.parallel_tool_calls)}`,
    );
  }

  // SP-PLANNED-013: the reasoning request field.
  await verifyReasoning(client);

  // SP-PLANNED-006 through -011: the stored and long-running planned items.
  await verifyStoredFlow(client);
  await verifyConversationFlow(client);
  await verifyBackgroundFlow(client);

  // SP-PLANNED-001: the streamed event order.
  await verifyStream(client);

  if (failures.length > 0) {
    console.log(`RESULT FAIL (${failures.length} check(s) failed): ${failures.join(", ")}`);
    return 1;
  }
  console.log("RESULT PASS");
  return 0;
}

process.exitCode = await main();
