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

  if (failures.length > 0) {
    console.log(`RESULT FAIL (${failures.length} check(s) failed): ${failures.join(", ")}`);
    return 1;
  }
  console.log("RESULT PASS");
  return 0;
}

process.exitCode = await main();
