# Responses API compatibility profile

Compatibility profile version: `1`.

This file is the compatibility-version and support-matrix holder of roadmap
phase `PH-2` (roadmap RD-6). The version value comes from RD-7. The SDK targets
and the test targets come from RD-8. This file is the only source of the
compatibility claim. `local-docs/` holds the roadmap and the decision records.

## 1. Claim

LTEngine declares compatibility profile version `1` with the OpenAI SDKs in
section 2, for the listed text scope of section 4, on the listed SDK versions
and runtime versions.

This profile claims no full OpenAI parity (`SP-NEVER-012`). It claims no
byte-for-byte response parity (`SP-NEVER-011`). It claims no upstream call to
the OpenAI service (RD-1). It claims nothing outside the capability lists of
section 4 and section 5. The limits of section 5 are part of the claim.

## 2. Test targets

| Field | Value |
| ----- | ----- |
| Compatibility profile version | `1` (RD-7) |
| Python SDK | `openai` 3.14.1 on Python >= 3.10 |
| JavaScript SDK | `openai` 7.15.0 on Node >= 22 |
| Endpoint under test | `POST /v1/responses` on a local LTEngine server |
| Generation mode | Non-streaming and SSE streaming (`stream: true`, RD-17) |
| Stream transport | `content-type: text/event-stream` for `stream: true` |
| Upstream service | None. The endpoint uses the loaded local GGUF model (RD-1). |
| New dependency | None. The smoke tests use the SDK that the test environment provides. |

## 3. How to run

Start the local server with a loaded GGUF model. `--model-file` selects the
model, so the loaded model identifier is the full `--model-file` value. Do not
start a request that triggers a model download.

```bash
export LTENGINE_BASE_URL=http://127.0.0.1:5050/v1
export LTENGINE_API_KEY=<value of --api-key, empty when the server has no key>
export LTENGINE_MODEL=<loaded model identifier>
```

Both scripts read `LTENGINE_BASE_URL`, `LTENGINE_API_KEY`, and `LTENGINE_MODEL`.
Both scripts refuse a base-URL host that is not a local host, so a test cannot
call the OpenAI service by accident. Both scripts print the SDK version they
used. `LTENGINE_SDK_VERSION_ANY=1` bypasses the pinned-SDK-version check.

| Client | Script | Command |
| ------ | ------ | ------- |
| Python | `tests/responses_smoke.py` | `python tests/responses_smoke.py` |
| JavaScript | `tests/responses_smoke.mjs` | `node tests/responses_smoke.mjs` |

Both clients are synchronous. Both send two non-streaming requests: one text
`input` and one message-array `input`. Both verify the bearer credential, the
`object` value `response`, the `resp_*` identifier, the loaded model, and
`output_text`. Request 1 sends a `metadata` object, and both clients check the
`metadata` echo plus the required response fields `parallel_tool_calls`,
`tool_choice`, and `tools`. Exit code 0 means every check passed.

## 4. Support matrix: `SP-MUST` capabilities

Status values:

- `PASS (PH-1)`: the `PH-1` gate evidence covers the capability. The evidence is
  the manual `curl` checks of roadmap section 6. This profile did not re-run
  them.
- `PASS (PH-2)`: the `PH-2` live smoke-test evidence covers the capability. The
  evidence is the smoke-test run of section 7.
- `PASS (static)`: the capability is a static deliverable, and this profile is
  the observable evidence.
- `NOT RUN`: no test of this profile executed the check. This status is not a
  pass and not a failure.
- `FAIL`: a check ran and failed.

| Id | Capability | Applied test | Test status | Evidence |
| -- | ---------- | ------------ | ----------- | -------- |
| SP-MUST-001 | The route `POST /v1/responses` | Manual `curl` against the local server | `PASS (PH-1)` | Roadmap section 6 gate evidence |
| SP-MUST-002 | Text `instructions` | Manual `curl`; smoke test request 1 | `PASS (PH-1)` | Roadmap section 6 gate evidence |
| SP-MUST-003 | Text `input` | Manual `curl`; smoke test request 1 | `PASS (PH-1)` | Roadmap section 6 gate evidence |
| SP-MUST-004 | Message-array `input` with the standard text roles | Manual `curl`; smoke test request 2 | `PASS (PH-1)` | Roadmap section 6 gate evidence |
| SP-MUST-005 | Non-streaming generation | Manual `curl` without `stream` and with `stream: false` | `PASS (PH-1)` | Roadmap section 6 gate evidence. `PH-3` added SSE streaming for `stream: true`; section 9 of this file records it. |
| SP-MUST-006 | OpenAI-shaped response objects | Manual `curl`; smoke test `object` and `output_text` checks; inline unit test `response_shape_has_message_output_and_no_usage` | `PASS (PH-1)` | Roadmap section 6 gate evidence for `object`, `id`, `model`, and `output_text`. The required fields `parallel_tool_calls`, `tool_choice`, and `tools` have the static coverage of section 8 and the live correction run of section 8. |
| SP-MUST-007 | OpenAI-shaped validation and error responses | Manual `curl` with an invalid body, a body `api_key`, and a bad `model` | `PASS (PH-1)` | Roadmap section 6 gate evidence |
| SP-MUST-008 | `Authorization: Bearer` with the `--api-key` value | Smoke test wrong-key check; manual `curl` auth check | `PASS (PH-1)` | Roadmap section 6 gate evidence |
| SP-MUST-009 | The response identifies the loaded model | Smoke test `model` check; manual `curl` model echo check | `PASS (PH-1)` | Roadmap section 6 gate evidence |
| SP-MUST-010 | A unique `resp_*` identifier per response | Manual `curl` twice; inline unit test `response_ids_are_unique_and_prefixed` | `PASS (PH-1)` | Roadmap section 6 gate evidence |
| SP-MUST-011 | A clear error for an unsupported field, tool, and modality | Manual `curl` with an unsupported tool; inline unit tests for the unsupported tool, modality, and role | `PASS (PH-1)` | Roadmap section 6 gate evidence. `metadata` is an accepted field after the correction, and section 8 records that outcome and its live evidence. |
| SP-MUST-012 | The existing LibreTranslate API compatibility | Manual `curl` for each `AF-5` route | `PASS (PH-1)` | Roadmap section 6 gate evidence |
| SP-MUST-013 | A Python smoke test and a JavaScript smoke test | `tests/responses_smoke.py`, `tests/responses_smoke.mjs` | `PASS (PH-2)` | Live run: Python `openai` 3.14.1 on Python 3.13.5, exit 0; JavaScript `openai` 7.15.0 on Node v24.16.0, exit 0. Section 7 records the commands and the target. |
| SP-MUST-014 | A documented compatibility version and a support matrix | This file, section 1 through section 5 | `PASS (static)` | This file states version `1` and names every `SP-MUST` capability. |

No row of this matrix is `FAIL`. Rows `SP-MUST-001` through `SP-MUST-012` carry
`PH-1` gate evidence. Row `SP-MUST-013` carries `PH-2` live smoke-test evidence.
Row `SP-MUST-014` is a static deliverable. All 14 `SP-MUST` capabilities are
delivered: the first increment is complete.

## 5. Limits of the claim: `SP-NEVER` capabilities

Each entry below is a permanent product-boundary limit. It is part of the
compatibility claim.

| Id | Limit | How this profile respects it |
| -- | ----- | ---------------------------- |
| SP-NEVER-001 | Exact GPT model behavior and exact output equivalence | The endpoint runs the loaded local GGUF model. No output-equivalence claim exists. |
| SP-NEVER-002 | OpenAI model weights and proprietary model internals | LTEngine ships no OpenAI weight and no OpenAI internal. |
| SP-NEVER-003 | Hidden chain-of-thought and proprietary reasoning traces | The response carries the generated text only. No reasoning trace is exposed. |
| SP-NEVER-004 | Exact OpenAI tokenization for a local model | The profile makes no OpenAI-tokenization claim. The `usage` counts come from the loaded local-model tokenizer (RD-18). |
| SP-NEVER-005 | OpenAI billing, credits, pricing, and invoices | The profile makes no billing, credit, pricing, or invoice claim. |
| SP-NEVER-006 | OpenAI service tiers and capacity guarantees | The profile makes no tier and no capacity claim. `run_prompt` serializes inference, and this limit appears in section 6. |
| SP-NEVER-007 | OpenAI infrastructure, regions, and data-residency guarantees | The profile makes no infrastructure, region, or residency claim. The server is self-hosted. |
| SP-NEVER-008 | Exact OpenAI moderation and refusal behavior | The profile makes no moderation and no refusal-behavior claim. |
| SP-NEVER-009 | OpenAI-hosted tool implementations | The endpoint executes no tool. It returns a clear error for an unsupported tool. |
| SP-NEVER-010 | Fake usage, fake tool execution, and silent unsupported-field handling | `usage` carries the exact local-model token counts of RD-18, never an estimate. An unsupported field returns a clear error. No field is ignored silently. |
| SP-NEVER-011 | Byte-for-byte response parity | The profile claims no byte-for-byte parity. The client checks field values only. |
| SP-NEVER-012 | An unversioned full OpenAI parity claim | The claim is versioned: compatibility profile version `1`. Section 1 states the limits. |

## 6. Known limits inside the tested profile

1. `LLM::run_prompt` serializes inference. A concurrent request can fail with
   `LLMError::Busy` after 120 seconds. The profile makes no concurrency claim.
2. `usage` carries the exact token counts of RD-18. `input_tokens` is the
   length of the chat-templated token-ID vector passed to decode, beginning-of-
   sequence token included. `output_tokens` counts the emitted non-end-of-
   generation token IDs. The endpoint omits `usage` when an exact count is
   unavailable and never estimates (`SP-NEVER-010`).
3. `stream: true` returns `text/event-stream` in the RD-17 event order with a
   monotonically increasing `sequence_number`. The SSE body is built after the
   single generation attempt, so it carries one `response.output_text.delta`
   that holds the whole text, and every failure happens before the first event.
   Section 9 records the evidence.
4. Stored responses, `previous_response_id`, conversations, background work,
   structured output, and function calling are absent. They are `PH-3` through
   `PH-6` work. `SP-MUST-011` covers the clear-error outcome for their fields.
5. The existing LibreTranslate routes keep their flat error body
   `{"error": "<string>"}`. Only this route uses the nested error body.

## 7. Test status of this profile

`PH-2` gate result: `PASS`. Exit criterion 1 of roadmap section 7, "both smoke
tests run against a local server and exit 0", is met. The first increment is
complete. All 14 `SP-MUST` capabilities are delivered.

Live smoke-test evidence:

| Client | SDK and runtime | Command | Result |
| ------ | --------------- | ------- | ------ |
| Python | `openai` 3.14.1 on Python 3.13.5 | `python tests/responses_smoke.py` | `PASS`: exit 0 |
| JavaScript | `openai` 7.15.0 on Node v24.16.0 | `node tests/responses_smoke.mjs` | `PASS`: exit 0 |

The Python client ran in a temporary virtual environment that carried the pinned
`openai` 3.14.1. The default `python` interpreter carries `openai` 2.54.0 and
stays below the pinned target.

Fixture target:

| Field | Value |
| ----- | ----- |
| Base URL | `http://127.0.0.1:5051/v1` |
| API key | `ph1-test-key` |
| Loaded model | the full `--model-file` GGUF path `gemma-3-1b-it-q4_0.gguf` in the local Hugging Face cache |
| Generation mode | Non-streaming text only |

Each client checked the bearer rejection of a wrong key, the text `input`, the
message-array `input`, the `object` value `response`, the `resp_*` identifier,
the loaded model, and `output_text`. The loaded model value is the full model
file path.

JavaScript retry note. The first JavaScript attempt failed, because the `openai`
package did not export `package.json`, so the version lookup failed. The
approved single retry (`RD-12`) used a fixed version lookup and exited 0. `RD-12`
permits at most 2 total attempts; this run used both.

The other `PH-2` evidence:

| Check | Command | Result |
| ----- | ------- | ------ |
| Rust tests | `CARGO_NET_OFFLINE=true cargo test` | `PASS`: 20 tests passed, 0 failed. Section 8 records the correction that raised the count from 16. |
| Release build | `CARGO_NET_OFFLINE=true cargo build --release` | `PASS`: exit 0 |
| Python syntax | `python3 -m py_compile tests/responses_smoke.py` | `PASS`: exit 0 |
| JavaScript syntax | `node --check tests/responses_smoke.mjs` | `PASS`: exit 0 |
| Python local-host guard | `LTENGINE_BASE_URL=https://api.openai.com/v1 LTENGINE_MODEL=m python tests/responses_smoke.py` | `PASS`: exit 1 with `base URL host 'api.openai.com' is not a local host` |
| Python SDK-version guard | `LTENGINE_MODEL=m python tests/responses_smoke.py` | `PASS`: exit 1, because the default SDK is `openai` 2.54.0 and the pinned target is 3.14.1 |
| Server cleanup | Stop the local server after the smoke tests | `PASS`: the server is stopped and not listening |

The two guard rows confirm two properties without a network request: the script
refuses a non-local base URL, and the script refuses an SDK that does not match
the pinned target. The two live runs in the first table are the exit-criterion
evidence.

This profile claims no full OpenAI parity (`SP-NEVER-012`). The live evidence
covers the text scope of section 4 only.

## 8. Correction after `PH-2`: `metadata` and the required response fields

A compatibility correction landed after the `PH-2` gate. It changes two things:

1. `metadata`. The route accepts the OpenAI request field. Omitted and `null` are
   accepted. Otherwise the value is an object of at most 16 entries, each key a
   string of at most 64 characters, each value a string of at most 512
   characters. An accepted value is echoed in the response, and `null` appears
   when the request carried none. An invalid value returns the OpenAI-shaped
   HTTP 400 body, and the message names `metadata`. This replaces the earlier
   RD-3 outcome, which rejected `metadata`.
2. Required response fields. The success body carries `parallel_tool_calls`
   `false`, `tool_choice` `"none"`, and `tools` `[]`. These values are truthful
   for the text-only route: it offers no tool and calls no tool. An unsupported
   request field, an unsupported request `tools` value, and an unsupported
   request `tool_choice` value stay rejected.

Static evidence for the correction:

| Check | Command | Result |
| ----- | ------- | ------ |
| Rust tests | `CARGO_NET_OFFLINE=true cargo test` | `PASS`: 20 tests passed, 0 failed. The inline tests `accepts_absent_null_and_object_metadata`, `metadata_boundaries_are_inclusive`, `rejects_invalid_metadata_naming_metadata`, `rejects_tools_and_other_unsupported_fields`, and `response_echoes_metadata_or_null` cover the correction. |
| Rust formatting | `rustfmt --edition 2024 --check ltengine/src/responses.rs` | `PASS`: exit 0 |
| Python syntax | `python3 -m py_compile tests/responses_smoke.py` | `PASS`: exit 0 |
| JavaScript syntax | `node --check tests/responses_smoke.mjs` | `PASS`: exit 0 |

Live evidence for the correction:

| Client | SDK and runtime | Command | Result |
| ------ | --------------- | ------- | ------ |
| Python | `openai` 3.14.1 on Python 3.13.5 | `python tests/responses_smoke.py` | `PASS`: exit 0 on the first attempt |
| JavaScript | `openai` 7.15.0 on Node v24.16.0 | `node tests/responses_smoke.mjs` | `PASS`: exit 0 on the first attempt |

Both clients ran against the fixture base URL `http://127.0.0.1:5051/v1` with
the API key `ph1-test-key` and the full `--model-file` GGUF path as the loaded
model. Neither client used a retry. Each client sent `metadata` and checked the
`metadata` echo and the `metadata` `null` result, and each client checked
`parallel_tool_calls` `false`, `tool_choice` `"none"`, and `tools` `[]`. Each
client also checked the bearer rejection of a wrong key, which returns HTTP 401,
the text `input`, the message-array `input`, the `object` value `response`, the
`resp_*` identifier, the full loaded model path, and `output_text`. The server
was stopped after the run, and port 5051 is not listening.

The static checks above and both live runs give the correction a `PASS`. The
correction changes no decision outcome beyond the recorded `RD-3` outcome and
item 2 above. The live rows of section 7 stay the `PH-2` record of the earlier
scripts. Rows `SP-MUST-006` and `SP-MUST-011` keep their `PH-1` gate status; the
live correction run adds evidence to those two rows and changes no status.

### 8.1 Tool request fields (post-`PH-3` correction)

A second compatibility correction aligns the request contract with the OpenAI
tool defaults. Before this change the route rejected `tools`, `tool_choice`, and
`parallel_tool_calls` with HTTP 400, while the response body reported
`tools: []`, `tool_choice: "none"`, and `parallel_tool_calls: false`.

The corrected behavior:

1. `tools` is accepted when it is absent or empty. A non-empty array returns the
   OpenAI-shaped HTTP 400 body, and the message names `tools`.
2. `tool_choice` is accepted when it is absent, `"none"`, or `"auto"`. Any other
   value returns the OpenAI-shaped HTTP 400 body, and the message names
   `tool_choice`.
3. `parallel_tool_calls` is accepted as a boolean. It changes no behavior while
   the route offers no tool.

An empty `tools` array is the OpenAI default, so the no-tool forms match the
OpenAI request contract. Tool calling itself stays follow-up work
(`SP-PLANNED-004`, `SP-PLANNED-005`, roadmap `PH-4`). A request that needs a tool
returns a clear error instead of a silent ignore (`SP-MUST-011`,
`SP-NEVER-010`). Other unknown request fields and the unimplemented OpenAI
generation parameters (`temperature`, `top_p`, `max_output_tokens`) stay
rejected.

Static evidence for this correction:

| Check | Command | Result |
| ----- | ------- | ------ |
| Rust tests | `CARGO_NET_OFFLINE=true cargo test` | `PASS`: 25 tests passed, 0 failed. The inline tests `accepts_openai_tool_fields_without_a_tool_call`, `rejects_a_tool_request_naming_the_field`, and `rejects_other_unknown_fields` cover the correction. |
| Rust build | `CARGO_NET_OFFLINE=true cargo build --release` | `PASS`: exit 0, no warning |
| Live gate | `bin/lt ph1` | `PASS`: 60 checks ok, 0 mismatch, exit 0 |

This correction changes no other decision outcome. The `PH-2` rows of section 7
keep their status. The success body is unchanged, so the live `tools` `[]` check
of section 8 stays valid.

## 9. `PH-3`: SSE streaming and token usage

The `PH-3` gate result is `PASS`. The phase delivers `SP-PLANNED-001` (SSE text
streaming) and `SP-PLANNED-002` (real token usage). `RD-17` fixes the event set
and the event order, `RD-13` fixes the attempt policy, and `RD-18` fixes the
count source.

Delivered:

1. `stream: true` on `POST /v1/responses` returns `content-type:
   text/event-stream` with the RD-17 order `response.created`,
   `response.in_progress`, `response.output_item.added`,
   `response.content_part.added`, zero or more `response.output_text.delta`,
   `response.output_text.done`, `response.content_part.done`,
   `response.output_item.done`, and `response.completed`. Each payload carries
   its event `type` and a monotonically increasing integer `sequence_number`.
2. One generation attempt and no retry, no reconnect, and no `Last-Event-ID`
   behavior (`RD-13`). Replay stays `PH-5` work.
3. Every failure before the first event uses the normal OpenAI-shaped HTTP error
   body.
4. A `usage` object on the non-streaming body and on the `response.completed`
   event with `input_tokens`, `output_tokens`, and the exact `total_tokens` sum.
   The counts come from the loaded local-model tokenizer (`RD-18`). The endpoint
   omits `usage` when an exact count is unavailable and never estimates
   (`SP-NEVER-010`).

Live evidence, local fixture `http://127.0.0.1:5051/v1`, key `ph1-test-key`,
loaded model = the full `--model-file` GGUF path:

| Check | Command or request | Result |
| ----- | ------------------ | ------ |
| Stream transport | `curl -N` with `stream: true` | `PASS`: HTTP 200 with `content-type: text/event-stream` |
| Event order | Parse the `event:` lines of the stream body | `PASS`: the nine RD-17 events in order, one delta, sequence numbers `0` through `8`, and every payload `type` equals its `event:` line |
| Stream usage | `usage` of the `response.completed` event | `PASS`: `input_tokens` 15, `output_tokens` 3, `total_tokens` 18 |
| Non-streaming usage | `curl` with no `stream` and with `stream: false` | `PASS`: JSON body with the same `15`/`3`/`18` |
| Exact sum | `input_tokens + output_tokens == total_tokens` | `PASS` for both responses |
| Prompt-length growth | A longer `instructions` and `input` | `PASS`: `input_tokens` 36, `output_tokens` 73, `total_tokens` 109 |
| Pre-first-event failure | `stream: true` with a bad `model`, a wrong bearer token, a body `api_key`, and a malformed body | `PASS`: HTTP 400 or 401 with the OpenAI-shaped JSON error, and no `text/event-stream` |
| Unit tests | `CARGO_NET_OFFLINE=true cargo test` | `PASS`: 23 tests passed, 0 failed. The tests `stream_body_follows_the_rd17_order_with_increasing_sequence_numbers`, `stream_body_sets_the_content_type_and_echoes_metadata`, `accepts_stream_true_and_selects_the_sse_body`, `response_shape_has_message_output_and_usage`, and `usage_total_is_the_exact_sum_and_eog_is_not_counted` cover the new shape. |
| Release build | `CARGO_NET_OFFLINE=true cargo build --release` | `PASS`: exit 0 |
| MTP pair load | Server with `--mtp-model-file` and `--mtp-n-max 2`, target file `gemma-4-E2B_q4_0-it.gguf`, draft file `gemma-4-E2B-it-qat-assistant-MTP-Q8_0.gguf` | `PASS`: log `ltengine: MTP draft model loaded`, then HTTP 200 |
| MTP draft acceptance | Server log of the same request | `PASS`: `ltengine: MTP proposed 34 tokens, accepted 34`. The accepted total is above 0, so the accepted draft-token counting ran |
| MTP output and usage equality | The same request against the target-only server and against the MTP server | `PASS`: both return the text `1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20`, and both carry `input_tokens` 53, `output_tokens` 50, and `total_tokens` 103 |
| MTP stream parity | `stream: true` against the MTP server | `PASS`: the RD-17 order with sequence numbers `0` through `8`, one delta that holds the whole text, and the same `53`/`50`/`103` usage in `response.completed` |

The MTP rows use this request:

```json
{"model":"/root/.cache/ltengine-ph3-mtp/gemma-4-E2B_q4_0-it.gguf","instructions":"You are a helpful assistant.","input":"Write the integers from 1 to 20 separated by single spaces. Output only the numbers."}
```

The MTP rows use this model pair:

| Role | Path | Byte size |
| ---- | ---- | --------- |
| Target | `/root/.cache/ltengine-ph3-mtp/gemma-4-E2B_q4_0-it.gguf` | 3,349,516,256 |
| Draft | `/root/.cache/ltengine-ph3-mtp/gemma-4-E2B-it-qat-assistant-MTP-Q8_0.gguf` | 97,835,456 |

The MTP rows use a different request and a different model from the rows above
them. Compare each row with its own request only. The `output_tokens` value of
50 holds the accepted draft tokens, because the MTP server counted them
(`RD-18`).

Limits of this record:

1. The SSE body is built after the single generation attempt, so it carries one
   delta that holds the whole text. RD-17 permits zero or more delta events.
   This profile does not claim incremental token delivery.
2. No independent tokenizer implementation is available offline, so the counts
   were checked by the exact-sum identity, by the agreement of the streamed and
   the non-streaming request on the same prompt, and by prompt-length growth,
   not against a second tokenizer.
3. `response.failed` is not emitted, because every failure happens before the
   first event (`RD-13`).

This profile claims no full OpenAI parity (`SP-NEVER-012`). The stream claim
covers the text scope of section 4 only.

## 10. `PH-4a`: structured output

The `PH-4a` change delivers `SP-PLANNED-003` (structured JSON output with
schema validation). The OpenAI Responses API contract is the reference
(`RD-19`). The route accepts `text.format` with the types `text`, `json_object`,
and `json_schema`.

| Field | Behavior |
| ----- | -------- |
| `text.format.type` | `text` (default), `json_object`, or `json_schema`. Any other value is a 400 that names `text.format.type`. |
| `json_object` | The decode is constrained to valid JSON. `output_text` parses as a JSON value. |
| `json_schema` | Requires `name` (1 to 64 characters of `a-z`, `A-Z`, `0-9`, `_`, `-`) and `schema` (a JSON object). |
| `json_schema.strict: true` | The schema becomes a GBNF grammar through `llama_cpp_2::json_schema_to_grammar`, and the sampler is constrained with it. The output conforms to the schema. |
| `json_schema.strict` omitted or `false` | Best effort. The output must still parse as JSON, and a non-conforming output is a 500. |
| `text.verbosity` | Part of the OpenAI contract, not implemented. A present value is a 400 that names `text.verbosity`, never a silent ignore (`SP-NEVER-010`). |
| Invalid or unsupported schema | A 400 that names `text.format` or `text.format.schema`, matching the OpenAI rejection of an unsupported schema. |
| Non-conforming generated output | The OpenAI-shaped HTTP 500 `server_error` error. The route never returns malformed JSON as a valid `output_text`, and it makes one generation attempt (`RD-14`). |

Live evidence, local fixture `http://127.0.0.1:5051/v1`, key `ph1-test-key`,
loaded model = the full `--model-file` GGUF path:

| Check | Result |
| ----- | ------ |
| `json_object` request | `PASS`: HTTP 200 and `output_text` parsed as a JSON object |
| strict `json_schema` request | `PASS`: HTTP 200 and `output_text` parsed to `{"answer": "Paris"}`, which conforms to the schema |
| Unknown `text.format.type` | `PASS`: HTTP 400 naming `text.format.type` |
| `json_schema` without `schema` | `PASS`: HTTP 400 naming `text.format` |
| Unsupported schema | `PASS`: HTTP 400 naming `text.format.schema` |
| `text.verbosity` | `PASS`: HTTP 400 naming `text.verbosity` |
| Non-strict non-conforming output | `PASS`: HTTP 500 `server_error` |
| Unit tests | `PASS`: `CARGO_NET_OFFLINE=true cargo test` exits 0 with 32 passed and 0 failed |
| Release build | `PASS`: `CARGO_NET_OFFLINE=true cargo build --release` exits 0 |
| Preserved behavior | `PASS`: `bin/lt ph1` reports 60 ok of 60, so the `RD-17` stream order, the error cases, and the legacy routes are unchanged |

Limits of this record:

1. A `refusal` content part is not implemented. Model refusal behavior is
   outside the permanent limits (`SP-NEVER-001`), so a local model refusal is
   ordinary `output_text`.
2. The route does not enforce the OpenAI strict-mode schema requirements that
   the converter accepts, such as `additionalProperties: false` on every object.
   The converter's rejection set is the route's unsupported-schema set.
3. `json_schema` with `strict` omitted defaults to non-strict. The OpenAI guides
   state no default for `text.format`.
4. The SDK smoke tests of section 3 do not cover `SP-PLANNED-003` yet.

`SP-PLANNED-004` and `SP-PLANNED-005` stay with `PH-4b` and are not delivered.
This profile still claims no full OpenAI parity (`SP-NEVER-012`).

## 11. `PH-4b`: function calling

The `PH-4b` change delivers `SP-PLANNED-004` (custom function calling) and
`SP-PLANNED-005` (`tool_choice` and parallel calls). The OpenAI Responses API
tool contract is the reference (`RD-20`).

| Field | Behavior |
| ----- | -------- |
| `tools` | A `function` tool is `{type, name, parameters, description?, strict?}`. A tool whose `type` is not `function` is a 400 that names `tools`. |
| `strict: true` | The tool `arguments` are constrained to the tool `parameters` schema. |
| `strict` omitted | The route attempts strict normalization and falls back to non-strict. The body echoes the effective `strict` value. |
| `strict: false` | Best effort; the body echoes `strict: false`. |
| `tool_choice` | `none`, `auto`, `required`, or `{type:"function", name}`. `required` needs at least one tool. A named function must be declared, or the route returns a 400 that names `tool_choice`. |
| `parallel_tool_calls` | `true` permits several `function_call` items; `false` permits at most one. |
| Output | One or more `function_call` items: `type`, `id` (`fc_*`), `call_id` (`call_*`), `name`, `arguments` (a JSON string), `status`. |
| `input` items | `function_call` and `function_call_output` are accepted so a client can close a tool loop in one request. |
| Stream | `response.output_item.added`, zero or more `response.function_call_arguments.delta`, `response.function_call_arguments.done`, `response.output_item.done`, inside the `RD-17` lifecycle. |
| Execution | The route executes no tool and calls no upstream service (`SP-NEVER-009`). The client executes the call. |

Live evidence, local fixture `http://127.0.0.1:5051/v1`, key `ph1-test-key`,
loaded model = the full `--model-file` GGUF path:

| Check | Result |
| ----- | ------ |
| `tool_choice: auto` with a `get_weather` tool | `PASS`: HTTP 200 and one `function_call` item with `fc_*` id, `call_*` call_id, `name` `get_weather`, and `arguments` `{"location":"Paris"}` |
| `tool_choice: required` | `PASS`: at least one `function_call` item |
| `tool_choice: {type:"function", name}` | `PASS`: the named function call |
| Tool `type` `web_search` | `PASS`: HTTP 400 naming `tools` |
| Named function that is not declared | `PASS`: HTTP 400 naming `tool_choice` |
| `stream: true` tool request | `PASS`: `text/event-stream` with the tool order, `sequence_number` from 0, and an added item with empty `arguments` |
| Tool loop with `function_call` and `function_call_output` input | `PASS`: HTTP 200 and a message that used the tool result |
| Unit tests | `PASS`: `CARGO_NET_OFFLINE=true cargo test` exits 0 with 40 passed and 0 failed |
| Release build | `PASS`: `CARGO_NET_OFFLINE=true cargo build --release` exits 0 |
| Preserved behavior | `PASS`: `bin/lt ph1` reports 60 ok of 60 |

Limits of this record:

1. A request that combines `tools` with a structured `text.format` is rejected
   with a 400 that names `text.format`. The two grammars are not merged.
2. `parallel_tool_calls: true` permits several calls; whether the model emits
   more than one depends on the model and the prompt.
3. The transcription envelope is LTEngine-defined: `{"message": "..."}` or
   `{"calls": [...]}`. It is not an OpenAI wire format, and the model must be
   able to follow it.

`SP-PLANNED-003` is delivered by `PH-4a` (section 10). This profile still claims
no full OpenAI parity (`SP-NEVER-012`).

## 12. `PH-5` and `PH-5a`: stored responses, retrieval, and replay

`PH-5` delivers `SP-PLANNED-006` through `SP-PLANNED-009`. The route family now
stores a response by default, chains with `previous_response_id`, and answers
retrieval, deletion, input-item listing, and replay. `PH-5a` adds replay under
`RD-27`.

| Behavior | Contract |
| -------- | -------- |
| `store` | Boolean, default `true`. `true` stores the completed response. `false` does not, so it is not retrievable. |
| `previous_response_id` | Prepends the referenced record's input items, then its output items as input items, then the request input. The referenced `instructions` are not carried. Chaining is transitive. |
| `GET /v1/responses/{response_id}` | HTTP 200 with the stored response object, equal to the created body. |
| `GET /v1/responses/{response_id}?stream=true` | HTTP 200 with `text/event-stream`, replaying the stored `RD-17` event sequence. Deterministic regeneration; no model call. `starting_after=N` keeps only the events with `sequence_number > N` and preserves their numbers. |
| `DELETE /v1/responses/{response_id}` | HTTP 200 with `{"id":"<id>","object":"response","deleted":true}`. |
| `GET /v1/responses/{response_id}/input_items` | HTTP 200 with `{"object":"list","data":[...],"first_id":...,"last_id":...,"has_more":false}`. |
| Unknown or deleted id | HTTP 404 with `{"error":{"message":"No response found with id '<id>'","type":"invalid_request_error","param":null,"code":null}}`, also with `stream=true`. |
| Unsupported query field | HTTP 400 that names the field (`CC-6`). `stream` and `starting_after` are supported. |
| Store layout | One JSON file per response at `<store-dir>/<resp_id>.json`. `--store-dir` defaults to `./ltengine-responses` and is created at startup. |
| Write failure | One attempt, fail-closed with HTTP 500 `server_error` (`RD-15`). |

Live evidence, local fixture `http://127.0.0.1:5051`, key `ph1-test-key`,
loaded model = the full `--model-file` GGUF path:

| Check | Result |
| ----- | ------ |
| Create then retrieve | `PASS`: `GET` returns a body equal to the created body |
| Input-item list | `PASS`: `object` `list`, one item for a string input, `has_more` `false` |
| `previous_response_id` | `PASS`: the chained record stores three items in order (first input, first output as an assistant message, new input) |
| `store:false` | `PASS`: HTTP 200 on create, HTTP 404 on `GET` |
| Unknown id | `PASS`: HTTP 404 with the body above |
| `previous_response_id` unknown | `PASS`: HTTP 404 with the body above |
| `DELETE` then `GET` | `PASS`: `{"id","object":"response","deleted":true}`, then HTTP 404 |
| `DELETE` unknown | `PASS`: HTTP 404 with the body above |
| Traversal-shaped id | `PASS`: HTTP 404, no path outside the store |
| Replay equals the create-time stream | `PASS`: event-for-event equality, `sequence_number` `0..n-1` |
| `starting_after=4` | `PASS`: events with `sequence_number` 5 through 8, unchanged |
| `stream=false` | `PASS`: the stored JSON body |
| `stream=maybe` | `PASS`: HTTP 400 naming `stream` |
| `starting_after` without `stream=true` | `PASS`: HTTP 400 naming `starting_after` |
| Unknown id with `stream=true` | `PASS`: HTTP 404 with the body above |
| Missing bearer | `PASS`: HTTP 401 |
| Unit tests | `PASS`: `CARGO_NET_OFFLINE=true cargo test` exits 0 with 75 passed and 0 failed |
| Release build | `PASS`: `CARGO_NET_OFFLINE=true cargo build --release` exits 0 |
| Preserved behavior | `PASS`: `bin/lt ph1` reports 60 ok of 60 |

Limits of this record:

1. The input-item list has no pagination and no ordering parameter.
2. Retention limits and storage limits stay `RD-9` and `RD-25` (`PH-6`).
3. Replay accepts `stream` and `starting_after` only. `include` and
   `include_obfuscation` return HTTP 400 that names the field.

## 13. `PH-6a`: conversations and conversation items

`PH-6a` delivers `SP-PLANNED-010` under `RD-23`. It adds the OpenAI
Conversations routes and the `conversation` request field.

| Behavior | Contract |
| -------- | -------- |
| `POST /v1/conversations` | HTTP 200 with a `conv_*` object: `object` `conversation`, `created_at`, and `metadata` (`null` when absent). |
| `GET /v1/conversations/{id}` | HTTP 200 with the object; an unknown id returns the conversation 404 body. |
| `POST /v1/conversations/{id}` | HTTP 200 with the updated object; `metadata` is replaced. |
| `DELETE /v1/conversations/{id}` | HTTP 200 with `{"id","object":"conversation.deleted","deleted":true}`. |
| `POST .../items` | HTTP 200 with the created items; an item without an `id` receives `msg_`, `fc_`, or `fco_` by shape. |
| `GET .../items` | HTTP 200 with `{"object":"list","data":[...],"first_id":...,"last_id":...,"has_more":false}` in insertion order. |
| `GET .../items/{item_id}` | HTTP 200 with the item; an unknown item returns the item 404 body. |
| `DELETE .../items/{item_id}` | HTTP 200 with `{"id","object":"conversation.item.deleted","deleted":true}`. |
| `conversation` on `POST /v1/responses` | Prepends the conversation items, echoes `conversation` `{"id":...}`, and appends the request input items and the output items. `conversation` with `previous_response_id` is a 400. |
| Item limit | `--max-conversation-items` default `100`, `0` disables. An add beyond the limit is HTTP 400 naming the limit. The response path checks the bound before generation. |
| Retention | `--retention-secs` default `86400`, `0` disables. An expired conversation returns the conversation 404 body and is removed lazily. |
| Credential and query guard | Every route enforces the `RD-4` bearer token and returns HTTP 400 that names an unsupported query field. |

Live evidence, local fixture `http://127.0.0.1:5051`, key `ph1-test-key`:

| Check | Result |
| ----- | ------ |
| Create, retrieve, update, delete | `PASS` |
| Add items, list order, get item, delete item | `PASS` |
| Response with `conversation` | `PASS`: `conversation` echo and the appended input and output items |
| `conversation` with `previous_response_id` | `PASS`: HTTP 400 naming `conversation` |
| Unknown conversation and item | `PASS`: the declared 404 bodies |
| Missing bearer and query field | `PASS`: HTTP 401 and HTTP 400 naming the field |
| Unit tests | `PASS`: `CARGO_NET_OFFLINE=true cargo test` exits 0 with 89 passed and 0 failed |
| Release build | `PASS`: `CARGO_NET_OFFLINE=true cargo build --release` exits 0 |
| Preserved behavior | `PASS`: `bin/lt ph1` reports 60 ok of 60 |

Limits of this record:

1. The item list has no pagination and no ordering parameter.
2. `POST .../items` stores items as sent. A stored item with an unsupported
   content part is skipped when the prompt is built.
3. The item limit and the retention boundary are unit checks. The live run uses
   the default flags.
