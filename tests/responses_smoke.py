"""OpenAI Python SDK smoke test for the local `POST /v1/responses` route.

Roadmap PH-2 smoke test. Target SDK: `openai` 3.14.1 on Python >= 3.10 (RD-8).
Command: `python tests/responses_smoke.py`.

The client is synchronous. The script sends no streaming request. It calls the
local LTEngine server only; it never calls the OpenAI service (RD-1). It adds no
dependency: it uses the `openai` package that the test environment provides.

Environment:

- `LTENGINE_BASE_URL`: base URL of the local server. Default
  `http://127.0.0.1:5050/v1`. The host MUST be a local host.
- `LTENGINE_API_KEY`: value of the server `--api-key` option. Empty when the
  server runs without a key.
- `LTENGINE_MODEL`: identifier of the loaded model. The request `model` field
  and the response `model` field MUST equal this value (SP-MUST-009, RD-5).
- `LTENGINE_SDK_VERSION_ANY`: value `1` bypasses the pinned-SDK-version check.

Both requests also check the required response fields of the text-only route
(`parallel_tool_calls`, `tool_choice`, `tools`) and the `metadata` echo. Request
1 sends `metadata`; request 2 omits it and the response MUST carry `null`.

The script also covers the planned items of `PH-4` through the SDK:
`text.format` `json_object` and strict `json_schema` (`SP-PLANNED-003`), a
forced function call (`SP-PLANNED-004`), and a named `tool_choice` with
`parallel_tool_calls: true` (`SP-PLANNED-005`). Those requests run against the
loaded local model, so the generated text is not part of the contract. The
checks assert the response shape, the identifier prefixes, and JSON validity,
and they never assert a particular generated value.

The script also covers the stored and long-running planned items through the
SDK: a stored response, `previous_response_id`, retrieval, deletion, and the
input-item list (`SP-PLANNED-006` through `SP-PLANNED-009`); conversations and
conversation items (`SP-PLANNED-010`); and a background response with polling
and cancellation (`SP-PLANNED-011`). Those checks create server-side records and
delete the records that they create.

The script also covers the streamed and counted planned items through the SDK:
the `RD-17` event order over `client.responses.stream(...)` (`SP-PLANNED-001`)
and the `usage` counts of a non-streaming response (`SP-PLANNED-002`).

The script also covers the `reasoning` request field (`SP-PLANNED-013`): a
supported `effort` value is accepted, and an unsupported one is a 400 that
names the field. The loaded model's template decides what the value changes, so
no check asserts a particular generated value.

Exit code 0 means every check passed.
"""

import json
import os
import sys
import time
from urllib.parse import urlparse

PINNED_SDK = "3.14.1"
DEFAULT_BASE_URL = "http://127.0.0.1:5050/v1"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
SMOKE_METADATA = {"suite": "responses-smoke"}

PLANNED_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
        "additionalProperties": False,
    },
}

CAPITAL_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
DECLARED_STATUSES = {"queued", "in_progress"} | TERMINAL_STATUSES
BACKGROUND_PROMPT = "Count from 1 to 300, one number per line."
EXPECTED_STREAM_TYPES = [
    "response.created",
    "response.in_progress",
    "response.output_item.added",
    "response.content_part.added",
    "response.output_text.done",
    "response.content_part.done",
    "response.output_item.done",
    "response.completed",
]

BASE_URL = os.environ.get("LTENGINE_BASE_URL", DEFAULT_BASE_URL)
API_KEY = os.environ.get("LTENGINE_API_KEY", "")
MODEL = os.environ.get("LTENGINE_MODEL", "").strip()

FAILURES = []


def report(label, status, detail=""):
    print(f"{status} {label}" + (f" - {detail}" if detail else ""))
    if status == "FAIL":
        FAILURES.append(label)


def config_error(message):
    print(f"FAIL config: {message}")
    return 1


def verify(label, response, metadata=None):
    """Check the OpenAI-shaped success contract of PH-1 (SP-MUST-006, SP-MUST-009, SP-MUST-010)."""
    value = getattr(response, "object", None)
    report(
        f"{label}: object is `response`",
        "PASS" if value == "response" else "FAIL",
        f"object={value!r}",
    )

    identifier = getattr(response, "id", None)
    report(
        f"{label}: id starts with `resp_`",
        "PASS" if str(identifier).startswith("resp_") else "FAIL",
        f"id={identifier!r}",
    )

    echoed = getattr(response, "model", None)
    report(
        f"{label}: model is the loaded model",
        "PASS" if echoed == MODEL else "FAIL",
        f"model={echoed!r} expected={MODEL!r}",
    )

    text = getattr(response, "output_text", None) or ""
    report(
        f"{label}: output_text is present",
        "PASS" if text.strip() else "FAIL",
        f"output_text={text!r}",
    )

    echoed = getattr(response, "metadata", None)
    report(
        f"{label}: metadata is echoed",
        "PASS" if echoed == metadata else "FAIL",
        f"metadata={echoed!r} expected={metadata!r}",
    )

    parallel = getattr(response, "parallel_tool_calls", None)
    report(
        f"{label}: parallel_tool_calls is false",
        "PASS" if parallel is False else "FAIL",
        f"parallel_tool_calls={parallel!r}",
    )

    tool_choice = getattr(response, "tool_choice", None)
    report(
        f"{label}: tool_choice is `none`",
        "PASS" if tool_choice == "none" else "FAIL",
        f"tool_choice={tool_choice!r}",
    )

    tools = getattr(response, "tools", None)
    report(
        f"{label}: tools is empty",
        "PASS" if tools == [] else "FAIL",
        f"tools={tools!r}",
    )

    verify_usage(label, response)


def verify_usage(label, response):
    """Check SP-PLANNED-002: `usage` carries real counts and the exact sum (RD-18)."""
    usage = getattr(response, "usage", None)
    if usage is None:
        report(f"{label}: usage is present", "FAIL", "usage is null")
        return
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    total = getattr(usage, "total_tokens", None)
    positive = all(
        isinstance(value, int) and value > 0
        for value in (input_tokens, output_tokens, total)
    )
    report(
        f"{label}: usage carries positive integer counts",
        "PASS" if positive else "FAIL",
        f"usage={input_tokens}/{output_tokens}/{total}",
    )
    report(
        f"{label}: total_tokens is the exact sum",
        "PASS" if total == input_tokens + output_tokens else "FAIL",
        f"total={total} sum={input_tokens}+{output_tokens}",
    )


def verify_stream(client):
    """Check SP-PLANNED-001: the SDK stream carries the RD-17 event order."""
    try:
        with client.responses.stream(model=MODEL, input="Say OK") as stream:
            events = [
                (event.type, getattr(event, "sequence_number", None)) for event in stream
            ]
            final = stream.get_final_response()
    except Exception as err:  # noqa: BLE001 - any error is a failed check
        report("streaming: the SDK stream completes", "FAIL", f"{type(err).__name__}: {err}")
        return

    collected = [event_type for event_type, _ in events]
    report("streaming: the SDK stream completes", "PASS", f"events={len(events)}")
    non_delta = [event_type for event_type in collected if event_type != "response.output_text.delta"]
    report(
        "streaming: the RD-17 event order",
        "PASS" if non_delta == EXPECTED_STREAM_TYPES else "FAIL",
        f"types={collected}",
    )
    numbers = [number for _, number in events]
    increasing = all(later > earlier for earlier, later in zip(numbers, numbers[1:]))
    report(
        "streaming: sequence_number increases",
        "PASS" if increasing else "FAIL",
        f"sequence_numbers={numbers}",
    )
    text = getattr(final, "output_text", None) or ""
    report(
        "streaming: the final response carries output_text",
        "PASS" if getattr(final, "object", None) == "response" and text.strip() else "FAIL",
        f"object={getattr(final, 'object', None)!r}",
    )


def request(label, call):
    """Run one request. Report a FAIL and return `None` when the SDK raises."""
    try:
        return call()
    except Exception as err:  # noqa: BLE001 - any error is a failed check
        report(label, "FAIL", f"{type(err).__name__}: {err}")
        return None


def function_calls(response):
    output = getattr(response, "output", None) or []
    return [item for item in output if getattr(item, "type", None) == "function_call"]


def verify_json_text(label, response, schema_key=None):
    """Check SP-PLANNED-003: `output_text` parses as JSON, and conforms to the schema."""
    if response is None:
        return
    text = getattr(response, "output_text", None) or ""
    try:
        parsed = json.loads(text)
    except ValueError as err:
        report(
            f"{label}: output_text parses as JSON",
            "FAIL",
            f"output_text={text!r} error={err}",
        )
        return
    report(f"{label}: output_text parses as JSON", "PASS", f"value={parsed!r}")
    if schema_key is None:
        return
    conforms = isinstance(parsed, dict) and isinstance(parsed.get(schema_key), str)
    report(
        f"{label}: output conforms to the schema",
        "PASS" if conforms else "FAIL",
        f"value={parsed!r}",
    )


def verify_function_call(label, response, expected_name):
    """Check SP-PLANNED-004 and SP-PLANNED-005: the `function_call` item fields."""
    if response is None:
        return
    calls = function_calls(response)
    report(
        f"{label}: at least one function_call item",
        "PASS" if calls else "FAIL",
        f"count={len(calls)}",
    )
    for index, item in enumerate(calls):
        call_label = f"{label}: call {index}"
        identifier = getattr(item, "id", None)
        report(
            f"{call_label} id starts with `fc_`",
            "PASS" if str(identifier).startswith("fc_") else "FAIL",
            f"id={identifier!r}",
        )
        call_id = getattr(item, "call_id", None)
        report(
            f"{call_label} call_id starts with `call_`",
            "PASS" if str(call_id).startswith("call_") else "FAIL",
            f"call_id={call_id!r}",
        )
        name = getattr(item, "name", None)
        report(
            f"{call_label} name is {expected_name!r}",
            "PASS" if name == expected_name else "FAIL",
            f"name={name!r}",
        )
        arguments = getattr(item, "arguments", None)
        try:
            json.loads(arguments)
            parsed = True
        except (TypeError, ValueError):
            parsed = False
        report(
            f"{call_label} arguments parse as JSON",
            "PASS" if parsed else "FAIL",
            f"arguments={arguments!r}",
        )


def expect_not_found(label, call):
    """Check that a request is rejected with the OpenAI-shaped HTTP 404 (RD-22)."""
    try:
        call()
    except Exception as err:  # noqa: BLE001 - the rejection is the expected outcome
        status = getattr(err, "status_code", None)
        report(label, "PASS" if status == 404 else "FAIL", f"{type(err).__name__}: {err}")
        return
    report(label, "FAIL", "the request succeeded")


def verify_reasoning(client):
    """`SP-PLANNED-013`: the `reasoning` request field.

    The loaded model's template decides what the effort value changes, so the
    check asserts the request contract only: a supported value is accepted, and
    an unsupported value is a 400 that names `reasoning.effort`.
    """
    accepted = request(
        "reasoning effort accepted",
        lambda: client.responses.create(
            model=MODEL,
            input="Say OK.",
            reasoning={"effort": "low"},
        ),
    )
    if accepted is not None:
        response_id = getattr(accepted, "id", None)
        report(
            "reasoning effort: the response carries a resp_ id",
            "PASS" if str(response_id).startswith("resp_") else "FAIL",
            f"id={response_id!r}",
        )

    try:
        client.responses.create(
            model=MODEL, input="Say OK.", reasoning={"effort": "enormous"}
        )
    except Exception as err:  # noqa: BLE001 - the rejection is the check
        status = getattr(err, "status_code", None)
        report(
            "reasoning effort: an unsupported value is a 400",
            "PASS" if status == 400 else "FAIL",
            f"status={status!r}",
        )
        report(
            "reasoning effort: the rejection names reasoning.effort",
            "PASS" if "reasoning.effort" in str(err) else "FAIL",
            str(err).splitlines()[0][:100],
        )
    else:
        report(
            "reasoning effort: an unsupported value is a 400",
            "FAIL",
            "the request succeeded",
        )


def verify_stored_flow(client):
    """Check SP-PLANNED-006 through SP-PLANNED-009."""
    stored = request(
        "stored response",
        lambda: client.responses.create(model=MODEL, input="Say OK", store=True),
    )
    if stored is None:
        return
    report(
        "stored response: id starts with `resp_`",
        "PASS" if str(stored.id).startswith("resp_") else "FAIL",
        f"id={stored.id!r}",
    )

    retrieved = request(
        "stored response: retrieve", lambda: client.responses.retrieve(stored.id)
    )
    if retrieved is not None:
        report(
            "stored response: retrieve returns the same id",
            "PASS" if retrieved.id == stored.id else "FAIL",
            f"id={retrieved.id!r}",
        )

    items = request(
        "stored response: input items",
        lambda: client.responses.input_items.list(stored.id),
    )
    if items is not None:
        data = getattr(items, "data", None) or []
        listed = any(getattr(item, "role", None) == "user" for item in data)
        report(
            "stored response: the input item is listed",
            "PASS" if listed else "FAIL",
            f"count={len(data)}",
        )

    chained = request(
        "previous_response_id",
        lambda: client.responses.create(
            model=MODEL,
            input="Say OK again",
            previous_response_id=stored.id,
            store=True,
        ),
    )
    if chained is not None:
        report(
            "previous_response_id: a new `resp_` id",
            "PASS"
            if str(chained.id).startswith("resp_") and chained.id != stored.id
            else "FAIL",
            f"id={chained.id!r}",
        )

    request("stored response: delete", lambda: client.responses.delete(stored.id))
    expect_not_found(
        "deleted response: retrieve is not found",
        lambda: client.responses.retrieve(stored.id),
    )
    if chained is not None:
        request(
            "previous_response_id: delete the chained response",
            lambda: client.responses.delete(chained.id),
        )


def verify_conversation_flow(client):
    """Check SP-PLANNED-010."""
    conversation = request("conversation: create", lambda: client.conversations.create())
    if conversation is None:
        return
    report(
        "conversation: id starts with `conv_`",
        "PASS" if str(conversation.id).startswith("conv_") else "FAIL",
        f"id={conversation.id!r}",
    )

    request(
        "conversation: add an item",
        lambda: client.conversations.items.create(
            conversation.id,
            items=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Say hi"}],
                }
            ],
        ),
    )

    items = request(
        "conversation: list items",
        lambda: client.conversations.items.list(conversation.id),
    )
    if items is not None:
        data = getattr(items, "data", None) or []
        listed = any(getattr(item, "role", None) == "user" for item in data)
        report(
            "conversation: the added item is listed",
            "PASS" if listed else "FAIL",
            f"count={len(data)}",
        )

    response = request(
        "conversation: response with the field",
        lambda: client.responses.create(
            model=MODEL, input="Say OK", conversation=conversation.id
        ),
    )
    if response is not None:
        report(
            "conversation: the response carries a `resp_` id",
            "PASS" if str(response.id).startswith("resp_") else "FAIL",
            f"id={response.id!r}",
        )

    deleted = request(
        "conversation: delete", lambda: client.conversations.delete(conversation.id)
    )
    if deleted is not None:
        shape = getattr(deleted, "object", None)
        report(
            "conversation: delete reports `conversation.deleted`",
            "PASS" if shape == "conversation.deleted" else "FAIL",
            f"object={shape!r}",
        )


def verify_background_flow(client):
    """Check SP-PLANNED-011."""
    job = request(
        "background: create",
        lambda: client.responses.create(
            model=MODEL, input="Say OK", background=True, store=True
        ),
    )
    if job is None:
        return
    report(
        "background: the created response is `queued`",
        "PASS" if job.status == "queued" else "FAIL",
        f"status={job.status!r}",
    )

    seen = []
    final = job
    for _ in range(120):
        final = request("background: poll", lambda: client.responses.retrieve(job.id))
        if final is None:
            break
        seen.append(final.status)
        if final.status in TERMINAL_STATUSES:
            break
        time.sleep(0.25)
    report(
        "background: the job reaches a terminal status",
        "PASS" if final is not None and final.status in TERMINAL_STATUSES else "FAIL",
        f"status={getattr(final, 'status', None)!r}",
    )
    report(
        "background: every observed status is declared",
        "PASS" if set(seen) <= DECLARED_STATUSES else "FAIL",
        f"observed={seen}",
    )

    target = request(
        "background cancel: create",
        lambda: client.responses.create(
            model=MODEL, input=BACKGROUND_PROMPT, background=True, store=True
        ),
    )
    if target is not None:
        cancelled = request(
            "background cancel: cancel",
            lambda: client.responses.cancel(target.id),
        )
        if cancelled is not None:
            report(
                "background cancel: the response id is unchanged",
                "PASS" if cancelled.id == target.id else "FAIL",
                f"id={cancelled.id!r}",
            )
        time.sleep(0.5)
        after = request(
            "background cancel: status", lambda: client.responses.retrieve(target.id)
        )
        if after is not None:
            report(
                "background cancel: the status is `cancelled`",
                "PASS" if after.status == "cancelled" else "FAIL",
                f"status={after.status!r}",
            )
        again = request(
            "background cancel: second cancel",
            lambda: client.responses.cancel(target.id),
        )
        if again is not None:
            report(
                "background cancel: the second cancel is idempotent",
                "PASS" if again.id == target.id else "FAIL",
                f"id={again.id!r}",
            )
        request(
            "background cancel: delete the cancelled job",
            lambda: client.responses.delete(target.id),
        )

    request(
        "background: delete the completed job",
        lambda: client.responses.delete(job.id),
    )


def main():
    host = urlparse(BASE_URL).hostname
    if host not in LOCAL_HOSTS:
        return config_error(
            f"base URL host {host!r} is not a local host; this test never calls the OpenAI service"
        )
    if not MODEL:
        return config_error(
            "LTENGINE_MODEL is empty; set it to the loaded model identifier"
        )

    try:
        import openai
        from openai import OpenAI
    except ImportError as err:
        print(f"FAIL import: {err}; install the pinned SDK (openai {PINNED_SDK})")
        return 1

    version = openai.__version__
    if version != PINNED_SDK and os.environ.get("LTENGINE_SDK_VERSION_ANY") != "1":
        return config_error(
            f"openai {version} does not match the pinned target {PINNED_SDK}; "
            "set LTENGINE_SDK_VERSION_ANY=1 to test a different version"
        )

    print(f"INFO sdk: openai {version} on Python {sys.version.split()[0]}")
    print(f"INFO target: {BASE_URL} with model {MODEL!r}")

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY or "not-configured")

    if API_KEY:
        try:
            OpenAI(base_url=BASE_URL, api_key="wrong-key").responses.create(
                model=MODEL, input="Say OK"
            )
            report("bearer auth rejects a wrong key", "FAIL", "the request succeeded")
        except Exception as err:  # noqa: BLE001 - any other error is a failed check
            report(
                "bearer auth rejects a wrong key",
                "PASS" if getattr(err, "status_code", None) == 401 else "FAIL",
                f"{type(err).__name__}: {err}",
            )
    else:
        report("bearer auth rejects a wrong key", "SKIP", "LTENGINE_API_KEY is empty")

    verify(
        "text input",
        client.responses.create(
            model=MODEL,
            instructions="Answer in one short word.",
            input="Say OK",
            metadata=SMOKE_METADATA,
        ),
        SMOKE_METADATA,
    )

    verify(
        "message-array input",
        client.responses.create(
            model=MODEL,
            input=[
                {"role": "system", "content": "Answer in one short word."},
                {"role": "user", "content": [{"type": "input_text", "text": "Say hi"}]},
            ],
        ),
    )

    # SP-PLANNED-003: structured output.
    verify_json_text(
        "structured output json_object",
        request(
            "structured output json_object",
            lambda: client.responses.create(
                model=MODEL,
                instructions="Answer with JSON only.",
                input=(
                    "Return a JSON object with one key, answer, whose value is the"
                    " capital of France."
                ),
                text={"format": {"type": "json_object"}},
            ),
        ),
    )

    verify_json_text(
        "structured output json_schema",
        request(
            "structured output json_schema",
            lambda: client.responses.create(
                model=MODEL,
                instructions="Answer with JSON only.",
                input="Return the capital of France in the answer field.",
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "capital",
                        "strict": True,
                        "schema": CAPITAL_SCHEMA,
                    }
                },
            ),
        ),
        schema_key="answer",
    )

    # SP-PLANNED-004: a forced custom function call.
    verify_function_call(
        "function calling required",
        request(
            "function calling required",
            lambda: client.responses.create(
                model=MODEL,
                input="What is the weather in Paris?",
                tools=[PLANNED_TOOL],
                tool_choice="required",
            ),
        ),
        "get_weather",
    )

    # SP-PLANNED-005: a named `tool_choice` and the `parallel_tool_calls` echo.
    named = request(
        "named tool_choice",
        lambda: client.responses.create(
            model=MODEL,
            input="What is the weather in Paris?",
            tools=[PLANNED_TOOL],
            tool_choice={"type": "function", "name": "get_weather"},
            parallel_tool_calls=True,
        ),
    )
    verify_function_call("named tool_choice", named, "get_weather")
    if named is not None:
        echoed = getattr(named, "parallel_tool_calls", None)
        report(
            "named tool_choice: parallel_tool_calls is echoed",
            "PASS" if echoed is True else "FAIL",
            f"parallel_tool_calls={echoed!r}",
        )

    # SP-PLANNED-013: the reasoning request field.
    verify_reasoning(client)

    # SP-PLANNED-006 through -011: the stored and long-running planned items.
    verify_stored_flow(client)
    verify_conversation_flow(client)
    verify_background_flow(client)

    # SP-PLANNED-001: the streamed event order.
    verify_stream(client)

    if FAILURES:
        print(f"RESULT FAIL ({len(FAILURES)} check(s) failed): {', '.join(FAILURES)}")
        return 1
    print("RESULT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
