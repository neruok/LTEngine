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

Exit code 0 means every check passed.
"""

import os
import sys
from urllib.parse import urlparse

PINNED_SDK = "3.14.1"
DEFAULT_BASE_URL = "http://127.0.0.1:5050/v1"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
SMOKE_METADATA = {"suite": "responses-smoke"}

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

    if FAILURES:
        print(f"RESULT FAIL ({len(FAILURES)} check(s) failed): {', '.join(FAILURES)}")
        return 1
    print("RESULT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
