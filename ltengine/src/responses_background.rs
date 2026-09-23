//! Background responses and cancellation (`PH-6b`, `SP-PLANNED-011`, `RD-16`,
//! `RD-24`).
//!
//! `background: true` stores a `queued` response, returns it, and runs one
//! generation in a blocking task. Cancellation is a per-job `AtomicBool` that
//! the decode path observes (`llm.rs`). The job writes exactly one terminal
//! status and never overwrites a cancelled response with a completed one.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use actix_web::{HttpRequest, HttpResponse, post, web};
use serde_json::{Value, json};

use crate::Args;
use crate::llm::{self, TokenUsage};
use crate::responses_http::{error_json, not_found, strict_guard};
use crate::responses_reasoning::ReasoningEffect;
use crate::responses_shape::{
    ResponseControls, build_response_object, calls_output, message_output, new_id, now_secs,
    usage_json,
};
use crate::responses_store::{AppStore, ConversationRecord, StoredResponse};
use crate::responses_tools::{ModelTurn, ToolEcho, ToolRequest, parse_turn};

/// The one generation path, behind a narrow boundary so a test can replace it.
pub(crate) trait Generate: Send + Sync {
    fn generate(
        &self,
        system: String,
        user: String,
        grammar: Option<String>,
        reasoning: ReasoningEffect,
        generation: llm::Generation,
        cancel: Arc<AtomicBool>,
    ) -> anyhow::Result<(String, TokenUsage)>;
}

impl Generate for llm::LLM {
    fn generate(
        &self,
        system: String,
        user: String,
        grammar: Option<String>,
        reasoning: ReasoningEffect,
        generation: llm::Generation,
        cancel: Arc<AtomicBool>,
    ) -> anyhow::Result<(String, TokenUsage)> {
        self.run_prompt_usage_grammar_cancellable(
            system,
            user,
            grammar.as_deref(),
            Some(cancel.as_ref()),
            &reasoning.as_reasoning(),
            &generation,
        )
    }
}

/// The per-job cancellation flags (`RD-24`).
#[derive(Default)]
pub(crate) struct CancelRegistry {
    tokens: Mutex<HashMap<String, Arc<AtomicBool>>>,
}

impl CancelRegistry {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Register one job and return its flag.
    pub(crate) fn register(&self, id: &str) -> Arc<AtomicBool> {
        let token = Arc::new(AtomicBool::new(false));
        self.tokens
            .lock()
            .expect("cancel registry lock")
            .insert(id.to_string(), Arc::clone(&token));
        token
    }

    /// Set the flag of one job. False when no job is registered for the id.
    pub(crate) fn cancel(&self, id: &str) -> bool {
        match self.tokens.lock().expect("cancel registry lock").get(id) {
            Some(token) => {
                token.store(true, Ordering::Relaxed);
                true
            }
            None => false,
        }
    }

    /// Drop the flag of a finished job.
    pub(crate) fn finish(&self, id: &str) {
        self.tokens
            .lock()
            .expect("cancel registry lock")
            .remove(id);
    }
}

/// Everything the background job needs after the route validated the request.
pub(crate) struct BackgroundRequest {
    pub store: AppStore,
    pub generator: Arc<dyn Generate>,
    pub registry: Arc<CancelRegistry>,
    pub model: String,
    pub metadata: Option<Value>,
    pub conversation: Option<String>,
    pub echo: ToolEcho,
    pub tools: ToolRequest,
    pub grammar: Option<String>,
    pub json_required: bool,
    /// The decoded `reasoning` request (`RD-28`).
    pub reasoning: ReasoningEffect,
    /// The decoded generation controls (`RD-29`).
    pub generation: llm::Generation,
    /// The response echo of the generation controls and verbosity (`RD-29`,
    /// `RD-30`).
    pub controls: ResponseControls,
    pub system: String,
    pub user: String,
    pub input_items: Vec<Value>,
    pub conversation_record: Option<ConversationRecord>,
    /// Retention and storage limits (`PH-6c`, `RD-9`, `RD-25`).
    pub retention_secs: u64,
    pub max_stored_responses: usize,
}

/// One background generation.
pub(crate) struct BackgroundJob {
    pub id: String,
    pub created_at: u64,
    pub request: BackgroundRequest,
    pub cancel: Arc<AtomicBool>,
}

/// Store a `queued` response, start the job, and return the queued body.
pub(crate) fn start(request: BackgroundRequest) -> HttpResponse {
    match crate::responses_limits::prepare_response_write(
        request.store.as_ref(),
        request.retention_secs,
        request.max_stored_responses,
    ) {
        Ok(true) => {}
        Ok(false) => {
            return error_json(
                507,
                crate::responses_limits::storage_full_message(request.max_stored_responses),
            );
        }
        Err(err) => {
            return error_json(500, format!("failed to prepare the response store: {err}"));
        }
    }
    let id = new_id("resp_");
    let created_at = now_secs();
    let queued = build_response_object(
        &id,
        created_at,
        "queued",
        &request.model,
        request.metadata.as_ref(),
        request.conversation.as_deref(),
        &request.echo,
        &request.controls,
        json!([]),
        Value::Null,
    );
    let record = StoredResponse {
        response: queued.clone(),
        input_items: request.input_items.clone(),
    };
    if let Err(err) = request.store.put(&id, &record) {
        return error_json(500, format!("failed to store the response: {err}"));
    }
    let cancel = request.registry.register(&id);
    let job = BackgroundJob {
        id,
        created_at,
        request,
        cancel,
    };
    actix_web::rt::task::spawn_blocking(move || job.run());
    HttpResponse::Ok().json(queued)
}

impl BackgroundJob {
    /// Run one attempt and write exactly one terminal status (`RD-16`).
    pub(crate) fn run(self) {
        let BackgroundJob {
            id,
            created_at,
            request,
            cancel,
        } = self;
        let in_progress = object_for(&id, created_at, "in_progress", &request, json!([]));
        let _ = request.store.put(
            &id,
            &StoredResponse {
                response: in_progress,
                input_items: request.input_items.clone(),
            },
        );

        let result = request.generator.generate(
            request.system.clone(),
            request.user.clone(),
            request.grammar.clone(),
            request.reasoning.clone(),
            request.generation,
            Arc::clone(&cancel),
        );
        let completed = match result {
            Ok((text, usage)) if !cancel.load(Ordering::Relaxed) => match output_items(&text, &request)
            {
                Some(output) => build_response_object(
                    &id,
                    created_at,
                    "completed",
                    &request.model,
                    request.metadata.as_ref(),
                    request.conversation.as_deref(),
                    &request.echo,
                    &request.controls,
                    output,
                    usage_json(&usage),
                ),
                None => object_for(&id, created_at, "failed", &request, json!([])),
            },
            Ok(_) => object_for(&id, created_at, "cancelled", &request, json!([])),
            Err(err) if is_cancelled_error(&err) => {
                object_for(&id, created_at, "cancelled", &request, json!([]))
            }
            Err(_) => object_for(&id, created_at, "failed", &request, json!([])),
        };
        let _ = request.store.put(
            &id,
            &StoredResponse {
                response: completed.clone(),
                input_items: request.input_items.clone(),
            },
        );

        if completed["status"] == "completed" {
            if let (Some(conversation_id), Some(mut record)) =
                (request.conversation.as_deref(), request.conversation_record)
            {
                record.items.extend(request.input_items.iter().cloned());
                if let Some(output) = completed["output"].as_array() {
                    record.items.extend(output.iter().cloned());
                }
                let _ = request.store.put_conversation(conversation_id, &record);
            }
        }
        request.registry.finish(&id);
    }
}

fn object_for(
    id: &str,
    created_at: u64,
    status: &str,
    request: &BackgroundRequest,
    output: Value,
) -> Value {
    build_response_object(
        id,
        created_at,
        status,
        &request.model,
        request.metadata.as_ref(),
        request.conversation.as_deref(),
        &request.echo,
        &request.controls,
        output,
        Value::Null,
    )
}

/// The output items of a successful generation, or `None` on non-conformance.
fn output_items(text: &str, request: &BackgroundRequest) -> Option<Value> {
    if request.tools.offers_tools() {
        match parse_turn(text, &request.tools) {
            Ok(ModelTurn::Message(answer)) => Some(message_output(&answer)),
            Ok(ModelTurn::Calls(calls)) => Some(calls_output(&calls)),
            Err(_) => None,
        }
    } else if request.json_required {
        serde_json::from_str::<Value>(text.trim()).ok()?;
        Some(message_output(text))
    } else {
        Some(message_output(text))
    }
}

fn is_cancelled_error(err: &anyhow::Error) -> bool {
    matches!(
        err.downcast_ref::<llm::LLMError>(),
        Some(llm::LLMError::Cancelled)
    )
}

/// `POST /v1/responses/{response_id}/cancel` (`RD-24`).
#[post("/v1/responses/{response_id}/cancel")]
pub async fn cancel_response(
    req: HttpRequest,
    path: web::Path<String>,
    args: web::Data<Arc<Args>>,
    store: web::Data<AppStore>,
    registry: web::Data<Arc<CancelRegistry>>,
) -> HttpResponse {
    if let Some(response) = strict_guard(&req, &args) {
        return response;
    }
    let id = path.into_inner();
    let record = match store.get_ref().get(&id) {
        Ok(Some(record)) => record,
        Ok(None) => return not_found(&id),
        Err(err) => return error_json(500, format!("failed to read the response store: {err}")),
    };
    let StoredResponse {
        response,
        input_items,
    } = record;
    let status = response["status"].as_str().unwrap_or_default();
    if matches!(status, "completed" | "failed" | "cancelled" | "incomplete") {
        // Idempotent: a terminal response is returned unchanged.
        return HttpResponse::Ok().json(response);
    }
    registry.cancel(&id);
    let mut object = response;
    object["status"] = Value::from("cancelled");
    object["output"] = json!([]);
    object["usage"] = Value::Null;
    let _ = store.get_ref().put(
        &id,
        &StoredResponse {
            response: object.clone(),
            input_items,
        },
    );
    HttpResponse::Ok().json(object)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::responses_store::{FileStore, tests::TempStoreDir};
    use actix_web::{App, http::StatusCode};
    use clap::Parser;

    /// A generator that returns a scripted result (`PH-6b` tests).
    struct FakeGenerator {
        behavior: Behavior,
    }

    enum Behavior {
        Message(String),
        Fail,
        Cancel,
    }

    impl Generate for FakeGenerator {
        fn generate(
            &self,
            _system: String,
            _user: String,
            _grammar: Option<String>,
            _reasoning: ReasoningEffect,
            _generation: llm::Generation,
            cancel: Arc<AtomicBool>,
        ) -> anyhow::Result<(String, TokenUsage)> {
            match &self.behavior {
                Behavior::Message(text) => Ok((
                    text.clone(),
                    TokenUsage {
                        input_tokens: 3,
                        output_tokens: 2,
                    },
                )),
                Behavior::Fail => Err(anyhow::anyhow!("boom")),
                Behavior::Cancel => {
                    cancel.store(true, Ordering::Relaxed);
                    Err(llm::LLMError::Cancelled.into())
                }
            }
        }
    }

    fn request(store: AppStore, generator: Arc<dyn Generate>) -> BackgroundRequest {
        BackgroundRequest {
            store,
            generator,
            registry: Arc::new(CancelRegistry::new()),
            model: "gemma3-4b".to_string(),
            metadata: None,
            conversation: None,
            echo: ToolEcho::text_only(),
            tools: crate::responses_tools::parse_tools(None, None, None).expect("no tools"),
            grammar: None,
            json_required: false,
            reasoning: ReasoningEffect::default(),
            generation: llm::Generation::default(),
            controls: ResponseControls::default(),
            system: String::new(),
            user: "hi".to_string(),
            input_items: Vec::new(),
            conversation_record: None,
            retention_secs: 0,
            max_stored_responses: 0,
        }
    }

    fn run_job(behavior: Behavior) -> (StoredResponse, Arc<CancelRegistry>) {        let temp = TempStoreDir::new();
        let store: AppStore = Arc::new(FileStore::new(temp.path.clone()).expect("store dir"));
        let request = request(store.clone(), Arc::new(FakeGenerator { behavior }));
        let registry = Arc::clone(&request.registry);
        let cancel = registry.register("resp_1");
        BackgroundJob {
            id: "resp_1".to_string(),
            created_at: 1,
            request,
            cancel,
        }
        .run();
        let record = store
            .get("resp_1")
            .expect("get")
            .expect("stored");
        (record, registry)
    }

    #[test]
    fn job_reaches_completed() {
        // PH6B-01 and PH6B-02.
        let (record, _) = run_job(Behavior::Message("hello".to_string()));
        assert_eq!(record.response["status"], "completed");
        assert_eq!(record.response["output"][0]["content"][0]["text"], "hello");
        assert_eq!(record.response["usage"]["input_tokens"], 3);
    }

    #[test]
    fn job_reaches_failed() {
        // PH6B-03: one attempt, then `failed`.
        let (record, _) = run_job(Behavior::Fail);
        assert_eq!(record.response["status"], "failed");
        assert_eq!(record.response["output"], json!([]));
    }

    #[test]
    fn job_reaches_cancelled() {
        // PH6B-04 and PH6B-05.
        let (record, _) = run_job(Behavior::Cancel);
        assert_eq!(record.response["status"], "cancelled");
        assert_eq!(record.response["output"], json!([]));
    }

    #[test]
    fn storage_limit_blocks_start() {
        // PH6C-07: a full store rejects a background write before it starts.
        let temp = TempStoreDir::new();
        let store: AppStore = Arc::new(FileStore::new(temp.path.clone()).expect("store dir"));
        store
            .put(
                "resp_1",
                &StoredResponse {
                    response: json!({"id": "resp_1", "created_at": now_secs(), "status": "completed"}),
                    input_items: Vec::new(),
                },
            )
            .expect("put");
        let mut background = request(
            store,
            Arc::new(FakeGenerator {
                behavior: Behavior::Message("hi".to_string()),
            }),
        );
        background.retention_secs = 0;
        background.max_stored_responses = 1;
        let response = start(background);
        assert_eq!(response.status().as_u16(), 507);
    }

    macro_rules! service {
        ($store:expr, $registry:expr) => {
            actix_web::test::init_service(
                App::new()
                    .app_data(web::Data::new(Arc::new(Args::parse_from([
                        "ltengine",
                        "--api-key",
                        "secret",
                    ]))))
                    .app_data(web::Data::new($store))
                    .app_data(web::Data::new($registry))
                    .service(cancel_response),
            )
            .await
        };
    }

    fn queued_record() -> StoredResponse {
        StoredResponse {
            response: json!({
                "id": "resp_1",
                "object": "response",
                "created_at": 1,
                "status": "queued",
                "model": "gemma3-4b",
                "output": [],
                "usage": null,
            }),
            input_items: Vec::new(),
        }
    }

    #[actix_web::test]
    async fn cancel_sets_cancelled() {
        // PH6B-04 (route part).
        let temp = TempStoreDir::new();
        let store: AppStore = Arc::new(FileStore::new(temp.path.clone()).expect("store dir"));
        store.put("resp_1", &queued_record()).expect("put");
        let app = service!(store, Arc::new(CancelRegistry::new()));
        let req = actix_web::test::TestRequest::post()
            .uri("/v1/responses/resp_1/cancel")
            .insert_header(("Authorization", "Bearer secret"))
            .to_request();
        let resp = actix_web::test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body: Value = actix_web::test::read_body_json(resp).await;
        assert_eq!(body["status"], "cancelled");
    }

    #[actix_web::test]
    async fn cancel_is_idempotent() {
        // PH6B-06.
        let temp = TempStoreDir::new();
        let store: AppStore = Arc::new(FileStore::new(temp.path.clone()).expect("store dir"));
        let record = queued_record();
        store.put("resp_1", &record).expect("put");
        let app = service!(store, Arc::new(CancelRegistry::new()));
        for _ in 0..2 {
            let req = actix_web::test::TestRequest::post()
                .uri("/v1/responses/resp_1/cancel")
                .insert_header(("Authorization", "Bearer secret"))
                .to_request();
            let resp = actix_web::test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            let body: Value = actix_web::test::read_body_json(resp).await;
            assert_eq!(body["status"], "cancelled");
        }
    }

    #[actix_web::test]
    async fn cancel_unknown_is_not_found() {
        // PH6B-07.
        let temp = TempStoreDir::new();
        let store: AppStore = Arc::new(FileStore::new(temp.path.clone()).expect("store dir"));
        let app = service!(store, Arc::new(CancelRegistry::new()));
        let req = actix_web::test::TestRequest::post()
            .uri("/v1/responses/resp_missing/cancel")
            .insert_header(("Authorization", "Bearer secret"))
            .to_request();
        let resp = actix_web::test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let body: Value = actix_web::test::read_body_json(resp).await;
        assert_eq!(
            body["error"]["message"],
            "No response found with id 'resp_missing'"
        );
    }

    #[actix_web::test]
    async fn cancel_credential_and_query_guard() {
        // PH6B-11.
        let temp = TempStoreDir::new();
        let store: AppStore = Arc::new(FileStore::new(temp.path.clone()).expect("store dir"));
        store.put("resp_1", &queued_record()).expect("put");
        let app = service!(store, Arc::new(CancelRegistry::new()));
        let req = actix_web::test::TestRequest::post()
            .uri("/v1/responses/resp_1/cancel")
            .to_request();
        let resp = actix_web::test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        let req = actix_web::test::TestRequest::post()
            .uri("/v1/responses/resp_1/cancel?force=1")
            .insert_header(("Authorization", "Bearer secret"))
            .to_request();
        let resp = actix_web::test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
}
