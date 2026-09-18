//! Retrieval, deletion, and input-item listing for stored responses (PH-5).
//!
//! `GET /v1/responses/{response_id}`, `DELETE /v1/responses/{response_id}`, and
//! `GET /v1/responses/{response_id}/input_items` read the store of
//! `responses_store.rs`. Every route enforces the `RD-4` bearer credential and
//! rejects an unsupported query field (`CC-6`).

use std::sync::Arc;

use actix_web::{HttpRequest, HttpResponse, delete, get, web};
use serde_json::Value;

use crate::Args;
use crate::responses_http::{auth_guard, error_json, not_found, query_fields, strict_guard};
use crate::responses_store::AppStore;

/// The supported query of retrieval (`RD-27`).
struct RetrieveQuery {
    stream: bool,
    starting_after: Option<u32>,
}

/// Parse the retrieval query (`RD-27`, `CC-6`).
///
/// `stream` accepts `true` and `false`. `starting_after` is a non-negative
/// integer and requires `stream=true`. Any other field, and any other value of a
/// supported field, is a clear 400 that names the field.
fn parse_retrieve_query(req: &HttpRequest) -> Result<RetrieveQuery, HttpResponse> {
    let mut query = RetrieveQuery {
        stream: false,
        starting_after: None,
    };
    for (name, value) in query_fields(req) {
        match name.as_str() {
            "stream" => match value.as_deref() {
                Some("true") => query.stream = true,
                Some("false") => query.stream = false,
                _ => {
                    return Err(error_json(
                        400,
                        "`stream` must be `true` or `false`".to_string(),
                    ));
                }
            },
            "starting_after" => {
                match value.as_deref().and_then(|value| value.parse::<u32>().ok()) {
                    Some(after) => query.starting_after = Some(after),
                    None => {
                        return Err(error_json(
                            400,
                            "`starting_after` must be a non-negative integer".to_string(),
                        ));
                    }
                }
            }
            other => {
                return Err(error_json(
                    400,
                    format!("unsupported query field `{other}` on this route"),
                ));
            }
        }
    }
    if query.starting_after.is_some() && !query.stream {
        return Err(error_json(
            400,
            "`starting_after` requires `stream=true`".to_string(),
        ));
    }
    Ok(query)
}

/// `GET /v1/responses/{response_id}` (SP-PLANNED-008). With `stream=true` it
/// replays the stored SSE sequence (`PH-5a`, `RD-27`); otherwise it returns the
/// stored JSON body.
#[get("/v1/responses/{response_id}")]
pub async fn get_response(
    req: HttpRequest,
    path: web::Path<String>,
    args: web::Data<Arc<Args>>,
    store: web::Data<AppStore>,
) -> HttpResponse {
    if let Some(response) = auth_guard(&req, &args) {
        return response;
    }
    let query = match parse_retrieve_query(&req) {
        Ok(query) => query,
        Err(response) => return response,
    };
    let id = path.into_inner();
    match store.get_ref().get(&id) {
        Ok(Some(record)) => {
            if query.stream {
                match crate::responses_shape::replay_stream_body(
                    &record.response,
                    query.starting_after,
                ) {
                    Some(body) => HttpResponse::Ok()
                        .content_type("text/event-stream")
                        .body(body),
                    None => error_json(
                        500,
                        "the stored response cannot be replayed".to_string(),
                    ),
                }
            } else {
                HttpResponse::Ok().json(record.response)
            }
        }
        Ok(None) => not_found(&id),
        Err(err) => error_json(500, format!("failed to read the response store: {err}")),
    }
}

/// `DELETE /v1/responses/{response_id}` (SP-PLANNED-008).
#[delete("/v1/responses/{response_id}")]
pub async fn delete_response(
    req: HttpRequest,
    path: web::Path<String>,
    args: web::Data<Arc<Args>>,
    store: web::Data<AppStore>,
) -> HttpResponse {
    if let Some(response) = strict_guard(&req, &args) {
        return response;
    }
    let id = path.into_inner();
    match store.get_ref().delete(&id) {
        Ok(true) => HttpResponse::Ok().json(serde_json::json!({
            "id": id,
            "object": "response",
            "deleted": true,
        })),
        Ok(false) => not_found(&id),
        Err(err) => error_json(500, format!("failed to delete from the response store: {err}")),
    }
}

/// `GET /v1/responses/{response_id}/input_items` (SP-PLANNED-009).
#[get("/v1/responses/{response_id}/input_items")]
pub async fn list_input_items(
    req: HttpRequest,
    path: web::Path<String>,
    args: web::Data<Arc<Args>>,
    store: web::Data<AppStore>,
) -> HttpResponse {
    if let Some(response) = strict_guard(&req, &args) {
        return response;
    }
    let id = path.into_inner();
    match store.get_ref().get(&id) {
        Ok(Some(record)) => {
            let data = record.input_items;
            let first = first_identifier(&data);
            let last = data.last().and_then(identifier_of);
            HttpResponse::Ok().json(serde_json::json!({
                "object": "list",
                "data": data,
                "first_id": first,
                "last_id": last,
                "has_more": false,
            }))
        }
        Ok(None) => not_found(&id),
        Err(err) => error_json(500, format!("failed to read the response store: {err}")),
    }
}

fn first_identifier(items: &[Value]) -> Option<String> {
    items.first().and_then(identifier_of)
}

fn identifier_of(item: &Value) -> Option<String> {
    item.get("id").and_then(Value::as_str).map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::responses_store::{FileStore, StoredResponse, tests::TempStoreDir};
    use actix_web::{App, test};
    use clap::Parser;

    /// A stored record whose response object is replayable (`RD-27`).
    fn record() -> StoredResponse {
        StoredResponse {
            response: serde_json::json!({
                "id": "resp_1",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gemma3-4b",
                "metadata": null,
                "parallel_tool_calls": false,
                "tool_choice": "none",
                "tools": [],
                "output": [{
                    "id": "msg_out_1",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hello", "annotations": []}],
                }],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            }),
            input_items: vec![
                serde_json::json!({"id": "msg_1", "role": "user", "content": "first"}),
                serde_json::json!({"id": "msg_2", "role": "user", "content": "second"}),
            ],
        }
    }

    fn store_with(dir: &std::path::Path) -> FileStore {
        FileStore::new(dir.to_path_buf()).expect("store dir")
    }

    /// Build the retrieval service over one store. A macro avoids naming the
    /// opaque `init_service` return type.
    macro_rules! service {
        ($store:expr, $key:expr) => {
            test::init_service(
                App::new()
                    .app_data(web::Data::new(Arc::new(Args::parse_from([
                        "ltengine", "--api-key", $key,
                    ]))))
                    .app_data(web::Data::new($store))
                    .service(get_response)
                    .service(delete_response)
                    .service(list_input_items),
            )
            .await
        };
    }

    fn bearer(key: &str) -> (&'static str, String) {
        ("Authorization", format!("Bearer {key}"))
    }

    #[actix_web::test]
    async fn get_returns_the_stored_body() {
        // PH5-01 (route part).
        let temp = TempStoreDir::new();
        let store: AppStore = Arc::new(store_with(&temp.path));
        store.put("resp_1", &record()).expect("put");
        let app = service!(store, "secret");

        let req = test::TestRequest::get()
            .uri("/v1/responses/resp_1")
            .insert_header(bearer("secret"))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
        let body: Value = test::read_body_json(resp).await;
        assert_eq!(body, record().response);
    }

    #[actix_web::test]
    async fn unknown_id_returns_the_rd22_body() {
        // PH5-03.
        let temp = TempStoreDir::new();
        let store: AppStore = Arc::new(store_with(&temp.path));
        let app = service!(store, "secret");

        let req = test::TestRequest::get()
            .uri("/v1/responses/resp_missing")
            .insert_header(bearer("secret"))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::NOT_FOUND);
        let body: Value = test::read_body_json(resp).await;
        assert_eq!(
            body["error"]["message"],
            "No response found with id 'resp_missing'"
        );
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert!(body["error"]["param"].is_null());
        assert!(body["error"]["code"].is_null());
    }

    #[actix_web::test]
    async fn delete_then_get_is_not_found() {
        // PH5-04.
        let temp = TempStoreDir::new();
        let store: AppStore = Arc::new(store_with(&temp.path));
        store.put("resp_1", &record()).expect("put");
        let app = service!(store, "secret");

        let delete = test::TestRequest::delete()
            .uri("/v1/responses/resp_1")
            .insert_header(bearer("secret"))
            .to_request();
        let resp = test::call_service(&app, delete).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
        let body: Value = test::read_body_json(resp).await;
        assert_eq!(
            body,
            serde_json::json!({"id": "resp_1", "object": "response", "deleted": true})
        );

        let get = test::TestRequest::get()
            .uri("/v1/responses/resp_1")
            .insert_header(bearer("secret"))
            .to_request();
        let resp = test::call_service(&app, get).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::NOT_FOUND);
    }

    #[actix_web::test]
    async fn delete_unknown_returns_the_rd22_body() {
        // PH5-05.
        let temp = TempStoreDir::new();
        let store: AppStore = Arc::new(store_with(&temp.path));
        let app = service!(store, "secret");

        let req = test::TestRequest::delete()
            .uri("/v1/responses/resp_missing")
            .insert_header(bearer("secret"))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::NOT_FOUND);
        let body: Value = test::read_body_json(resp).await;
        assert_eq!(
            body["error"]["message"],
            "No response found with id 'resp_missing'"
        );
    }

    #[actix_web::test]
    async fn input_items_list_shape_and_order() {
        // PH5-06.
        let temp = TempStoreDir::new();
        let store: AppStore = Arc::new(store_with(&temp.path));
        store.put("resp_1", &record()).expect("put");
        let app = service!(store, "secret");

        let req = test::TestRequest::get()
            .uri("/v1/responses/resp_1/input_items")
            .insert_header(bearer("secret"))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
        let body: Value = test::read_body_json(resp).await;
        assert_eq!(body["object"], "list");
        assert_eq!(body["has_more"], false);
        assert_eq!(body["first_id"], "msg_1");
        assert_eq!(body["last_id"], "msg_2");
        assert_eq!(body["data"][0]["content"], "first");
        assert_eq!(body["data"][1]["content"], "second");
    }

    #[actix_web::test]
    async fn unsupported_query_field_is_400() {
        // PH5A-06: a field other than `stream` and `starting_after` is a 400.
        let temp = TempStoreDir::new();
        let store: AppStore = Arc::new(store_with(&temp.path));
        store.put("resp_1", &record()).expect("put");
        let app = service!(store, "secret");

        for (uri, field) in [
            ("/v1/responses/resp_1?include=x", "include"),
            ("/v1/responses/resp_1/input_items?limit=1", "limit"),
        ] {
            let req = test::TestRequest::get()
                .uri(uri)
                .insert_header(bearer("secret"))
                .to_request();
            let resp = test::call_service(&app, req).await;
            assert_eq!(resp.status(), actix_web::http::StatusCode::BAD_REQUEST, "{uri}");
            let body: Value = test::read_body_json(resp).await;
            let message = body["error"]["message"].as_str().unwrap();
            assert!(message.contains(field), "{uri} -> {message}");
        }
    }

    #[actix_web::test]
    async fn list_requires_the_bearer_credential() {
        // PH5-12.
        let temp = TempStoreDir::new();
        let store: AppStore = Arc::new(store_with(&temp.path));
        store.put("resp_1", &record()).expect("put");
        let app = service!(store, "secret");

        for uri in [
            "/v1/responses/resp_1",
            "/v1/responses/resp_1/input_items",
        ] {
            let req = test::TestRequest::get().uri(uri).to_request();
            let resp = test::call_service(&app, req).await;
            assert_eq!(resp.status(), actix_web::http::StatusCode::UNAUTHORIZED, "{uri}");
        }

        let req = test::TestRequest::get()
            .uri("/v1/responses/resp_1")
            .insert_header(bearer("wrong"))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::UNAUTHORIZED);
    }

    #[actix_web::test]
    async fn an_invalid_identifier_is_not_found() {
        // PH5-10 (route part): a traversal-shaped identifier never reaches the
        // store path.
        let temp = TempStoreDir::new();
        let store: AppStore = Arc::new(store_with(&temp.path));
        let app = service!(store, "secret");

        let req = test::TestRequest::get()
            .uri("/v1/responses/..%2Fescape")
            .insert_header(bearer("secret"))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::NOT_FOUND);
    }

    #[actix_web::test]
    async fn background_response_is_retrievable() {
        // PH6B-10: a background response is a stored response.
        let temp = TempStoreDir::new();
        let store: AppStore = Arc::new(store_with(&temp.path));
        let mut record = record();
        record.response["status"] = serde_json::json!("queued");
        store.put("resp_1", &record).expect("put");
        let app = service!(store, "secret");
        let req = test::TestRequest::get()
            .uri("/v1/responses/resp_1")
            .insert_header(bearer("secret"))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
        let body: Value = test::read_body_json(resp).await;
        assert_eq!(body["status"], "queued");
    }

    /// Parse an SSE body into its JSON payloads.
    fn parse_sse(body: &str) -> Vec<Value> {
        body.split("\n\n")
            .filter(|frame| !frame.trim().is_empty())
            .map(|frame| {
                let data = frame
                    .lines()
                    .find_map(|line| line.strip_prefix("data: "))
                    .expect("frame has a data line");
                serde_json::from_str(data).expect("payload is JSON")
            })
            .collect()
    }

    /// A `GET` that returns the status, the content type, and the raw body.
    async fn get_raw(
        store: AppStore,
        uri: &str,
    ) -> (actix_web::http::StatusCode, String, String) {
        let app = service!(store, "secret");
        let req = test::TestRequest::get()
            .uri(uri)
            .insert_header(bearer("secret"))
            .to_request();
        let resp = test::call_service(&app, req).await;
        let status = resp.status();
        let content_type = resp
            .headers()
            .get(actix_web::http::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_string();
        let body = test::read_body(resp).await;
        (status, content_type, String::from_utf8(body.to_vec()).unwrap())
    }

    #[actix_web::test]
    async fn stream_true_returns_sse() {
        // PH5A-01 (route part).
        let temp = TempStoreDir::new();
        let store: AppStore = Arc::new(store_with(&temp.path));
        store.put("resp_1", &record()).expect("put");
        let (status, content_type, body) =
            get_raw(store, "/v1/responses/resp_1?stream=true").await;
        assert_eq!(status, actix_web::http::StatusCode::OK);
        assert!(content_type.starts_with("text/event-stream"), "{content_type}");
        let events = parse_sse(&body);
        let types: Vec<&str> = events
            .iter()
            .map(|event| event["type"].as_str().unwrap())
            .collect();
        assert_eq!(
            types,
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        for (index, event) in events.iter().enumerate() {
            assert_eq!(event["sequence_number"], index);
        }
        assert_eq!(events[8]["response"], record().response);
    }

    #[actix_web::test]
    async fn starting_after_filters() {
        // PH5A-02 (route part): only the events after the given number remain,
        // with their original sequence numbers.
        let temp = TempStoreDir::new();
        let store: AppStore = Arc::new(store_with(&temp.path));
        store.put("resp_1", &record()).expect("put");
        let (_, _, full) = get_raw(store.clone(), "/v1/responses/resp_1?stream=true").await;
        let (_, _, tail) = get_raw(
            store,
            "/v1/responses/resp_1?stream=true&starting_after=2",
        )
        .await;
        let full_events = parse_sse(&full);
        let tail_events = parse_sse(&tail);
        assert_eq!(tail_events, full_events[3..]);
        assert_eq!(tail_events[0]["sequence_number"], 3);
    }

    #[actix_web::test]
    async fn stream_false_returns_json() {
        // PH5A-03.
        let temp = TempStoreDir::new();
        let store: AppStore = Arc::new(store_with(&temp.path));
        store.put("resp_1", &record()).expect("put");
        let app = service!(store, "secret");
        for uri in ["/v1/responses/resp_1", "/v1/responses/resp_1?stream=false"] {
            let req = test::TestRequest::get()
                .uri(uri)
                .insert_header(bearer("secret"))
                .to_request();
            let resp = test::call_service(&app, req).await;
            assert_eq!(resp.status(), actix_web::http::StatusCode::OK, "{uri}");
            let body: Value = test::read_body_json(resp).await;
            assert_eq!(body, record().response, "{uri}");
        }
    }

    #[actix_web::test]
    async fn bad_stream_value_is_400() {
        // PH5A-04.
        let temp = TempStoreDir::new();
        let store: AppStore = Arc::new(store_with(&temp.path));
        store.put("resp_1", &record()).expect("put");
        let app = service!(store, "secret");
        let req = test::TestRequest::get()
            .uri("/v1/responses/resp_1?stream=maybe")
            .insert_header(bearer("secret"))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::BAD_REQUEST);
        let body: Value = test::read_body_json(resp).await;
        assert!(body["error"]["message"].as_str().unwrap().contains("stream"));
    }

    #[actix_web::test]
    async fn starting_after_requires_stream() {
        // PH5A-05.
        let temp = TempStoreDir::new();
        let store: AppStore = Arc::new(store_with(&temp.path));
        store.put("resp_1", &record()).expect("put");
        let app = service!(store, "secret");
        for uri in [
            "/v1/responses/resp_1?starting_after=2",
            "/v1/responses/resp_1?stream=true&starting_after=abc",
        ] {
            let req = test::TestRequest::get()
                .uri(uri)
                .insert_header(bearer("secret"))
                .to_request();
            let resp = test::call_service(&app, req).await;
            assert_eq!(resp.status(), actix_web::http::StatusCode::BAD_REQUEST, "{uri}");
            let body: Value = test::read_body_json(resp).await;
            assert!(
                body["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("starting_after"),
                "{uri} -> {body}"
            );
        }
    }

    #[actix_web::test]
    async fn unknown_stream_is_404() {
        // PH5A-07.
        let temp = TempStoreDir::new();
        let store: AppStore = Arc::new(store_with(&temp.path));
        let app = service!(store, "secret");
        let req = test::TestRequest::get()
            .uri("/v1/responses/resp_missing?stream=true")
            .insert_header(bearer("secret"))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::NOT_FOUND);
        let body: Value = test::read_body_json(resp).await;
        assert_eq!(
            body["error"]["message"],
            "No response found with id 'resp_missing'"
        );
    }
}
