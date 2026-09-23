//! Conversations and conversation items for the Responses API (`PH-6a`,
//! `SP-PLANNED-010`, `RD-23`).
//!
//! The routes store a `conv_*` object and an ordered item list behind the same
//! file-backed store as `PH-5` (`responses_store.rs`). The item limit and the
//! retention point are enforced on write and on read.

use std::io;
use std::sync::Arc;

use actix_web::{HttpRequest, HttpResponse, delete, get, post, web};
use serde::Deserialize;
use serde_json::Value;

use crate::Args;
use crate::responses_http::{error_json, extractor_error, strict_guard};
use crate::responses_shape::{new_id, now_secs};
use crate::responses_store::{AppStore, ConversationRecord, ResponseStore};

/// The `RD-23` not-found body of a conversation.
pub(crate) fn conversation_not_found(id: &str) -> HttpResponse {
    error_json(
        404,
        format!("No conversation found with id '{id}'"),
    )
}

fn item_not_found(id: &str) -> HttpResponse {
    error_json(
        404,
        format!("No conversation item found with id '{id}'"),
    )
}

fn read_error(err: io::Error) -> HttpResponse {
    error_json(500, format!("failed to read the conversation store: {err}"))
}

fn write_error(err: io::Error) -> HttpResponse {
    error_json(500, format!("failed to store the conversation: {err}"))
}

/// True when the record is older than the retention value. The value `0`
/// disables expiry (`RD-23`).
fn is_expired(created_at: u64, retention_secs: u64) -> bool {
    retention_secs != 0 && now_secs().saturating_sub(created_at) > retention_secs
}

/// Load a conversation and remove it lazily when it is expired (`RD-23`).
pub(crate) fn load_conversation(
    store: &dyn ResponseStore,
    id: &str,
    retention_secs: u64,
) -> io::Result<Option<ConversationRecord>> {
    let record = match store.get_conversation(id)? {
        Some(record) => record,
        None => return Ok(None),
    };
    if is_expired(record.created_at, retention_secs) {
        let _ = store.delete_conversation(id);
        return Ok(None);
    }
    Ok(Some(record))
}

/// Parse an optional JSON body. An absent or blank body is the default value,
/// so `POST /v1/conversations` accepts an empty body (`CC-6` still applies to a
/// present but malformed body).
fn parse_body<T>(body: &[u8]) -> Result<T, (u16, String)>
where
    T: for<'de> Deserialize<'de> + Default,
{
    if body.iter().all(u8::is_ascii_whitespace) {
        return Ok(T::default());
    }
    serde_json::from_slice(body).map_err(|err| (400, format!("invalid request body: {err}")))
}

/// `metadata` uses the `RD-3` limits (`§3.2`).
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConversationBody {
    #[serde(default, deserialize_with = "crate::responses_metadata::deserialize_metadata")]
    metadata: Option<Value>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ItemsBody {
    #[serde(default)]
    items: Vec<Value>,
}

/// Assign an `id` to an item that carries none (`§3.3`).
fn assign_item_id(mut item: Value) -> Value {
    if item.get("id").and_then(Value::as_str).is_some() {
        return item;
    }
    let prefix = if item.get("role").is_some() {
        "msg_"
    } else {
        match item.get("type").and_then(Value::as_str) {
            Some("function_call") => "fc_",
            Some("function_call_output") => "fco_",
            _ => "msg_",
        }
    };
    if let Some(map) = item.as_object_mut() {
        map.insert("id".to_string(), Value::from(new_id(prefix)));
    }
    item
}

/// The `§3.1` list shape.
fn list_body(items: Vec<Value>) -> Value {
    let first = items
        .first()
        .and_then(|item| item.get("id"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let last = items
        .last()
        .and_then(|item| item.get("id"))
        .and_then(Value::as_str)
        .map(str::to_string);
    serde_json::json!({
        "object": "list",
        "data": items,
        "first_id": first,
        "last_id": last,
        "has_more": false,
    })
}

/// `POST /v1/conversations`.
#[post("/v1/conversations")]
pub async fn create_conversation(
    req: HttpRequest,
    body: Result<web::Bytes, actix_web::Error>,
    args: web::Data<Arc<Args>>,
    store: web::Data<AppStore>,
) -> HttpResponse {
    if let Some(response) = strict_guard(&req, &args) {
        return response;
    }
    let body = match body {
        Ok(body) => body,
        Err(err) => return extractor_error(err),
    };
    let request: ConversationBody = match parse_body(&body) {
        Ok(request) => request,
        Err((status, message)) => return error_json(status, message),
    };
    let id = new_id("conv_");
    let record = ConversationRecord {
        id: id.clone(),
        created_at: now_secs(),
        metadata: request.metadata.unwrap_or(Value::Null),
        items: Vec::new(),
    };
    if let Err(err) = store.get_ref().put_conversation(&id, &record) {
        return write_error(err);
    }
    HttpResponse::Ok().json(record.response())
}

/// `GET /v1/conversations/{conversation_id}`.
#[get("/v1/conversations/{conversation_id}")]
pub async fn get_conversation(
    req: HttpRequest,
    path: web::Path<String>,
    args: web::Data<Arc<Args>>,
    store: web::Data<AppStore>,
) -> HttpResponse {
    if let Some(response) = strict_guard(&req, &args) {
        return response;
    }
    let id = path.into_inner();
    match load_conversation(store.get_ref().as_ref(), &id, args.retention_secs) {
        Ok(Some(record)) => HttpResponse::Ok().json(record.response()),
        Ok(None) => conversation_not_found(&id),
        Err(err) => read_error(err),
    }
}

/// `POST /v1/conversations/{conversation_id}`. Replaces `metadata`.
#[post("/v1/conversations/{conversation_id}")]
pub async fn update_conversation(
    req: HttpRequest,
    path: web::Path<String>,
    body: Result<web::Bytes, actix_web::Error>,
    args: web::Data<Arc<Args>>,
    store: web::Data<AppStore>,
) -> HttpResponse {
    if let Some(response) = strict_guard(&req, &args) {
        return response;
    }
    let body = match body {
        Ok(body) => body,
        Err(err) => return extractor_error(err),
    };
    let request: ConversationBody = match parse_body(&body) {
        Ok(request) => request,
        Err((status, message)) => return error_json(status, message),
    };
    let id = path.into_inner();
    let mut record = match load_conversation(store.get_ref().as_ref(), &id, args.retention_secs) {
        Ok(Some(record)) => record,
        Ok(None) => return conversation_not_found(&id),
        Err(err) => return read_error(err),
    };
    record.metadata = request.metadata.unwrap_or(Value::Null);
    if let Err(err) = store.get_ref().put_conversation(&id, &record) {
        return write_error(err);
    }
    HttpResponse::Ok().json(record.response())
}

/// `DELETE /v1/conversations/{conversation_id}`.
#[delete("/v1/conversations/{conversation_id}")]
pub async fn delete_conversation(
    req: HttpRequest,
    path: web::Path<String>,
    args: web::Data<Arc<Args>>,
    store: web::Data<AppStore>,
) -> HttpResponse {
    if let Some(response) = strict_guard(&req, &args) {
        return response;
    }
    let id = path.into_inner();
    match load_conversation(store.get_ref().as_ref(), &id, args.retention_secs) {
        Ok(Some(_)) => match store.get_ref().delete_conversation(&id) {
            Ok(true) => HttpResponse::Ok().json(serde_json::json!({
                "id": id,
                "object": "conversation.deleted",
                "deleted": true,
            })),
            Ok(false) => conversation_not_found(&id),
            Err(err) => write_error(err),
        },
        Ok(None) => conversation_not_found(&id),
        Err(err) => read_error(err),
    }
}

/// `POST /v1/conversations/{conversation_id}/items`.
#[post("/v1/conversations/{conversation_id}/items")]
pub async fn add_items(
    req: HttpRequest,
    path: web::Path<String>,
    body: Result<web::Bytes, actix_web::Error>,
    args: web::Data<Arc<Args>>,
    store: web::Data<AppStore>,
) -> HttpResponse {
    if let Some(response) = strict_guard(&req, &args) {
        return response;
    }
    let body = match body {
        Ok(body) => body,
        Err(err) => return extractor_error(err),
    };
    let request: ItemsBody = match parse_body(&body) {
        Ok(request) => request,
        Err((status, message)) => return error_json(status, message),
    };
    if request.items.is_empty() {
        return error_json(400, "`items` must contain at least one item".to_string());
    }
    let id = path.into_inner();
    let mut record = match load_conversation(store.get_ref().as_ref(), &id, args.retention_secs) {
        Ok(Some(record)) => record,
        Ok(None) => return conversation_not_found(&id),
        Err(err) => return read_error(err),
    };
    let limit = args.max_conversation_items;
    if limit != 0 && record.items.len() + request.items.len() > limit {
        return error_json(
            400,
            format!("the conversation item limit of {limit} would be exceeded"),
        );
    }
    let created: Vec<Value> = request.items.into_iter().map(assign_item_id).collect();
    record.items.extend(created.iter().cloned());
    if let Err(err) = store.get_ref().put_conversation(&id, &record) {
        return write_error(err);
    }
    HttpResponse::Ok().json(list_body(created))
}

/// `GET /v1/conversations/{conversation_id}/items`.
#[get("/v1/conversations/{conversation_id}/items")]
pub async fn list_items(
    req: HttpRequest,
    path: web::Path<String>,
    args: web::Data<Arc<Args>>,
    store: web::Data<AppStore>,
) -> HttpResponse {
    if let Some(response) = strict_guard(&req, &args) {
        return response;
    }
    let id = path.into_inner();
    match load_conversation(store.get_ref().as_ref(), &id, args.retention_secs) {
        Ok(Some(record)) => HttpResponse::Ok().json(list_body(record.items)),
        Ok(None) => conversation_not_found(&id),
        Err(err) => read_error(err),
    }
}

/// `GET /v1/conversations/{conversation_id}/items/{item_id}`.
#[get("/v1/conversations/{conversation_id}/items/{item_id}")]
pub async fn get_item(
    req: HttpRequest,
    path: web::Path<(String, String)>,
    args: web::Data<Arc<Args>>,
    store: web::Data<AppStore>,
) -> HttpResponse {
    if let Some(response) = strict_guard(&req, &args) {
        return response;
    }
    let (id, item_id) = path.into_inner();
    match load_conversation(store.get_ref().as_ref(), &id, args.retention_secs) {
        Ok(Some(record)) => match find_item(&record, &item_id) {
            Some(item) => HttpResponse::Ok().json(item),
            None => item_not_found(&item_id),
        },
        Ok(None) => conversation_not_found(&id),
        Err(err) => read_error(err),
    }
}

/// `DELETE /v1/conversations/{conversation_id}/items/{item_id}`.
#[delete("/v1/conversations/{conversation_id}/items/{item_id}")]
pub async fn delete_item(
    req: HttpRequest,
    path: web::Path<(String, String)>,
    args: web::Data<Arc<Args>>,
    store: web::Data<AppStore>,
) -> HttpResponse {
    if let Some(response) = strict_guard(&req, &args) {
        return response;
    }
    let (id, item_id) = path.into_inner();
    let mut record = match load_conversation(store.get_ref().as_ref(), &id, args.retention_secs) {
        Ok(Some(record)) => record,
        Ok(None) => return conversation_not_found(&id),
        Err(err) => return read_error(err),
    };
    let before = record.items.len();
    record
        .items
        .retain(|item| item.get("id").and_then(Value::as_str) != Some(item_id.as_str()));
    if record.items.len() == before {
        return item_not_found(&item_id);
    }
    if let Err(err) = store.get_ref().put_conversation(&id, &record) {
        return write_error(err);
    }
    HttpResponse::Ok().json(serde_json::json!({
        "id": item_id,
        "object": "conversation.item.deleted",
        "deleted": true,
    }))
}

fn find_item<'a>(record: &'a ConversationRecord, item_id: &str) -> Option<&'a Value> {
    record
        .items
        .iter()
        .find(|item| item.get("id").and_then(Value::as_str) == Some(item_id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::responses_store::{FileStore, tests::TempStoreDir};
    use actix_web::{App, http::StatusCode, test};
    use clap::Parser;

    macro_rules! service {
        ($store:expr, $args:expr) => {
            test::init_service(
                App::new()
                    .app_data(web::Data::new(Arc::new($args)))
                    .app_data(web::Data::new($store))
                    .service(create_conversation)
                    .service(get_conversation)
                    .service(update_conversation)
                    .service(delete_conversation)
                    .service(add_items)
                    .service(list_items)
                    .service(get_item)
                    .service(delete_item),
            )
            .await
        };
    }

    fn args(extra: &[&str]) -> Args {
        let mut argv = vec!["ltengine", "--api-key", "secret", "--responses-api", "open-responses"];
        argv.extend_from_slice(extra);
        Args::parse_from(argv)
    }

    fn store(dir: &std::path::Path) -> AppStore {
        Arc::new(FileStore::new(dir.to_path_buf()).expect("store dir"))
    }

    fn auth() -> (&'static str, &'static str) {
        ("Authorization", "Bearer secret")
    }

    /// Create a conversation and return its identifier.
    macro_rules! create {
        ($app:expr) => {{
            let req = test::TestRequest::post()
                .uri("/v1/conversations")
                .insert_header(auth())
                .to_request();
            let resp = test::call_service($app, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            let body: Value = test::read_body_json(resp).await;
            body["id"].as_str().unwrap().to_string()
        }};
    }

    #[actix_web::test]
    async fn create_returns_the_object() {
        // PH6A-01.
        let temp = TempStoreDir::new();
        let app = service!(store(&temp.path), args(&[]));
        let req = test::TestRequest::post()
            .uri("/v1/conversations")
            .insert_header(auth())
            .set_json(serde_json::json!({"metadata": {"a": "b"}}))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body: Value = test::read_body_json(resp).await;
        assert!(body["id"].as_str().unwrap().starts_with("conv_"));
        assert_eq!(body["object"], "conversation");
        assert!(body["created_at"].is_u64());
        assert_eq!(body["metadata"], serde_json::json!({"a": "b"}));

        // An invalid metadata value is a 400 naming `metadata`.
        let req = test::TestRequest::post()
            .uri("/v1/conversations")
            .insert_header(auth())
            .set_payload(r#"{"metadata":{"a":1}}"#)
            .insert_header(("Content-Type", "application/json"))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body: Value = test::read_body_json(resp).await;
        assert!(body["error"]["message"].as_str().unwrap().contains("metadata"));
    }

    #[actix_web::test]
    async fn get_and_unknown_id() {
        // PH6A-02.
        let temp = TempStoreDir::new();
        let app = service!(store(&temp.path), args(&[]));
        let id = create!(&app);
        let req = test::TestRequest::get()
            .uri(&format!("/v1/conversations/{id}"))
            .insert_header(auth())
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body: Value = test::read_body_json(resp).await;
        assert_eq!(body["id"], id);
        assert!(body["metadata"].is_null());

        let req = test::TestRequest::get()
            .uri("/v1/conversations/conv_missing")
            .insert_header(auth())
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let body: Value = test::read_body_json(resp).await;
        assert_eq!(
            body["error"]["message"],
            "No conversation found with id 'conv_missing'"
        );
    }

    #[actix_web::test]
    async fn update_replaces_metadata() {
        // PH6A-03.
        let temp = TempStoreDir::new();
        let app = service!(store(&temp.path), args(&[]));
        let id = create!(&app);
        let req = test::TestRequest::post()
            .uri(&format!("/v1/conversations/{id}"))
            .insert_header(auth())
            .set_json(serde_json::json!({"metadata": {"b": "c"}}))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body: Value = test::read_body_json(resp).await;
        assert_eq!(body["metadata"], serde_json::json!({"b": "c"}));
    }

    #[actix_web::test]
    async fn delete_then_get_is_not_found() {
        // PH6A-04.
        let temp = TempStoreDir::new();
        let app = service!(store(&temp.path), args(&[]));
        let id = create!(&app);
        let req = test::TestRequest::delete()
            .uri(&format!("/v1/conversations/{id}"))
            .insert_header(auth())
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body: Value = test::read_body_json(resp).await;
        assert_eq!(body["object"], "conversation.deleted");
        assert_eq!(body["deleted"], true);

        let req = test::TestRequest::get()
            .uri(&format!("/v1/conversations/{id}"))
            .insert_header(auth())
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[actix_web::test]
    async fn add_items_assigns_ids_and_lists() {
        // PH6A-05 and PH6A-06.
        let temp = TempStoreDir::new();
        let app = service!(store(&temp.path), args(&[]));
        let id = create!(&app);
        for text in ["first", "second"] {
            let req = test::TestRequest::post()
                .uri(&format!("/v1/conversations/{id}/items"))
                .insert_header(auth())
                .set_json(serde_json::json!({"items": [{"role": "user", "content": text}]}))
                .to_request();
            let resp = test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            let body: Value = test::read_body_json(resp).await;
            assert!(body["data"][0]["id"].as_str().unwrap().starts_with("msg_"));
        }
        let req = test::TestRequest::get()
            .uri(&format!("/v1/conversations/{id}/items"))
            .insert_header(auth())
            .to_request();
        let resp = test::call_service(&app, req).await;
        let body: Value = test::read_body_json(resp).await;
        assert_eq!(body["object"], "list");
        assert_eq!(body["has_more"], false);
        assert_eq!(body["data"][0]["content"], "first");
        assert_eq!(body["data"][1]["content"], "second");
        assert_eq!(body["first_id"], body["data"][0]["id"]);
        assert_eq!(body["last_id"], body["data"][1]["id"]);
    }

    #[actix_web::test]
    async fn get_and_delete_item() {
        // PH6A-07 and PH6A-08.
        let temp = TempStoreDir::new();
        let app = service!(store(&temp.path), args(&[]));
        let id = create!(&app);
        let req = test::TestRequest::post()
            .uri(&format!("/v1/conversations/{id}/items"))
            .insert_header(auth())
            .set_json(serde_json::json!({"items": [{"role": "user", "content": "hi"}]}))
            .to_request();
        let resp = test::call_service(&app, req).await;
        let body: Value = test::read_body_json(resp).await;
        let item_id = body["data"][0]["id"].as_str().unwrap().to_string();

        let req = test::TestRequest::get()
            .uri(&format!("/v1/conversations/{id}/items/{item_id}"))
            .insert_header(auth())
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let req = test::TestRequest::delete()
            .uri(&format!("/v1/conversations/{id}/items/{item_id}"))
            .insert_header(auth())
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body: Value = test::read_body_json(resp).await;
        assert_eq!(body["object"], "conversation.item.deleted");

        let req = test::TestRequest::get()
            .uri(&format!("/v1/conversations/{id}/items/{item_id}"))
            .insert_header(auth())
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let body: Value = test::read_body_json(resp).await;
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("No conversation item found"));
    }

    #[actix_web::test]
    async fn item_limit_is_a_400() {
        // PH6A-11.
        let temp = TempStoreDir::new();
        let app = service!(store(&temp.path), args(&["--max-conversation-items", "2"]));
        let id = create!(&app);
        let req = test::TestRequest::post()
            .uri(&format!("/v1/conversations/{id}/items"))
            .insert_header(auth())
            .set_json(serde_json::json!({"items": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]}))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let req = test::TestRequest::post()
            .uri(&format!("/v1/conversations/{id}/items"))
            .insert_header(auth())
            .set_json(serde_json::json!({"items": [{"role": "user", "content": "c"}]}))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body: Value = test::read_body_json(resp).await;
        assert!(body["error"]["message"].as_str().unwrap().contains("limit"));
    }

    #[actix_web::test]
    async fn expired_conversation_is_not_found() {
        // PH6A-12: the record is removed lazily on read.
        let temp = TempStoreDir::new();
        let backend = store(&temp.path);
        backend
            .put_conversation(
                "conv_1",
                &ConversationRecord {
                    id: "conv_1".to_string(),
                    created_at: 1,
                    metadata: Value::Null,
                    items: Vec::new(),
                },
            )
            .expect("put");
        let app = service!(backend, args(&["--retention-secs", "1"]));
        let req = test::TestRequest::get()
            .uri("/v1/conversations/conv_1")
            .insert_header(auth())
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[actix_web::test]
    async fn credential_and_query_guard() {
        // PH6A-13.
        let temp = TempStoreDir::new();
        let app = service!(store(&temp.path), args(&[]));
        let req = test::TestRequest::post()
            .uri("/v1/conversations")
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        let req = test::TestRequest::get()
            .uri("/v1/conversations/conv_1?limit=1")
            .insert_header(auth())
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body: Value = test::read_body_json(resp).await;
        assert!(body["error"]["message"].as_str().unwrap().contains("limit"));
    }
}
