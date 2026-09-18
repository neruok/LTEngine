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
use crate::responses_http::{bearer, check_auth, error_json, first_query_field, not_found};
use crate::responses_store::AppStore;

/// The credential check and the query guard shared by the retrieval routes.
///
/// `RD-4` fixes bearer auth. `CC-6` fixes one outcome for an unsupported query
/// field: a clear 400 that names it. `PH-5` supports no query field, so
/// `stream=true` and every other field take the 400 path. The credential is
/// checked first, so a request without it is a 401 even when a query field is
/// also present.
fn guard(req: &HttpRequest, args: &Args) -> Option<HttpResponse> {
    if let Err(message) = check_auth(&args.api_key, bearer(req)) {
        return Some(error_json(401, message));
    }
    if let Some(field) = first_query_field(req) {
        return Some(error_json(
            400,
            format!("unsupported query field `{field}` on this route"),
        ));
    }
    None
}

/// `GET /v1/responses/{response_id}` (SP-PLANNED-008).
#[get("/v1/responses/{response_id}")]
pub async fn get_response(
    req: HttpRequest,
    path: web::Path<String>,
    args: web::Data<Arc<Args>>,
    store: web::Data<AppStore>,
) -> HttpResponse {
    if let Some(response) = guard(&req, &args) {
        return response;
    }
    let id = path.into_inner();
    match store.get_ref().get(&id) {
        Ok(Some(record)) => HttpResponse::Ok().json(record.response),
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
    if let Some(response) = guard(&req, &args) {
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
    if let Some(response) = guard(&req, &args) {
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

    fn record() -> StoredResponse {
        StoredResponse {
            response: serde_json::json!({
                "id": "resp_1",
                "object": "response",
                "status": "completed",
                "output": [{"type": "message"}],
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
    async fn query_field_is_a_400() {
        // PH5-11.
        let temp = TempStoreDir::new();
        let store: AppStore = Arc::new(store_with(&temp.path));
        store.put("resp_1", &record()).expect("put");
        let app = service!(store, "secret");

        for (uri, field) in [
            ("/v1/responses/resp_1?stream=true", "stream"),
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
}
