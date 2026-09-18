//! HTTP error shaping, credential check, and query guard shared by the
//! `/v1/responses` route family (PH-1, PH-5, RD-4, CC-6).
//!
//! The new route family uses the OpenAI-shaped nested error body. The existing
//! LibreTranslate routes keep their flat `{"error": "<string>"}` body (CC-1).
//! The extraction of some of these helpers from `responses.rs` keeps that file
//! under the split threshold of the workspace instructions.

use actix_web::{HttpRequest, HttpResponse, http::StatusCode, http::header};

/// The OpenAI-shaped nested error body of `SP-MUST-007`. `server_error` applies
/// to HTTP 5xx, `invalid_request_error` to everything else.
pub(crate) fn error_json(status: u16, message: String) -> HttpResponse {
    let error_type = if status >= 500 {
        "server_error"
    } else {
        "invalid_request_error"
    };
    HttpResponse::build(StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR))
        .json(serde_json::json!({
            "error": {
                "message": message,
                "type": error_type,
                "param": serde_json::Value::Null,
                "code": serde_json::Value::Null,
            }
        }))
}

/// The `RD-22` not-found body. An unknown identifier, a deleted identifier, a
/// `DELETE` of an unknown identifier, and a `previous_response_id` that names
/// an unknown identifier all share this response.
pub(crate) fn not_found(id: &str) -> HttpResponse {
    error_json(404, format!("No response found with id '{id}'"))
}

/// Convert a payload-extractor failure (for example the default oversized-body
/// `413`) into the same OpenAI-shaped nested error body as every other route
/// error. Without this the extractor rejection bypasses the handler and Actix
/// answers with `text/plain`.
pub(crate) fn extractor_error(err: actix_web::Error) -> HttpResponse {
    let status = err.as_response_error().status_code();
    error_json(status.as_u16(), format!("invalid request body: {}", err))
}

/// The new route family accepts `Authorization: Bearer` only (RD-4).
pub(crate) fn check_auth(api_key: &str, authorization: Option<&str>) -> Result<(), String> {
    if api_key.is_empty() {
        return Ok(());
    }
    match authorization {
        Some(value) if value == format!("Bearer {api_key}") => Ok(()),
        Some(_) => Err("incorrect API key provided".to_string()),
        None => Err("missing bearer token".to_string()),
    }
}

/// The `Authorization` header of a request, when it is valid UTF-8.
pub(crate) fn bearer(req: &HttpRequest) -> Option<&str> {
    req.headers()
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
}

/// The name of the first unsupported query field, or `None` when the query is
/// empty (`CC-6`). `PH-5` supports no query field on retrieval, so any field
/// takes this path, including `stream=true`.
pub(crate) fn first_query_field(req: &HttpRequest) -> Option<String> {
    let query = req.query_string();
    if query.is_empty() {
        return None;
    }
    let field = query.split('&').next().unwrap_or(query);
    let name = field.split('=').next().unwrap_or(field);
    Some(name.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn probe_body(body: Result<actix_web::web::Bytes, actix_web::Error>) -> HttpResponse {
        match body {
            Ok(_) => HttpResponse::Ok().finish(),
            Err(err) => extractor_error(err),
        }
    }

    #[actix_web::test]
    async fn oversized_body_returns_nested_error() {
        let app = actix_web::test::init_service(
            actix_web::App::new()
                .app_data(actix_web::web::PayloadConfig::new(16))
                .route("/probe", actix_web::web::post().to(probe_body)),
        )
        .await;

        let req = actix_web::test::TestRequest::post()
            .uri("/probe")
            .set_payload("x".repeat(64))
            .to_request();
        let resp = actix_web::test::call_service(&app, req).await;

        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let body: serde_json::Value = actix_web::test::read_body_json(resp).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], serde_json::Value::Null);
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("invalid request body")
        );
    }

    #[test]
    fn error_json_selects_the_type_from_the_status() {
        let bad = error_json(400, "x".to_string());
        assert_eq!(bad.status(), StatusCode::BAD_REQUEST);
        let server = error_json(500, "boom".to_string());
        assert_eq!(server.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn not_found_names_the_identifier() {
        let body = not_found("resp_abc");
        assert_eq!(body.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn check_auth_accepts_the_configured_bearer() {
        assert!(check_auth("", None).is_ok());
        assert!(check_auth("secret", Some("Bearer secret")).is_ok());
        assert!(check_auth("secret", Some("Bearer wrong")).is_err());
        assert!(check_auth("secret", None).is_err());
    }

    #[test]
    fn first_query_field_reports_the_first_name() {
        let none = actix_web::test::TestRequest::default().to_http_request();
        assert_eq!(first_query_field(&none), None);
        let one = actix_web::test::TestRequest::default()
            .uri("/v1/responses/resp_1?stream=true")
            .to_http_request();
        assert_eq!(first_query_field(&one), Some("stream".to_string()));
        let many = actix_web::test::TestRequest::default()
            .uri("/v1/responses/resp_1?a=1&b=2")
            .to_http_request();
        assert_eq!(first_query_field(&many), Some("a".to_string()));
        let bare = actix_web::test::TestRequest::default()
            .uri("/v1/responses/resp_1?include")
            .to_http_request();
        assert_eq!(first_query_field(&bare), Some("include".to_string()));
    }
}
