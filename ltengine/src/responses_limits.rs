//! Retention and storage-limit policy for the response store (`PH-6c`,
//! `RD-9`, `RD-25`).
//!
//! Expiry is lazy: a read removes an expired record and reports it as missing,
//! and a write removes the expired records before it checks the count limit.

use std::io;

use crate::responses_shape::now_secs;
use crate::responses_store::{ResponseStore, StoredResponse};

/// True when the record is older than the retention value. The value `0`
/// disables expiry (`RD-9`).
pub(crate) fn is_expired(created_at: u64, retention_secs: u64) -> bool {
    retention_secs != 0 && now_secs().saturating_sub(created_at) > retention_secs
}

/// Load a stored response and remove it lazily when it is expired (`RD-9`).
pub(crate) fn load_response(
    store: &dyn ResponseStore,
    id: &str,
    retention_secs: u64,
) -> io::Result<Option<StoredResponse>> {
    let record = match store.get(id)? {
        Some(record) => record,
        None => return Ok(None),
    };
    if is_expired(created_at(&record), retention_secs) {
        let _ = store.delete(id);
        return Ok(None);
    }
    Ok(Some(record))
}

fn created_at(record: &StoredResponse) -> u64 {
    record.response["created_at"].as_u64().unwrap_or(0)
}

/// Remove the expired records and report whether a new write fits (`RD-25`).
///
/// `limit` is `0` when the count limit is disabled.
pub(crate) fn prepare_response_write(
    store: &dyn ResponseStore,
    retention_secs: u64,
    limit: usize,
) -> io::Result<bool> {
    let mut live = 0usize;
    for id in store.list_response_ids()? {
        if let Some(record) = store.get(&id)? {
            if is_expired(created_at(&record), retention_secs) {
                let _ = store.delete(&id);
            } else {
                live += 1;
            }
        }
    }
    Ok(limit == 0 || live < limit)
}

/// The `RD-25` `server_error` message.
pub(crate) fn storage_full_message(limit: usize) -> String {
    format!("the response storage limit of {limit} is reached")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::responses_store::{FileStore, tests::TempStoreDir};
    use serde_json::json;

    fn record(id: &str, created_at: u64) -> StoredResponse {
        StoredResponse {
            response: json!({"id": id, "created_at": created_at, "status": "completed"}),
            input_items: Vec::new(),
        }
    }

    #[test]
    fn prepare_removes_expired() {
        // PH6C-01 and PH6C-05: the expired record is removed before the count,
        // so a store at the limit accepts a new write.
        let temp = TempStoreDir::new();
        let store = FileStore::new(temp.path.clone()).expect("store dir");
        store.put("resp_1", &record("resp_1", 1)).expect("put");
        store.put("resp_2", &record("resp_2", now_secs())).expect("put");
        assert!(prepare_response_write(&store, 1, 2).expect("prepare"));
        assert_eq!(store.get("resp_1").expect("get"), None);
        assert!(store.get("resp_2").expect("get").is_some());
    }

    #[test]
    fn limit_is_reached() {
        // PH6C-03 and PH6C-04.
        let temp = TempStoreDir::new();
        let store = FileStore::new(temp.path.clone()).expect("store dir");
        store.put("resp_1", &record("resp_1", now_secs())).expect("put");
        assert!(!prepare_response_write(&store, 0, 1).expect("prepare"));
        assert!(prepare_response_write(&store, 0, 0).expect("prepare"));
        assert!(prepare_response_write(&store, 0, 2).expect("prepare"));
    }

    #[test]
    fn fresh_response_is_kept() {
        // PH6C-02.
        let temp = TempStoreDir::new();
        let store = FileStore::new(temp.path.clone()).expect("store dir");
        store.put("resp_1", &record("resp_1", now_secs())).expect("put");
        assert!(load_response(&store, "resp_1", 3600).expect("load").is_some());
        assert!(load_response(&store, "resp_1", 0).expect("load").is_some());
    }

    #[test]
    fn expired_response_is_not_found() {
        // PH6C-01.
        let temp = TempStoreDir::new();
        let store = FileStore::new(temp.path.clone()).expect("store dir");
        store.put("resp_1", &record("resp_1", 1)).expect("put");
        assert_eq!(load_response(&store, "resp_1", 1).expect("load"), None);
        assert_eq!(store.get("resp_1").expect("get"), None);
    }
}
