//! File-backed response store for `POST /v1/responses` (PH-5, RD-21, RD-22).
//!
//! One JSON file per response at `<store-dir>/<resp_id>.json`. The file holds the
//! OpenAI-shaped response object and the effective input items. Access sits
//! behind the narrow [`ResponseStore`] boundary with one file-backed
//! implementation, so a later backend (SQLite, Valkey, PostgreSQL) can replace
//! it without a route change (RD-21).

use std::fs;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The stored record of one response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct StoredResponse {
    /// The OpenAI-shaped response object, equal to the body returned to the
    /// client when the response was created.
    pub response: Value,
    /// The effective input items of `RD-22`, in order.
    pub input_items: Vec<Value>,
}

/// The storage boundary of `RD-21`. One narrow interface with one file-backed
/// implementation. Every method reports an `io::Error` so the route can answer
/// fail-closed with the OpenAI-shaped HTTP 500 body (`RD-15`).
pub(crate) trait ResponseStore: Send + Sync {
    fn put(&self, id: &str, record: &StoredResponse) -> io::Result<()>;
    fn get(&self, id: &str) -> io::Result<Option<StoredResponse>>;
    fn delete(&self, id: &str) -> io::Result<bool>;
}

/// Shared handle that the Actix app data carries.
pub(crate) type AppStore = Arc<dyn ResponseStore>;

/// True only for `resp_` followed by one or more lowercase hexadecimal digits.
///
/// Every identifier passes this check before a path is built, so a path
/// parameter can never escape the store directory (`PH5-10`).
pub(crate) fn valid_id(id: &str) -> bool {
    id.strip_prefix("resp_").is_some_and(|rest| {
        !rest.is_empty()
            && rest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}

/// A file-backed store (RD-21).
pub(crate) struct FileStore {
    dir: PathBuf,
}

impl FileStore {
    /// Create the store directory when it is absent. A failure is fatal for the
    /// process, because `RD-21` requires a usable store at startup.
    pub(crate) fn new(dir: PathBuf) -> io::Result<Self> {
        fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    /// The absolute-or-relative path of one record.
    pub(crate) fn path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.json"))
    }

    fn temp_path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.json.tmp"))
    }
}

impl ResponseStore for FileStore {
    fn put(&self, id: &str, record: &StoredResponse) -> io::Result<()> {
        if !valid_id(id) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid response id: {id}"),
            ));
        }
        let bytes = serde_json::to_vec(record).map_err(io::Error::other)?;
        fs::write(self.temp_path(id), &bytes)?;
        fs::rename(self.temp_path(id), self.path(id))?;
        Ok(())
    }

    fn get(&self, id: &str) -> io::Result<Option<StoredResponse>> {
        if !valid_id(id) {
            return Ok(None);
        }
        match fs::read(self.path(id)) {
            Ok(bytes) => serde_json::from_slice(&bytes).map(Some).map_err(io::Error::other),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(err),
        }
    }

    fn delete(&self, id: &str) -> io::Result<bool> {
        if !valid_id(id) {
            return Ok(false);
        }
        match fs::remove_file(self.path(id)) {
            Ok(()) => Ok(true),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(err) => Err(err),
        }
    }
}

/// A store that always fails, for the `PH5-09` fail-closed check.
#[cfg(test)]
pub(crate) struct FailingStore;

#[cfg(test)]
impl ResponseStore for FailingStore {
    fn put(&self, _id: &str, _record: &StoredResponse) -> io::Result<()> {
        Err(io::Error::other("store unavailable"))
    }

    fn get(&self, _id: &str) -> io::Result<Option<StoredResponse>> {
        Err(io::Error::other("store unavailable"))
    }

    fn delete(&self, _id: &str) -> io::Result<bool> {
        Err(io::Error::other("store unavailable"))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::responses_shape::new_id;
    use serde_json::json;

    /// A unique temporary store directory that removes itself on drop.
    pub(crate) struct TempStoreDir {
        pub path: PathBuf,
    }

    impl TempStoreDir {
        pub(crate) fn new() -> Self {
            let path = std::env::temp_dir().join(format!("ltengine-store-{}", new_id("d")));
            Self { path }
        }
    }

    impl Drop for TempStoreDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn record(text: &str) -> StoredResponse {
        StoredResponse {
            response: json!({"id": "resp_1", "object": "response", "output": text}),
            input_items: vec![json!({"role": "user", "content": "hi"})],
        }
    }

    #[test]
    fn round_trips_one_response() {
        // PH5-01 (store part).
        let temp = TempStoreDir::new();
        let store = FileStore::new(temp.path.clone()).expect("store dir");
        store.put("resp_1", &record("hello")).expect("put");
        let loaded = store.get("resp_1").expect("get").expect("present");
        assert_eq!(loaded, record("hello"));
    }

    #[test]
    fn get_and_delete_report_a_missing_record() {
        // PH5-03 and PH5-05 (store part).
        let temp = TempStoreDir::new();
        let store = FileStore::new(temp.path.clone()).expect("store dir");
        assert_eq!(store.get("resp_1").expect("get"), None);
        assert!(!store.delete("resp_1").expect("delete"));
    }

    #[test]
    fn delete_removes_the_record() {
        // PH5-04 (store part).
        let temp = TempStoreDir::new();
        let store = FileStore::new(temp.path.clone()).expect("store dir");
        store.put("resp_1", &record("hello")).expect("put");
        assert!(store.delete("resp_1").expect("delete"));
        assert_eq!(store.get("resp_1").expect("get"), None);
    }

    #[test]
    fn rejects_a_traversal_id() {
        // PH5-10 (store part): an invalid id is never turned into a path.
        let temp = TempStoreDir::new();
        let store = FileStore::new(temp.path.clone()).expect("store dir");
        for id in ["../escape", "resp_", "resp_AB", "resp_xyz", "other_1"] {
            assert!(!valid_id(id), "{id}");
            assert_eq!(store.get(id).expect("get"), None, "{id}");
            assert!(!store.delete(id).expect("delete"), "{id}");
            assert!(store.put(id, &record("x")).is_err(), "{id}");
        }
        assert!(!temp.path.join("..").join("escape.json").exists());
    }

    #[test]
    fn store_dir_is_created() {
        let temp = TempStoreDir::new();
        assert!(!temp.path.exists());
        let _ = FileStore::new(temp.path.clone()).expect("store dir");
        assert!(temp.path.is_dir());
    }

    #[test]
    fn an_unreadable_record_is_an_error() {
        // A corrupt file is a store failure, not a missing record: fail-closed.
        let temp = TempStoreDir::new();
        let store = FileStore::new(temp.path.clone()).expect("store dir");
        fs::write(store.path("resp_3"), b"{not json").expect("write");
        assert!(store.get("resp_3").is_err());
    }

    #[test]
    fn valid_id_accepts_the_generated_form() {
        assert!(valid_id(&new_id("resp_")));
        assert!(valid_id("resp_0123456789abcdef"));
    }
}
