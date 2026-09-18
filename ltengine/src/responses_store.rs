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

/// The stored record of one conversation (`PH-6a`, `RD-23`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct ConversationRecord {
    pub id: String,
    pub created_at: u64,
    pub metadata: Value,
    pub items: Vec<Value>,
}

impl ConversationRecord {
    /// The OpenAI-shaped conversation object. The items are not part of it.
    pub(crate) fn response(&self) -> Value {
        serde_json::json!({
            "id": self.id,
            "object": "conversation",
            "created_at": self.created_at,
            "metadata": self.metadata,
        })
    }
}

/// The storage boundary of `RD-21`. One narrow interface with one file-backed
/// implementation. It holds responses (`PH-5`) and conversations (`PH-6a`).
/// Every method reports an `io::Error` so a route can answer fail-closed with
/// the OpenAI-shaped HTTP 500 body (`RD-15`).
pub(crate) trait ResponseStore: Send + Sync {
    fn put(&self, id: &str, record: &StoredResponse) -> io::Result<()>;
    fn get(&self, id: &str) -> io::Result<Option<StoredResponse>>;
    fn delete(&self, id: &str) -> io::Result<bool>;
    fn put_conversation(&self, id: &str, record: &ConversationRecord) -> io::Result<()>;
    fn get_conversation(&self, id: &str) -> io::Result<Option<ConversationRecord>>;
    fn delete_conversation(&self, id: &str) -> io::Result<bool>;
    /// The identifiers of the live response records (`PH-6c`).
    fn list_response_ids(&self) -> io::Result<Vec<String>>;
}

/// Shared handle that the Actix app data carries.
pub(crate) type AppStore = Arc<dyn ResponseStore>;

/// True only for `resp_` followed by one or more lowercase hexadecimal digits.
///
/// Every identifier passes this check before a path is built, so a path
/// parameter can never escape the store directory (`PH5-10`).
pub(crate) fn valid_id(id: &str) -> bool {
    is_hex_id(id, "resp_")
}

/// True only for `conv_` followed by one or more lowercase hexadecimal digits.
pub(crate) fn valid_conversation_id(id: &str) -> bool {
    is_hex_id(id, "conv_")
}

fn is_hex_id(id: &str, prefix: &str) -> bool {
    id.strip_prefix(prefix).is_some_and(|rest| {
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

    /// The path of one conversation record.
    pub(crate) fn conversation_path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.json"))
    }

    fn conversation_temp_path(&self, id: &str) -> PathBuf {
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

    fn put_conversation(&self, id: &str, record: &ConversationRecord) -> io::Result<()> {
        if !valid_conversation_id(id) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid conversation id: {id}"),
            ));
        }
        let bytes = serde_json::to_vec(record).map_err(io::Error::other)?;
        fs::write(self.conversation_temp_path(id), &bytes)?;
        fs::rename(self.conversation_temp_path(id), self.conversation_path(id))?;
        Ok(())
    }

    fn get_conversation(&self, id: &str) -> io::Result<Option<ConversationRecord>> {
        if !valid_conversation_id(id) {
            return Ok(None);
        }
        match fs::read(self.conversation_path(id)) {
            Ok(bytes) => serde_json::from_slice(&bytes).map(Some).map_err(io::Error::other),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(err),
        }
    }

    fn delete_conversation(&self, id: &str) -> io::Result<bool> {
        if !valid_conversation_id(id) {
            return Ok(false);
        }
        match fs::remove_file(self.conversation_path(id)) {
            Ok(()) => Ok(true),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(err) => Err(err),
        }
    }

    fn list_response_ids(&self) -> io::Result<Vec<String>> {
        let mut ids = Vec::new();
        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            let Some(id) = name.strip_suffix(".json") else {
                continue;
            };
            if name.starts_with("resp_") && valid_id(id) {
                ids.push(id.to_string());
            }
        }
        Ok(ids)
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

    fn put_conversation(&self, _id: &str, _record: &ConversationRecord) -> io::Result<()> {
        Err(io::Error::other("store unavailable"))
    }

    fn get_conversation(&self, _id: &str) -> io::Result<Option<ConversationRecord>> {
        Err(io::Error::other("store unavailable"))
    }

    fn delete_conversation(&self, _id: &str) -> io::Result<bool> {
        Err(io::Error::other("store unavailable"))
    }

    fn list_response_ids(&self) -> io::Result<Vec<String>> {
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

    fn conversation() -> ConversationRecord {
        ConversationRecord {
            id: "conv_1".to_string(),
            created_at: 1,
            metadata: serde_json::json!(null),
            items: vec![serde_json::json!({"id": "msg_1", "role": "user", "content": "hi"})],
        }
    }

    #[test]
    fn conversation_round_trip_and_delete() {
        // PH6A-01 through PH6A-04 (store part).
        let temp = TempStoreDir::new();
        let store = FileStore::new(temp.path.clone()).expect("store dir");
        store.put_conversation("conv_1", &conversation()).expect("put");
        assert_eq!(
            store.get_conversation("conv_1").expect("get"),
            Some(conversation())
        );
        assert!(store.delete_conversation("conv_1").expect("delete"));
        assert_eq!(store.get_conversation("conv_1").expect("get"), None);
        assert!(!store.delete_conversation("conv_1").expect("delete"));
    }

    #[test]
    fn rejects_a_conversation_traversal_id() {
        // PH6A-13 (store part): an invalid id never becomes a path.
        let temp = TempStoreDir::new();
        let store = FileStore::new(temp.path.clone()).expect("store dir");
        for id in ["../escape", "conv_", "conv_AB", "conv_xyz", "resp_1"] {
            assert!(!valid_conversation_id(id), "{id}");
            assert_eq!(store.get_conversation(id).expect("get"), None, "{id}");
            assert!(!store.delete_conversation(id).expect("delete"), "{id}");
            assert!(store.put_conversation(id, &conversation()).is_err(), "{id}");
        }
        assert!(!temp.path.join("..").join("escape.json").exists());
    }
}
