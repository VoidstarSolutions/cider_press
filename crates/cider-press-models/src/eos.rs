//! `eos_token_id` stop ids declared by a checkpoint's `config.json` and
//! `generation_config.json`. `mlx_lm` lets `generation_config` override
//! `config.json`; unioning them instead yields a superset of its stop set.

use std::collections::HashSet;
use std::fs;
use std::path::Path;

use serde_json::Value;

use crate::{Error, Result};

/// Union of the top-level `eos_token_id` in `config.json` (when present) and
/// in `generation_config.json` (when that file exists). Each may be an int, a
/// list of ints, or null/absent (contributing nothing).
///
/// # Errors
/// [`Error::Io`] when a present file can't be read, a JSON parse error, or
/// [`Error::InvalidArgument`] for an `eos_token_id` entry that isn't a `u32`.
pub fn config_eos_ids(checkpoint_dir: &Path) -> Result<HashSet<u32>> {
    let mut ids = HashSet::new();
    for name in ["config.json", "generation_config.json"] {
        let path = checkpoint_dir.join(name);
        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(source) => return Err(Error::Io { path, source }),
        };
        let cfg: Value = serde_json::from_slice(&bytes)?;
        if let Some(v) = cfg.get("eos_token_id") {
            collect_ids(v, &mut ids)?;
        }
    }
    Ok(ids)
}

fn collect_ids(value: &Value, out: &mut HashSet<u32>) -> Result<()> {
    let to_id = |n: &Value| -> Result<u32> {
        n.as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| Error::InvalidArgument(format!("invalid eos_token_id entry {n}")))
    };
    match value {
        Value::Null => {}
        Value::Array(items) => {
            for item in items {
                out.insert(to_id(item)?);
            }
        }
        other => {
            out.insert(to_id(other)?);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids_for(config: Option<&str>, generation: Option<&str>) -> HashSet<u32> {
        let dir = tempfile::tempdir().unwrap();
        if let Some(c) = config {
            fs::write(dir.path().join("config.json"), c).unwrap();
        }
        if let Some(g) = generation {
            fs::write(dir.path().join("generation_config.json"), g).unwrap();
        }
        config_eos_ids(dir.path()).unwrap()
    }

    fn set(ids: &[u32]) -> HashSet<u32> {
        ids.iter().copied().collect()
    }

    #[test]
    fn int_form() {
        assert_eq!(ids_for(Some(r#"{"eos_token_id": 7}"#), None), set(&[7]));
    }

    #[test]
    fn list_form() {
        assert_eq!(
            ids_for(Some(r#"{"eos_token_id": [7, 9]}"#), None),
            set(&[7, 9])
        );
    }

    #[test]
    fn null_and_absent_field_contribute_nothing() {
        assert!(ids_for(Some(r#"{"eos_token_id": null}"#), None).is_empty());
        assert!(ids_for(Some(r#"{"hidden_size": 4}"#), None).is_empty());
    }

    #[test]
    fn absent_files_contribute_nothing() {
        assert!(ids_for(None, None).is_empty());
        assert_eq!(ids_for(Some(r#"{"eos_token_id": 7}"#), None), set(&[7]));
    }

    #[test]
    fn generation_config_is_unioned() {
        assert_eq!(
            ids_for(
                Some(r#"{"eos_token_id": 7}"#),
                Some(r#"{"eos_token_id": [9, 7]}"#)
            ),
            set(&[7, 9])
        );
    }

    #[test]
    fn nested_text_config_is_ignored() {
        assert!(ids_for(Some(r#"{"text_config": {"eos_token_id": 5}}"#), None).is_empty());
    }

    #[test]
    fn rejects_non_integer_entry() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("config.json"), r#"{"eos_token_id": ["x"]}"#).unwrap();
        assert!(config_eos_ids(dir.path()).is_err());
    }
}
