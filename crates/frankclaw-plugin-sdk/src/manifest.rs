//! Plugin manifest parsing and validation.

use frankclaw_core::error::{ConfigIo, ConfigValidation, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Maximum plugin id length.
const MAX_PLUGIN_ID_LEN: usize = 128;

/// A plugin manifest read from `plugin.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginManifest {
    pub id: String,
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub channels: Vec<String>,
    #[serde(default)]
    pub tools: Vec<String>,
    #[serde(default)]
    pub config_schema: Option<serde_json::Value>,
}

/// Validate a plugin id: lowercase alphanumeric + hyphens, 1-128 chars.
pub fn validate_plugin_id(id: &str) -> Result<()> {
    if id.is_empty() || id.len() > MAX_PLUGIN_ID_LEN {
        return Err(ConfigValidation {
            msg: format!(
                "plugin id must be 1-{MAX_PLUGIN_ID_LEN} characters, got {}",
                id.len()
            ),
        }
        .build());
    }
    if !id
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
    {
        return Err(ConfigValidation {
            msg: format!(
                "plugin id '{id}' must contain only lowercase alphanumeric, hyphens, and underscores"
            ),
        }
        .build());
    }
    Ok(())
}

/// Load and validate a plugin manifest from a `plugin.json` file.
pub fn load_plugin_manifest(path: &Path) -> Result<PluginManifest> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        ConfigIo {
            msg: format!("failed to read plugin manifest '{}': {e}", path.display()),
        }
        .build()
    })?;
    let manifest: PluginManifest = serde_json::from_str(&content).map_err(|e| {
        ConfigValidation {
            msg: format!("invalid plugin manifest '{}': {e}", path.display()),
        }
        .build()
    })?;
    validate_manifest(&manifest)?;
    Ok(manifest)
}

fn validate_manifest(manifest: &PluginManifest) -> Result<()> {
    validate_plugin_id(&manifest.id)?;
    if manifest.name.trim().is_empty() {
        return Err(ConfigValidation {
            msg: format!("plugin '{}' manifest is missing name", manifest.id),
        }
        .build());
    }
    if manifest.version.trim().is_empty() {
        return Err(ConfigValidation {
            msg: format!("plugin '{}' manifest is missing version", manifest.id),
        }
        .build());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_plugin_ids() {
        validate_plugin_id("my-plugin").expect("expected 'my-plugin' to be valid");
        validate_plugin_id("plugin_v2").expect("expected 'plugin_v2' to be valid");
        validate_plugin_id("a").expect("expected 'a' to be valid");
    }

    #[test]
    fn invalid_plugin_ids() {
        assert!(validate_plugin_id("").is_err());
        assert!(validate_plugin_id("Has Spaces").is_err());
        assert!(validate_plugin_id("UpperCase").is_err());
        assert!(validate_plugin_id("path/traversal").is_err());
        assert!(validate_plugin_id(&"a".repeat(MAX_PLUGIN_ID_LEN + 1)).is_err());
    }

    #[test]
    fn load_valid_manifest() {
        let dir = std::env::temp_dir().join(format!("fc-plugin-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create dir");
        let path = dir.join("plugin.json");
        std::fs::write(
            &path,
            serde_json::json!({
                "id": "my-plugin",
                "name": "My Plugin",
                "version": "1.0.0",
                "description": "A test plugin",
                "channels": ["web"],
                "tools": ["my.tool"]
            })
            .to_string(),
        )
        .expect("write");

        let manifest = load_plugin_manifest(&path).expect("should load");
        assert_eq!(manifest.id, "my-plugin");
        assert_eq!(manifest.name, "My Plugin");
        assert_eq!(manifest.version, "1.0.0");
        assert_eq!(manifest.channels, vec!["web"]);
        assert_eq!(manifest.tools, vec!["my.tool"]);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn load_missing_name_fails() {
        let dir = std::env::temp_dir().join(format!("fc-plugin-noname-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create dir");
        let path = dir.join("plugin.json");
        std::fs::write(
            &path,
            serde_json::json!({
                "id": "bad",
                "name": "",
                "version": "1.0.0"
            })
            .to_string(),
        )
        .expect("write");

        let _ = load_plugin_manifest(&path).expect_err("expected missing name to fail");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn load_missing_version_fails() {
        let dir = std::env::temp_dir().join(format!("fc-plugin-nover-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create dir");
        let path = dir.join("plugin.json");
        std::fs::write(
            &path,
            serde_json::json!({
                "id": "bad",
                "name": "Bad",
                "version": ""
            })
            .to_string(),
        )
        .expect("write");

        let _ = load_plugin_manifest(&path).expect_err("expected missing version to fail");
        let _ = std::fs::remove_dir_all(dir);
    }
}
