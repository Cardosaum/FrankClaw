//! Plugin lifecycle management: enable/disable, state persistence.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::discovery::{DiscoveredPlugin, PluginOrigin};
use crate::manifest::PluginManifest;

/// Persisted state for a single plugin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginState {
    pub enabled: bool,
}

impl Default for PluginState {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// A plugin with its runtime state.
#[derive(Debug, Clone)]
pub struct PluginRecord {
    pub manifest: PluginManifest,
    pub path: PathBuf,
    pub enabled: bool,
    pub origin: PluginOrigin,
}

/// Manages discovered plugins with enable/disable state.
pub struct PluginManager {
    plugins: HashMap<String, PluginRecord>,
}

impl PluginManager {
    /// Create a manager from discovered plugins and persisted state.
    pub fn new(discovered: Vec<DiscoveredPlugin>, state: &HashMap<String, PluginState>) -> Self {
        let mut plugins = HashMap::new();
        for dp in discovered {
            let enabled = state.get(&dp.manifest.id).is_none_or(|s| s.enabled);
            plugins.insert(
                dp.manifest.id.clone(),
                PluginRecord {
                    manifest: dp.manifest,
                    path: dp.path,
                    enabled,
                    origin: dp.origin,
                },
            );
        }
        Self { plugins }
    }

    /// List all plugins sorted by id.
    pub fn list(&self) -> Vec<&PluginRecord> {
        let mut records: Vec<_> = self.plugins.values().collect();
        records.sort_by(|a, b| a.manifest.id.cmp(&b.manifest.id));
        records
    }

    /// Get a plugin by id.
    pub fn get(&self, id: &str) -> Option<&PluginRecord> {
        self.plugins.get(id)
    }

    /// Enable a plugin. Returns true if the plugin was found.
    pub fn enable(&mut self, id: &str) -> bool {
        if let Some(record) = self.plugins.get_mut(id) {
            record.enabled = true;
            true
        } else {
            false
        }
    }

    /// Disable a plugin. Returns true if the plugin was found.
    pub fn disable(&mut self, id: &str) -> bool {
        if let Some(record) = self.plugins.get_mut(id) {
            record.enabled = false;
            true
        } else {
            false
        }
    }

    /// Export current state for persistence.
    pub fn export_state(&self) -> HashMap<String, PluginState> {
        self.plugins
            .iter()
            .map(|(id, record)| {
                (
                    id.clone(),
                    PluginState {
                        enabled: record.enabled,
                    },
                )
            })
            .collect()
    }

    /// Number of plugins.
    pub fn count(&self) -> usize {
        self.plugins.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::PluginOrigin;

    fn make_discovered(id: &str, origin: PluginOrigin) -> DiscoveredPlugin {
        DiscoveredPlugin {
            manifest: PluginManifest {
                id: id.to_string(),
                name: format!("Plugin {id}"),
                version: "1.0.0".to_string(),
                description: None,
                channels: vec![],
                tools: vec![],
                config_schema: None,
            },
            path: PathBuf::from(format!("/plugins/{id}")),
            origin,
        }
    }

    #[test]
    fn new_defaults_enabled() {
        let plugins = vec![make_discovered("a", PluginOrigin::User)];
        let mgr = PluginManager::new(plugins, &HashMap::new());
        assert!(mgr.get("a").expect("plugin 'a' should exist").enabled);
    }

    #[test]
    fn new_respects_persisted_state() {
        let plugins = vec![make_discovered("a", PluginOrigin::User)];
        let mut state = HashMap::new();
        state.insert("a".to_string(), PluginState { enabled: false });
        let mgr = PluginManager::new(plugins, &state);
        assert!(!mgr.get("a").expect("plugin 'a' should exist").enabled);
    }

    #[test]
    fn enable_disable() {
        let plugins = vec![make_discovered("a", PluginOrigin::User)];
        let mut mgr = PluginManager::new(plugins, &HashMap::new());

        assert!(mgr.disable("a"));
        assert!(!mgr.get("a").expect("plugin 'a' should exist").enabled);

        assert!(mgr.enable("a"));
        assert!(mgr.get("a").expect("plugin 'a' should exist").enabled);

        assert!(!mgr.enable("nonexistent"));
        assert!(!mgr.disable("nonexistent"));
    }

    #[test]
    fn list_sorted() {
        let plugins = vec![
            make_discovered("zeta", PluginOrigin::User),
            make_discovered("alpha", PluginOrigin::Workspace),
        ];
        let mgr = PluginManager::new(plugins, &HashMap::new());
        let list = mgr.list();
        assert_eq!(list[0].manifest.id, "alpha");
        assert_eq!(list[1].manifest.id, "zeta");
    }

    #[test]
    fn export_state_roundtrip() {
        let plugins = vec![
            make_discovered("a", PluginOrigin::User),
            make_discovered("b", PluginOrigin::User),
        ];
        let mut mgr = PluginManager::new(plugins, &HashMap::new());
        mgr.disable("b");

        let state = mgr.export_state();
        assert!(state["a"].enabled);
        assert!(!state["b"].enabled);

        // Re-create from exported state.
        let plugins2 = vec![
            make_discovered("a", PluginOrigin::User),
            make_discovered("b", PluginOrigin::User),
        ];
        let mgr2 = PluginManager::new(plugins2, &state);
        assert!(mgr2.get("a").expect("plugin 'a' should exist").enabled);
        assert!(!mgr2.get("b").expect("plugin 'b' should exist").enabled);
    }

    #[test]
    fn count() {
        let plugins = vec![
            make_discovered("a", PluginOrigin::User),
            make_discovered("b", PluginOrigin::User),
        ];
        let mgr = PluginManager::new(plugins, &HashMap::new());
        assert_eq!(mgr.count(), 2);
    }
}
