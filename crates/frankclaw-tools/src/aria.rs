#![forbid(unsafe_code)]

//! ARIA accessibility tree snapshot utilities.
//!
//! Converts Chrome DevTools Protocol `Accessibility.getFullAXTree` responses
//! into a compact, indented text representation with element references.

use std::{collections::HashMap, fmt::Write as _};

/// Reference to an ARIA element by role and name.
#[derive(Debug, Clone)]
pub struct RoleRef {
    pub role: String,
    pub name: String,
    /// Disambiguator when multiple elements share the same role+name.
    pub nth: Option<usize>,
}

/// Map from ref ID (e.g. "e1") to the role reference.
pub type RoleRefMap = HashMap<String, RoleRef>;

/// Options for building an ARIA snapshot.
pub struct AriaSnapshotOptions {
    /// Only include interactive elements (buttons, links, inputs, etc.).
    pub interactive_only: bool,
    /// Maximum tree depth to traverse.
    pub max_depth: usize,
}

impl Default for AriaSnapshotOptions {
    fn default() -> Self {
        Self {
            interactive_only: false,
            max_depth: 20,
        }
    }
}

/// Roles that represent interactive elements.
const INTERACTIVE_ROLES: &[&str] = &[
    "button",
    "link",
    "textbox",
    "checkbox",
    "radio",
    "combobox",
    "listbox",
    "menuitem",
    "menuitemcheckbox",
    "menuitemradio",
    "option",
    "searchbox",
    "slider",
    "spinbutton",
    "switch",
    "tab",
    "treeitem",
];

/// Roles that carry content worth showing.
const CONTENT_ROLES: &[&str] = &[
    "heading",
    "cell",
    "columnheader",
    "rowheader",
    "listitem",
    "img",
    "figure",
    "status",
    "alert",
    "dialog",
    "tooltip",
    "banner",
    "navigation",
    "main",
    "complementary",
    "contentinfo",
    "form",
    "region",
    "article",
    "document",
];

/// Structural roles that are typically unnamed wrappers.
const STRUCTURAL_ROLES: &[&str] = &[
    "generic",
    "group",
    "list",
    "table",
    "row",
    "grid",
    "gridcell",
    "presentation",
    "none",
    "separator",
    "toolbar",
    "menu",
    "menubar",
    "tablist",
    "tabpanel",
    "tree",
    "treegrid",
    "directory",
    "feed",
    "log",
    "marquee",
    "timer",
    "application",
    "paragraph",
    "blockquote",
    "section",
    "WebArea",
    "RootWebArea",
];

fn is_interactive(role: &str) -> bool {
    INTERACTIVE_ROLES.contains(&role)
}

fn is_content(role: &str) -> bool {
    CONTENT_ROLES.contains(&role)
}

fn is_structural(role: &str) -> bool {
    STRUCTURAL_ROLES.contains(&role)
}

/// A processed node from the accessibility tree.
#[derive(Debug)]
struct AXNode {
    node_id: String,
    role: String,
    name: String,
    value: String,
    parent_id: Option<String>,
    children_ids: Vec<String>,
    ignored: bool,
}

fn extract_ax_node(node: &serde_json::Value) -> AXNode {
    let node_id = node["nodeId"].as_str().unwrap_or_default().to_string();
    let role = node["role"]["value"]
        .as_str()
        .unwrap_or("generic")
        .to_string();
    let name = node["name"]["value"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    let value = node["value"]["value"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    let parent_id = node["parentId"].as_str().map(String::from);
    let children_ids = node["childIds"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let ignored = node["ignored"].as_bool().unwrap_or(false);

    AXNode {
        node_id,
        role,
        name,
        value,
        parent_id,
        children_ids,
        ignored,
    }
}

/// Build a compact ARIA tree snapshot from CDP `Accessibility.getFullAXTree` nodes.
///
/// Returns the indented tree text and a map of element references.
pub fn build_role_snapshot(
    nodes: &[serde_json::Value],
    options: &AriaSnapshotOptions,
) -> (String, RoleRefMap) {
    if nodes.is_empty() {
        return (String::new(), RoleRefMap::new());
    }

    // Parse all nodes
    let ax_nodes: Vec<AXNode> = nodes.iter().map(extract_ax_node).collect();
    let node_map: HashMap<&str, &AXNode> =
        ax_nodes.iter().map(|n| (n.node_id.as_str(), n)).collect();

    // Find root nodes (nodes without a parent or with a parent not in the set)
    let roots: Vec<&str> = ax_nodes
        .iter()
        .filter(|n| !n.ignored)
        .filter(|n| {
            n.parent_id
                .as_deref()
                .is_none_or(|pid| !node_map.contains_key(pid))
        })
        .map(|n| n.node_id.as_str())
        .collect();

    let mut output = String::new();
    let mut refs = RoleRefMap::new();
    let mut ref_counter: usize = 0;
    // Track role+name occurrences for nth disambiguation
    let mut role_name_counts: HashMap<(String, String), usize> = HashMap::new();

    // Pre-count role+name occurrences
    for node in &ax_nodes {
        if node.ignored {
            continue;
        }
        if !node.name.is_empty() && (is_interactive(&node.role) || is_content(&node.role)) {
            *role_name_counts
                .entry((node.role.clone(), node.name.clone()))
                .or_insert(0) += 1;
        }
    }

    let mut render_state = RenderState {
        output: &mut output,
        refs: &mut refs,
        ref_counter: &mut ref_counter,
        role_name_counts: &role_name_counts,
    };

    for root_id in &roots {
        render_node(root_id, &node_map, options, 0, &mut render_state);
    }

    (output, refs)
}

struct RenderState<'render> {
    output: &'render mut String,
    refs: &'render mut RoleRefMap,
    ref_counter: &'render mut usize,
    role_name_counts: &'render HashMap<(String, String), usize>,
}

fn render_node(
    node_id: &str,
    node_map: &HashMap<&str, &AXNode>,
    options: &AriaSnapshotOptions,
    depth: usize,
    render_state: &mut RenderState<'_>,
) {
    if depth > options.max_depth {
        return;
    }

    let Some(node) = node_map.get(node_id) else {
        return;
    };

    if node.ignored {
        return;
    }

    let role = &node.role;
    let name = &node.name;
    let is_inter = is_interactive(role);
    let is_cont = is_content(role);
    let is_struct = is_structural(role);

    // In interactive_only mode, skip non-interactive non-structural nodes.
    // Still recurse through structural nodes to find nested interactive ones.
    let should_print = if options.interactive_only {
        is_inter
    } else {
        is_inter || is_cont || (is_struct && !name.is_empty())
    };

    if should_print {
        let indent = "  ".repeat(depth);
        *render_state.ref_counter += 1;
        let ref_id = format!("e{}", *render_state.ref_counter);

        let nth = if name.is_empty() {
            None
        } else {
            let key = (role.clone(), name.clone());
            let count = render_state
                .role_name_counts
                .get(&key)
                .copied()
                .unwrap_or(1);
            (count > 1).then(|| {
                // Count how many we've assigned so far
                let assigned = render_state
                    .refs
                    .values()
                    .filter(|r| r.role == *role && r.name == *name)
                    .count();
                assigned + 1
            })
        };

        render_state.refs.insert(
            ref_id.clone(),
            RoleRef {
                role: role.clone(),
                name: name.clone(),
                nth,
            },
        );

        let name_part = if name.is_empty() {
            String::new()
        } else {
            format!(" \"{}\"", truncate_name(name, 80))
        };

        let value_part = if node.value.is_empty() {
            String::new()
        } else {
            format!(" value=\"{}\"", truncate_name(&node.value, 40))
        };

        let nth_part = nth.map(|n| format!(" nth={n}")).unwrap_or_default();

        let _ = writeln!(
            render_state.output,
            "{indent}- {role}{name_part}{value_part} [ref={ref_id}]{nth_part}"
        );
    }

    // Recurse into children
    for child_id in &node.children_ids {
        render_node(
            child_id,
            node_map,
            options,
            if should_print { depth + 1 } else { depth },
            render_state,
        );
    }
}

fn truncate_name(name: &str, max_len: usize) -> String {
    if name.len() <= max_len {
        name.to_string()
    } else {
        format!("{}...", &name[..max_len.saturating_sub(3)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_nodes() -> Vec<serde_json::Value> {
        serde_json::from_str(
            r#"[
                {
                    "nodeId": "1",
                    "role": {"value": "RootWebArea"},
                    "name": {"value": "Test Page"},
                    "childIds": ["2", "3", "4"]
                },
                {
                    "nodeId": "2",
                    "role": {"value": "heading"},
                    "name": {"value": "Welcome"},
                    "parentId": "1",
                    "childIds": []
                },
                {
                    "nodeId": "3",
                    "role": {"value": "button"},
                    "name": {"value": "Submit"},
                    "parentId": "1",
                    "childIds": []
                },
                {
                    "nodeId": "4",
                    "role": {"value": "textbox"},
                    "name": {"value": "Email"},
                    "value": {"value": "test@example.com"},
                    "parentId": "1",
                    "childIds": []
                }
            ]"#,
        )
        .expect("sample accessibility tree JSON should deserialize")
    }

    #[test]
    fn build_role_snapshot_assigns_refs() {
        let nodes = sample_nodes();
        let options = AriaSnapshotOptions::default();
        let (tree, refs) = build_role_snapshot(&nodes, &options);

        // Should have refs for RootWebArea (named structural), heading, button, textbox
        assert_eq!(refs.len(), 4);
        assert!(tree.contains("heading"));
        assert!(tree.contains("button"));
        assert!(tree.contains("textbox"));
        assert!(tree.contains("[ref=e1]"));
        assert!(tree.contains("[ref=e2]"));
        assert!(tree.contains("[ref=e3]"));
        assert!(tree.contains("[ref=e4]"));
    }

    #[test]
    fn interactive_only_filters_non_interactive() {
        let nodes = sample_nodes();
        let options = AriaSnapshotOptions {
            interactive_only: true,
            max_depth: 20,
        };
        let (tree, refs) = build_role_snapshot(&nodes, &options);

        // Should only have button and textbox (not heading)
        assert_eq!(refs.len(), 2);
        assert!(!tree.contains("heading"));
        assert!(tree.contains("button"));
        assert!(tree.contains("textbox"));
    }

    #[test]
    fn max_depth_truncates() {
        let nodes = sample_nodes();
        let options = AriaSnapshotOptions {
            interactive_only: false,
            max_depth: 0,
        };
        let (tree, refs) = build_role_snapshot(&nodes, &options);

        // Depth 0 means only root node (which is structural with a name)
        // Children at depth 1 should not be rendered
        assert!(refs.is_empty() || refs.len() <= 1);
        // The root "RootWebArea" is structural with a name, so it shows
        assert!(!tree.contains("button"));
    }

    #[test]
    fn duplicate_role_name_gets_nth() {
        let nodes: Vec<serde_json::Value> = serde_json::from_str(
            r#"[
                {
                    "nodeId": "1",
                    "role": {"value": "RootWebArea"},
                    "name": {"value": ""},
                    "childIds": ["2", "3"]
                },
                {
                    "nodeId": "2",
                    "role": {"value": "button"},
                    "name": {"value": "Save"},
                    "parentId": "1",
                    "childIds": []
                },
                {
                    "nodeId": "3",
                    "role": {"value": "button"},
                    "name": {"value": "Save"},
                    "parentId": "1",
                    "childIds": []
                }
            ]"#,
        )
        .expect("duplicate role-name tree JSON should deserialize");

        let options = AriaSnapshotOptions::default();
        let (tree, refs) = build_role_snapshot(&nodes, &options);

        assert_eq!(refs.len(), 2);
        assert!(tree.contains("nth="));
        // Both should have nth disambiguation
        assert_eq!(refs.values().filter(|r| r.nth.is_some()).count(), 2);
    }

    #[test]
    fn empty_nodes_returns_empty() {
        let (tree, refs) = build_role_snapshot(&[], &AriaSnapshotOptions::default());
        assert!(tree.is_empty());
        assert!(refs.is_empty());
    }

    #[test]
    fn ignored_nodes_are_skipped() {
        let nodes: Vec<serde_json::Value> = serde_json::from_str(
            r#"[
                {
                    "nodeId": "1",
                    "role": {"value": "RootWebArea"},
                    "name": {"value": ""},
                    "childIds": ["2"]
                },
                {
                    "nodeId": "2",
                    "role": {"value": "button"},
                    "name": {"value": "Hidden"},
                    "parentId": "1",
                    "childIds": [],
                    "ignored": true
                }
            ]"#,
        )
        .expect("ignored-node tree JSON should deserialize");

        let (tree, refs) = build_role_snapshot(&nodes, &AriaSnapshotOptions::default());
        assert!(refs.is_empty());
        assert!(!tree.contains("Hidden"));
    }

    #[test]
    fn value_is_shown_for_inputs() {
        let nodes = sample_nodes();
        let options = AriaSnapshotOptions::default();
        let (tree, _refs) = build_role_snapshot(&nodes, &options);
        assert!(tree.contains("value=\"test@example.com\""));
    }
}
