//! Browser profile management: named CDP configurations with port allocation.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use frankclaw_core::error::{FrankClawError, Result};

/// Port range reserved for browser profiles (avoids gateway 18789-18799).
pub const CDP_PORT_RANGE_START: u16 = 18800;
pub const CDP_PORT_RANGE_END: u16 = 18899;

/// Maximum profile name length.
const MAX_PROFILE_NAME_LEN: usize = 64;

/// A named browser profile with optional CDP endpoint configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserProfile {
    pub name: String,
    pub cdp_port: Option<u16>,
    pub cdp_url: Option<String>,
    #[serde(default)]
    pub color: String,
}

impl BrowserProfile {
    /// Derive the CDP base URL for this profile.
    pub fn base_url(&self) -> String {
        if let Some(url) = &self.cdp_url {
            url.clone()
        } else if let Some(port) = self.cdp_port {
            format!("http://127.0.0.1:{port}/")
        } else {
            "http://127.0.0.1:9222/".to_string()
        }
    }
}

/// Validate a browser profile name.
///
/// Rules: lowercase alphanumeric + hyphens, 1-64 chars, must not start/end with hyphen.
pub fn validate_profile_name(name: &str) -> Result<()> {
    if name.is_empty() || name.len() > MAX_PROFILE_NAME_LEN {
        return Err(FrankClawError::ConfigValidation {
            msg: format!(
                "browser profile name must be 1-{MAX_PROFILE_NAME_LEN} characters, got {}",
                name.len()
            ),
        });
    }
    if name.starts_with('-') || name.ends_with('-') {
        return Err(FrankClawError::ConfigValidation {
            msg: "browser profile name must not start or end with a hyphen".into(),
        });
    }
    if !name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-') {
        return Err(FrankClawError::ConfigValidation {
            msg: "browser profile name must contain only lowercase alphanumeric characters and hyphens".into(),
        });
    }
    Ok(())
}

/// Extract CDP ports that are already in use from a set of profiles.
pub fn get_used_ports(profiles: &[BrowserProfile]) -> HashSet<u16> {
    let mut ports = HashSet::new();
    for profile in profiles {
        if let Some(port) = profile.cdp_port {
            ports.insert(port);
        }
        if let Some(url) = &profile.cdp_url {
            if let Some(port) = extract_port_from_url(url) {
                ports.insert(port);
            }
        }
    }
    ports
}

/// Parse the port from a CDP URL like `http://127.0.0.1:18805/`.
fn extract_port_from_url(url: &str) -> Option<u16> {
    url::Url::parse(url).ok()?.port()
}

/// Allocate the first free port in the CDP port range.
pub fn allocate_cdp_port(used_ports: &HashSet<u16>) -> Option<u16> {
    (CDP_PORT_RANGE_START..=CDP_PORT_RANGE_END).find(|port| !used_ports.contains(port))
}

/// Color palette for browser profiles (cycles on overflow).
const PROFILE_COLORS: &[&str] = &[
    "#4A90D9", "#50C878", "#E74C3C", "#F39C12", "#9B59B6",
    "#1ABC9C", "#E67E22", "#3498DB", "#E91E63", "#00BCD4",
];

/// Allocate a color for a new profile, cycling through the palette.
pub fn allocate_color(used_colors: &HashSet<String>) -> String {
    for color in PROFILE_COLORS {
        if !used_colors.contains(*color) {
            return (*color).to_string();
        }
    }
    // All colors used — cycle from the beginning.
    let index = used_colors.len() % PROFILE_COLORS.len();
    PROFILE_COLORS[index].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_profile_names() {
        assert!(validate_profile_name("default").is_ok());
        assert!(validate_profile_name("my-browser").is_ok());
        assert!(validate_profile_name("test123").is_ok());
        assert!(validate_profile_name("a").is_ok());
    }

    #[test]
    fn invalid_profile_names() {
        assert!(validate_profile_name("").is_err());
        assert!(validate_profile_name("-start").is_err());
        assert!(validate_profile_name("end-").is_err());
        assert!(validate_profile_name("UpperCase").is_err());
        assert!(validate_profile_name("has space").is_err());
        assert!(validate_profile_name("has.dot").is_err());
        let long_name = "a".repeat(MAX_PROFILE_NAME_LEN + 1);
        assert!(validate_profile_name(&long_name).is_err());
    }

    #[test]
    fn max_length_name_is_ok() {
        let name = "a".repeat(MAX_PROFILE_NAME_LEN);
        assert!(validate_profile_name(&name).is_ok());
    }

    #[test]
    fn port_allocation_returns_first_free() {
        let used = HashSet::new();
        assert_eq!(allocate_cdp_port(&used), Some(CDP_PORT_RANGE_START));

        let used: HashSet<u16> = [18800, 18801, 18802].into_iter().collect();
        assert_eq!(allocate_cdp_port(&used), Some(18803));
    }

    #[test]
    fn port_allocation_returns_none_when_full() {
        let used: HashSet<u16> = (CDP_PORT_RANGE_START..=CDP_PORT_RANGE_END).collect();
        assert_eq!(allocate_cdp_port(&used), None);
    }

    #[test]
    fn get_used_ports_extracts_from_profiles() {
        let profiles = vec![
            BrowserProfile {
                name: "one".into(),
                cdp_port: Some(18800),
                cdp_url: None,
                color: String::new(),
            },
            BrowserProfile {
                name: "two".into(),
                cdp_port: None,
                cdp_url: Some("http://127.0.0.1:18805/".into()),
                color: String::new(),
            },
        ];
        let ports = get_used_ports(&profiles);
        assert!(ports.contains(&18800));
        assert!(ports.contains(&18805));
        assert_eq!(ports.len(), 2);
    }

    #[test]
    fn color_cycling() {
        let empty: HashSet<String> = HashSet::new();
        let first = allocate_color(&empty);
        assert_eq!(first, PROFILE_COLORS[0]);

        let mut used = HashSet::new();
        for color in PROFILE_COLORS {
            used.insert(color.to_string());
        }
        let cycled = allocate_color(&used);
        assert_eq!(cycled, PROFILE_COLORS[0]);
    }

    #[test]
    fn profile_base_url_from_port() {
        let profile = BrowserProfile {
            name: "test".into(),
            cdp_port: Some(18810),
            cdp_url: None,
            color: String::new(),
        };
        assert_eq!(profile.base_url(), "http://127.0.0.1:18810/");
    }

    #[test]
    fn profile_base_url_from_url() {
        let profile = BrowserProfile {
            name: "test".into(),
            cdp_port: Some(18810),
            cdp_url: Some("http://remote:9222/".into()),
            color: String::new(),
        };
        assert_eq!(profile.base_url(), "http://remote:9222/");
    }

    #[test]
    fn profile_base_url_default() {
        let profile = BrowserProfile {
            name: "test".into(),
            cdp_port: None,
            cdp_url: None,
            color: String::new(),
        };
        assert_eq!(profile.base_url(), "http://127.0.0.1:9222/");
    }
}
