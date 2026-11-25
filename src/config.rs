//! Configuration management for eomoe.
//!
//! Precedence: CLI flag > YAML config file > built-in default.

use serde::Deserialize;
use std::path::{Path, PathBuf};

use crate::cli::ServeArgs;
use crate::model::ModelConfig;

/// Top-level YAML config. Every key is optional; unknown keys are ignored.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct EomoeConfig {
    pub model: ModelSection,
    pub server: ServerSection,
    pub gpu: GpuSection,
    pub logging: LoggingSection,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct ModelSection {
    pub path: Option<PathBuf>,
    pub cache_size_mib: Option<u32>,
    pub num_experts: Option<usize>,
    pub expert_dim: Option<usize>,
    pub num_layers: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct ServerSection {
    pub host: Option<String>,
    pub port: Option<u16>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct GpuSection {
    pub device_id: Option<u32>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LoggingSection {
    pub level: Option<String>,
}

/// Fully resolved settings for `serve`.
#[derive(Debug)]
pub struct ServeSettings {
    pub model: ModelConfig,
    pub host: String,
    pub port: u16,
    pub device: u32,
}

impl EomoeConfig {
    /// Load configuration from a YAML file
    pub fn load(path: &Path) -> Result<Self, String> {
        let content =
            std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        serde_yaml::from_str(&content).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Merge CLI args over this config, falling back to defaults.
    pub fn resolve_serve(&self, cli: &ServeArgs) -> Result<ServeSettings, String> {
        let m = &self.model;
        let path = cli
            .model
            .clone()
            .or_else(|| m.path.clone())
            .ok_or("no model path: pass --model or set model.path in the config file")?;

        Ok(ServeSettings {
            model: ModelConfig {
                model_path: path.to_string_lossy().into_owned(),
                cache_size_mib: cli.cache_size.or(m.cache_size_mib).unwrap_or(1024),
                num_experts: m.num_experts.unwrap_or(128),
                expert_dim: m.expert_dim.unwrap_or(1024),
                num_layers: m.num_layers.unwrap_or(24),
            },
            host: cli
                .host
                .clone()
                .or_else(|| self.server.host.clone())
                .unwrap_or_else(|| "0.0.0.0".to_string()),
            port: cli.port.or(self.server.port).unwrap_or(8080),
            device: cli.device.or(self.gpu.device_id).unwrap_or(0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_overrides_config_overrides_defaults() {
        let cfg: EomoeConfig = serde_yaml::from_str(
            "model:\n  path: /cfg/model.gguf\n  num_experts: 8\nserver:\n  host: 127.0.0.1\n  port: 9000\n",
        )
        .unwrap();
        let cli = ServeArgs {
            model: None,
            host: None,
            port: Some(7000),
            cache_size: None,
            device: None,
        };
        let s = cfg.resolve_serve(&cli).unwrap();
        assert_eq!(s.model.model_path, "/cfg/model.gguf"); // from config
        assert_eq!(s.model.num_experts, 8); // from config
        assert_eq!(s.host, "127.0.0.1"); // from config
        assert_eq!(s.port, 7000); // CLI beats config
        assert_eq!(s.model.cache_size_mib, 1024); // default
        assert_eq!(s.device, 0); // default

        // No model path anywhere is an error, not a silent default.
        assert!(EomoeConfig::default().resolve_serve(&cli).is_err());
    }

    #[test]
    fn example_config_parses() {
        let cfg: EomoeConfig = serde_yaml::from_str(include_str!("../config.example.yaml")).unwrap();
        assert!(cfg.model.path.is_some());
    }
}
