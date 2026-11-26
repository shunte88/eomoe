//! Configuration for `serve`. Precedence: CLI flag / env var > YAML config file > default.

use serde::Deserialize;
use std::path::{Path, PathBuf};

use crate::cli::ServeArgs;

/// Fits a stock Claude Code prompt (~19K tokens); with the f16 KV cache it costs what 16K did in f32.
pub const DEFAULT_CTX: usize = 32768;

/// Top-level YAML config. Every key is optional; unknown keys are ignored.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct EomoeConfig {
    pub model: ModelSection,
    pub server: ServerSection,
    pub logging: LoggingSection,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct ModelSection {
    pub path: Option<PathBuf>,
    pub ctx_size: Option<usize>,
    pub cache_mib: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct ServerSection {
    pub host: Option<String>,
    pub port: Option<u16>,
    pub alias: Option<String>,
    pub parallel: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LoggingSection {
    pub level: Option<String>,
}

/// Fully resolved settings for `serve`.
#[derive(Debug)]
pub struct ServeSettings {
    pub model: PathBuf,
    pub ctx_size: usize,
    /// None: all free VRAM.
    pub cache_mib: Option<usize>,
    pub host: String,
    pub port: u16,
    pub alias: String,
    pub parallel: usize,
}

impl EomoeConfig {
    pub fn load(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        serde_yaml::from_str(&content).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn resolve_serve(&self, cli: &ServeArgs) -> ServeSettings {
        let (m, s) = (&self.model, &self.server);
        ServeSettings {
            model: cli.model.clone().or_else(|| m.path.clone()).unwrap_or_else(crate::cli::default_model),
            ctx_size: cli.ctx_size.or(m.ctx_size).unwrap_or(DEFAULT_CTX),
            cache_mib: cli.cache_mib.or(m.cache_mib),
            host: cli.host.clone().or_else(|| s.host.clone()).unwrap_or_else(|| "127.0.0.1".into()),
            port: cli.port.or(s.port).unwrap_or(8080),
            alias: cli.alias.clone().or_else(|| s.alias.clone()).unwrap_or_else(|| "local-code".into()),
            parallel: cli.parallel.or(s.parallel).unwrap_or(4),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_overrides_config_overrides_defaults() {
        let cfg: EomoeConfig = serde_yaml::from_str("model:\n  path: /cfg/model.gguf\n  cache_mib: 2000\nserver:\n  host: 0.0.0.0\n  port: 9000\n").unwrap();
        let cli = ServeArgs { model: None, host: None, port: Some(7000), ctx_size: None, cache_mib: None, parallel: None, alias: None };
        let s = cfg.resolve_serve(&cli);
        assert_eq!(s.model, PathBuf::from("/cfg/model.gguf")); // config
        assert_eq!(s.cache_mib, Some(2000)); // config
        assert_eq!(s.host, "0.0.0.0"); // config
        assert_eq!(s.port, 7000); // CLI beats config
        assert_eq!((s.ctx_size, s.alias.as_str()), (DEFAULT_CTX, "local-code")); // defaults
        assert_eq!(EomoeConfig::default().resolve_serve(&cli).cache_mib, None);
    }

    #[test]
    fn example_config_parses() {
        let cfg: EomoeConfig = serde_yaml::from_str(include_str!("../config.example.yaml")).unwrap();
        assert!(cfg.model.path.is_some());
    }
}
