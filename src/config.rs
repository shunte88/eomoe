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
    pub api_key: Option<String>,
    pub api_key_file: Option<PathBuf>,
    pub ssl_key_file: Option<PathBuf>,
    pub ssl_cert_file: Option<PathBuf>,
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
    /// Empty: no auth.
    pub api_keys: Vec<String>,
    /// (key, cert): serve HTTPS.
    pub tls: Option<(PathBuf, PathBuf)>,
}

impl EomoeConfig {
    pub fn load(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        serde_yaml::from_str(&content).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn resolve_serve(&self, cli: &ServeArgs) -> Result<ServeSettings, String> {
        let (m, s) = (&self.model, &self.server);
        // As llama-server: comma-separated keys, plus a file's lines that aren't `#` comments.
        let mut api_keys: Vec<String> = cli.api_key.as_ref().or(s.api_key.as_ref()).map_or(vec![], |k| k.split(',').filter(|k| !k.is_empty()).map(str::to_owned).collect());
        if let Some(f) = cli.api_key_file.as_ref().or(s.api_key_file.as_ref()) {
            let text = std::fs::read_to_string(f).map_err(|e| format!("{}: {e}", f.display()))?;
            api_keys.extend(text.lines().filter(|l| !l.is_empty() && !l.starts_with('#')).map(str::to_owned));
        }
        let tls = match (cli.ssl_key_file.clone().or_else(|| s.ssl_key_file.clone()), cli.ssl_cert_file.clone().or_else(|| s.ssl_cert_file.clone())) {
            (Some(k), Some(c)) => Some((k, c)),
            (None, None) => None,
            _ => return Err("HTTPS needs both --ssl-key-file and --ssl-cert-file".into()),
        };
        Ok(ServeSettings {
            model: cli.model.clone().or_else(|| m.path.clone()).unwrap_or_else(crate::cli::default_model),
            ctx_size: cli.ctx_size.or(m.ctx_size).unwrap_or(DEFAULT_CTX),
            cache_mib: cli.cache_mib.or(m.cache_mib),
            host: cli.host.clone().or_else(|| s.host.clone()).unwrap_or_else(|| "127.0.0.1".into()),
            port: cli.port.or(s.port).unwrap_or(8080),
            alias: cli.alias.clone().or_else(|| s.alias.clone()).unwrap_or_else(|| "local-code".into()),
            parallel: cli.parallel.or(s.parallel).unwrap_or(4),
            api_keys,
            tls,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_overrides_config_overrides_defaults() {
        let cfg: EomoeConfig = serde_yaml::from_str("model:\n  path: /cfg/model.gguf\n  cache_mib: 2000\nserver:\n  host: 0.0.0.0\n  port: 9000\n").unwrap();
        let cli = ServeArgs { model: None, host: None, port: Some(7000), ctx_size: None, cache_mib: None, parallel: None, alias: None, api_key: None, api_key_file: None, ssl_key_file: None, ssl_cert_file: None };
        let s = cfg.resolve_serve(&cli).unwrap();
        assert_eq!(s.model, PathBuf::from("/cfg/model.gguf")); // config
        assert_eq!(s.cache_mib, Some(2000)); // config
        assert_eq!(s.host, "0.0.0.0"); // config
        assert_eq!(s.port, 7000); // CLI beats config
        assert_eq!((s.ctx_size, s.alias.as_str()), (DEFAULT_CTX, "local-code")); // defaults
        assert_eq!(EomoeConfig::default().resolve_serve(&cli).unwrap().cache_mib, None);
    }

    /// Keys as llama-server reads them: CSV from the flag plus the file's non-comment lines.
    #[test]
    fn api_keys_from_flag_and_file() {
        let f = std::env::temp_dir().join(format!("eomoe-keys-{}", std::process::id()));
        std::fs::write(&f, "# team keys\nk3\n\nk4\n").unwrap();
        let cli = ServeArgs { model: None, host: None, port: None, ctx_size: None, cache_mib: None, parallel: None, alias: None, api_key: Some("k1,,k2".into()), api_key_file: Some(f.clone()), ssl_key_file: None, ssl_cert_file: Some("c.pem".into()) };
        let cfg = EomoeConfig::default();
        assert!(cfg.resolve_serve(&cli).is_err(), "cert without key");
        let cli = ServeArgs { ssl_cert_file: None, ..cli };
        assert_eq!(cfg.resolve_serve(&cli).unwrap().api_keys, ["k1", "k2", "k3", "k4"]);
        std::fs::remove_file(f).ok();
    }

    #[test]
    fn example_config_parses() {
        let cfg: EomoeConfig = serde_yaml::from_str(include_str!("../config.example.yaml")).unwrap();
        assert!(cfg.model.path.is_some());
    }
}
